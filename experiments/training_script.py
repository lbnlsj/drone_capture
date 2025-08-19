#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import json
from datetime import datetime

from multi_drone_env import MultiDroneEnvironment
from algorithms.ham_dtan_maddpg import HAMDTANMADDPGAlgorithm


class ExperimentTracker:
    """实验跟踪器"""
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        self.metrics = {
            'episodes': [],
            'rewards': [],
            'success_rates': [],
            'coverage_rates': [],
            'intercept_rates': [],
            'episode_lengths': [],
            'training_losses': []
        }
        
    def log_episode(self, episode, total_reward, success, coverage_rate, intercept_rate, episode_length):
        """记录单个episode的结果"""
        self.metrics['episodes'].append(episode)
        self.metrics['rewards'].append(total_reward)
        self.metrics['success_rates'].append(1.0 if success else 0.0)
        self.metrics['coverage_rates'].append(coverage_rate)
        self.metrics['intercept_rates'].append(intercept_rate)
        self.metrics['episode_lengths'].append(episode_length)
    
    def log_training_loss(self, losses):
        """记录训练损失"""
        self.metrics['training_losses'].append(losses)
    
    def get_recent_performance(self, window=100):
        """获取最近的性能指标"""
        if len(self.metrics['rewards']) < window:
            window = len(self.metrics['rewards'])
        
        if window == 0:
            return {}
        
        recent_rewards = self.metrics['rewards'][-window:]
        recent_success = self.metrics['success_rates'][-window:]
        recent_coverage = self.metrics['coverage_rates'][-window:]
        recent_intercept = self.metrics['intercept_rates'][-window:]
        
        return {
            'avg_reward': np.mean(recent_rewards),
            'success_rate': np.mean(recent_success),
            'coverage_rate': np.mean(recent_coverage),
            'intercept_rate': np.mean(recent_intercept),
            'episodes': window
        }
    
    def save_results(self, filepath):
        """保存实验结果"""
        results = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_episodes': len(self.metrics['episodes']),
            'metrics': self.metrics,
            'final_performance': self.get_recent_performance()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)


def evaluate_algorithm(algorithm, env, num_episodes=100, deterministic=True):
    """评估算法性能"""
    total_rewards = []
    success_rates = []
    coverage_rates = []
    intercept_rates = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        episode_length = 0
        
        # 重置算法状态
        if hasattr(algorithm, 'reset_noise'):
            algorithm.reset_noise()
        if hasattr(algorithm, 'reset_memory'):
            algorithm.reset_memory()
        
        done = False
        while not done and episode_length < 500:
            # 选择动作
            actions = algorithm.select_actions(obs, add_noise=not deterministic)
            
            # 执行动作
            next_obs, reward, terminated, truncated, info = env.step(actions)
            
            total_reward += reward
            episode_length += 1
            done = terminated or truncated
            obs = next_obs
        
        # 计算性能指标
        final_coverage = sum(status['satisfied'] for status in info['coverage_status'])
        final_intercept = sum(status['intercepted'] for status in info['intercept_status'])
        
        coverage_rate = final_coverage / len(info['coverage_status'])
        intercept_rate = final_intercept / len(info['intercept_status'])
        success = terminated  # 任务成功完成
        
        total_rewards.append(total_reward)
        success_rates.append(1.0 if success else 0.0)
        coverage_rates.append(coverage_rate)
        intercept_rates.append(intercept_rate)
        episode_lengths.append(episode_length)
    
    return {
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'success_rate': np.mean(success_rates),
        'coverage_rate': np.mean(coverage_rates),
        'intercept_rate': np.mean(intercept_rates),
        'avg_episode_length': np.mean(episode_lengths),
        'total_episodes': num_episodes
    }




def train_ham_dtan_maddpg(env, num_episodes=2000):
    """训练HAM-DTAN-MADDPG"""
    print("开始训练HAM-DTAN-MADDPG...")
    
    obs_dim = env.observation_space.shape[0]
    action_dim = 3
    num_agents = env.num_friendly_drones
    num_enemies = env.num_enemy_drones
    
    algorithm = HAMDTANMADDPGAlgorithm(
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_enemies=num_enemies
    )
    
    tracker = ExperimentTracker("HAM_DTAN_MADDPG")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        algorithm.reset_noise()
        algorithm.reset_memory()
        
        total_reward = 0
        episode_length = 0
        
        done = False
        while not done and episode_length < 500:
            # 选择动作
            actions = algorithm.select_actions(obs, add_noise=True)
            
            # 执行动作
            next_obs, reward, terminated, truncated, next_info = env.step(actions)
            
            # 存储经验
            algorithm.store_transition(obs, actions.flatten(), reward, next_obs, terminated or truncated)
            
            total_reward += reward
            episode_length += 1
            done = terminated or truncated
            obs = next_obs
            info = next_info
        
        # 计算性能指标
        final_coverage = sum(status['satisfied'] for status in info['coverage_status'])
        final_intercept = sum(status['intercepted'] for status in info['intercept_status'])
        
        coverage_rate = final_coverage / len(info['coverage_status'])
        intercept_rate = final_intercept / len(info['intercept_status'])
        success = terminated
        
        # 记录结果
        tracker.log_episode(episode, total_reward, success, coverage_rate, intercept_rate, episode_length)
        
        # 更新网络
        if episode % 2 == 0 and len(algorithm.replay_buffer) > 1000:
            losses = algorithm.update(batch_size=256)
            if losses:
                tracker.log_training_loss(losses)
        
        # 打印进度
        if episode % 100 == 0:
            recent_perf = tracker.get_recent_performance(100)
            print(f"Episode {episode}: "
                  f"Avg Reward: {recent_perf.get('avg_reward', 0):.2f}, "
                  f"Success Rate: {recent_perf.get('success_rate', 0):.2f}, "
                  f"Coverage Rate: {recent_perf.get('coverage_rate', 0):.2f}")
            
            # 打印算法信息
            if episode % 500 == 0:
                print(f"算法信息: {algorithm.get_algorithm_info()}")
    
    # 保存模型和结果
    algorithm.save('models/ham_dtan_maddpg.pth')
    tracker.save_results('results/ham_dtan_maddpg_results.json')
    
    return algorithm, tracker


def run_ham_dtan_experiment():
    """运行HAM-DTAN-MADDPG实验"""
    print("=" * 80)
    print("HAM-DTAN-MADDPG 多无人机协同强化学习实验")
    print("=" * 80)
    
    # 创建目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 创建环境
    env = MultiDroneEnvironment(num_friendly_drones=9, num_enemy_drones=3)
    
    print(f"环境信息:")
    print(f"- 观测空间: {env.observation_space.shape}")
    print(f"- 动作空间: {env.action_space.shape}")
    print(f"- 我方无人机: {env.num_friendly_drones}")
    print(f"- 敌方无人机: {env.num_enemy_drones}")
    print(f"- 3D空间: {env.map_size}")
    print()
    
    # 训练HAM-DTAN-MADDPG
    print("开始训练HAM-DTAN-MADDPG算法...")
    start_time = time.time()
    ham_dtan_agent, ham_dtan_tracker = train_ham_dtan_maddpg(env, num_episodes=2000)
    training_time = time.time() - start_time
    
    print(f"✅ 训练完成，耗时: {training_time:.2f}秒")
    print(f"📊 最终训练性能: {ham_dtan_tracker.get_recent_performance()}")
    
    # 评估算法
    print("\n🔍 开始最终评估...")
    eval_results = evaluate_algorithm(ham_dtan_agent, env, num_episodes=200, deterministic=True)
    
    # 打印详细结果
    print("\n" + "=" * 80)
    print("🎯 HAM-DTAN-MADDPG 实验结果")
    print("=" * 80)
    
    print(f"📈 性能指标:")
    print(f"  - 平均奖励: {eval_results['avg_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  - 任务成功率: {eval_results['success_rate']:.2%}")
    print(f"  - 目标覆盖率: {eval_results['coverage_rate']:.2%}")
    print(f"  - 拦截成功率: {eval_results['intercept_rate']:.2%}")
    print(f"  - 平均episode长度: {eval_results['avg_episode_length']:.1f}步")
    
    print(f"\n⏱️ 训练统计:")
    print(f"  - 总训练时间: {training_time:.1f}秒")
    print(f"  - 平均每episode: {training_time/2000:.3f}秒")
    print(f"  - 算法参数量: {ham_dtan_agent.get_algorithm_info()['total_parameters']:,}")
    
    # 保存结果
    final_results = {
        'algorithm': 'HAM-DTAN-MADDPG',
        'experiment_time': datetime.now().isoformat(),
        'training_time': training_time,
        'training_episodes': 2000,
        'evaluation_episodes': 200,
        'final_performance': ham_dtan_tracker.get_recent_performance(),
        'evaluation_results': eval_results,
        'algorithm_info': ham_dtan_agent.get_algorithm_info(),
        'environment_config': {
            'num_friendly_drones': env.num_friendly_drones,
            'num_enemy_drones': env.num_enemy_drones,
            'map_size': env.map_size,
            'obs_dim': env.observation_space.shape[0],
            'action_dim': env.action_space.shape
        }
    }
    
    # 保存到文件
    with open('results/ham_dtan_experiment_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\n💾 实验结果已保存到 results/ 目录")
    print(f"📁 模型已保存到 models/ 目录")
    
    env.close()
    
    return final_results


if __name__ == "__main__":
    try:
        results = run_ham_dtan_experiment()
        print("\n🎉 HAM-DTAN-MADDPG实验完成！")
        print("📊 可以运行 python experiments/visualization.py 查看可视化结果")
        print("🎮 可以运行 python demo_visualization.py 查看实时演示")
    except KeyboardInterrupt:
        print("\n⚠️ 实验被用户中断")
    except Exception as e:
        print(f"\n❌ 实验过程中出现错误: {e}")
        import traceback
        traceback.print_exc()