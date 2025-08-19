#!/usr/bin/env python3

import numpy as np
import time
from multi_drone_env import MultiDroneEnvironment
from algorithms.ham_dtan_maddpg import HAMDTANMADDPGAlgorithm


def test_short_training():
    """测试短时间训练"""
    print("=" * 60)
    print("HAM-DTAN-MADDPG 短时间训练测试")
    print("=" * 60)
    
    # 创建环境
    env = MultiDroneEnvironment(num_friendly_drones=9, num_enemy_drones=3)
    
    print(f"环境信息:")
    print(f"- 观测空间: {env.observation_space.shape}")
    print(f"- 动作空间: {env.action_space.shape}")
    print(f"- 我方无人机: {env.num_friendly_drones}")
    print(f"- 敌方无人机: {env.num_enemy_drones}")
    
    # 创建算法
    algorithm = HAMDTANMADDPGAlgorithm(
        num_agents=9,
        obs_dim=78,
        action_dim=3,
        num_enemies=3
    )
    
    print(f"算法参数量: {algorithm.get_algorithm_info()['total_parameters']:,}")
    
    # 短时间训练
    num_episodes = 50
    print(f"\n开始训练 {num_episodes} 个episodes...")
    
    start_time = time.time()
    total_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        algorithm.reset_noise()
        algorithm.reset_memory()
        
        total_reward = 0
        episode_length = 0
        
        done = False
        while not done and episode_length < 100:  # 限制episode长度
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
        
        total_rewards.append(total_reward)
        
        # 更新网络
        if episode % 2 == 0 and len(algorithm.replay_buffer) > 100:
            losses = algorithm.update(batch_size=64)
        
        # 打印进度
        if episode % 10 == 0:
            recent_avg = np.mean(total_rewards[-10:]) if len(total_rewards) >= 10 else np.mean(total_rewards)
            print(f"Episode {episode}: 平均奖励 {recent_avg:.2f}, Episode长度 {episode_length}")
    
    training_time = time.time() - start_time
    
    print(f"\n✅ 训练完成!")
    print(f"⏱️ 总训练时间: {training_time:.2f}秒")
    print(f"📊 平均奖励: {np.mean(total_rewards):.2f}")
    print(f"📈 最后10个episodes平均奖励: {np.mean(total_rewards[-10:]):.2f}")
    print(f"🔢 经验池大小: {len(algorithm.replay_buffer)}")
    
    # 测试训练后的表现
    print("\n🎯 测试训练后表现...")
    test_rewards = []
    test_success = []
    
    for test_ep in range(10):
        obs, info = env.reset()
        algorithm.reset_memory()
        
        total_reward = 0
        episode_length = 0
        
        done = False
        while not done and episode_length < 200:
            actions = algorithm.select_actions(obs, add_noise=False)  # 确定性动作
            next_obs, reward, terminated, truncated, next_info = env.step(actions)
            
            total_reward += reward
            episode_length += 1
            done = terminated or truncated
            obs = next_obs
            info = next_info
        
        test_rewards.append(total_reward)
        test_success.append(terminated)
        
        if test_ep < 3:  # 打印前3个测试episode的详细信息
            final_coverage = sum(status['satisfied'] for status in info['coverage_status'])
            final_intercept = sum(status['intercepted'] for status in info['intercept_status'])
            print(f"  测试 {test_ep+1}: 奖励 {total_reward:.2f}, 覆盖 {final_coverage}/3, 拦截 {final_intercept}/3")
    
    print(f"\n📊 测试结果:")
    print(f"  - 平均奖励: {np.mean(test_rewards):.2f}")
    print(f"  - 成功率: {np.mean(test_success):.2%}")
    
    env.close()
    
    return {
        'training_time': training_time,
        'avg_training_reward': np.mean(total_rewards),
        'avg_test_reward': np.mean(test_rewards),
        'test_success_rate': np.mean(test_success)
    }


if __name__ == "__main__":
    try:
        results = test_short_training()
        print(f"\n🎉 测试完成! 结果: {results}")
    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()