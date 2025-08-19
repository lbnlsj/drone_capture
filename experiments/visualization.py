#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib.animation import FuncAnimation
import torch

from multi_drone_env import MultiDroneEnvironment
from algorithms.ham_dtan_maddpg import HAMDTANMADDPGAlgorithm

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')


class ExperimentVisualizer:
    """实验结果可视化器"""
    
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        self.colors = {
            'PPO': '#FF6B6B',
            'MADDPG': '#4ECDC4', 
            'HAM-DTAN-MADDPG': '#45B7D1'
        }
        
    def load_results(self, filename):
        """加载实验结果"""
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def plot_training_curves(self, save_path='figures/training_curves.png'):
        """绘制训练曲线对比"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Performance Comparison', fontsize=16, fontweight='bold')
        
        algorithms = ['PPO', 'MADDPG', 'HAM-DTAN-MADDPG']
        filenames = [
            'ppo_baseline_results.json',
            'maddpg_baseline_results.json', 
            'ham_dtan_maddpg_results.json'
        ]
        
        # 加载所有结果
        all_results = {}
        for alg, filename in zip(algorithms, filenames):
            try:
                results = self.load_results(filename)
                all_results[alg] = results
            except FileNotFoundError:
                print(f"警告: 找不到 {filename}")
                continue
        
        if not all_results:
            print("没有找到任何结果文件，请先运行训练脚本")
            return
        
        # 绘制奖励曲线
        ax1 = axes[0, 0]
        for alg, results in all_results.items():
            episodes = results['metrics']['episodes']
            rewards = results['metrics']['rewards']
            
            # 计算滑动平均
            window = min(50, len(rewards) // 10)
            if window > 1:
                smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
                smoothed_episodes = episodes[window-1:]
            else:
                smoothed_rewards = rewards
                smoothed_episodes = episodes
            
            ax1.plot(smoothed_episodes, smoothed_rewards, 
                    label=alg, color=self.colors[alg], linewidth=2)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Training Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 绘制成功率
        ax2 = axes[0, 1]
        for alg, results in all_results.items():
            episodes = results['metrics']['episodes']
            success_rates = results['metrics']['success_rates']
            
            # 计算滑动平均成功率
            window = min(100, len(success_rates) // 5)
            if window > 1:
                smoothed_success = np.convolve(success_rates, np.ones(window)/window, mode='valid')
                smoothed_episodes = episodes[window-1:]
            else:
                smoothed_success = success_rates
                smoothed_episodes = episodes
            
            ax2.plot(smoothed_episodes, smoothed_success, 
                    label=alg, color=self.colors[alg], linewidth=2)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Success Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 绘制覆盖率
        ax3 = axes[1, 0]
        for alg, results in all_results.items():
            episodes = results['metrics']['episodes']
            coverage_rates = results['metrics']['coverage_rates']
            
            window = min(100, len(coverage_rates) // 5)
            if window > 1:
                smoothed_coverage = np.convolve(coverage_rates, np.ones(window)/window, mode='valid')
                smoothed_episodes = episodes[window-1:]
            else:
                smoothed_coverage = coverage_rates
                smoothed_episodes = episodes
            
            ax3.plot(smoothed_episodes, smoothed_coverage, 
                    label=alg, color=self.colors[alg], linewidth=2)
        
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Coverage Rate')
        ax3.set_title('Target Coverage Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 绘制拦截率
        ax4 = axes[1, 1]
        for alg, results in all_results.items():
            episodes = results['metrics']['episodes']
            intercept_rates = results['metrics']['intercept_rates']
            
            window = min(100, len(intercept_rates) // 5)
            if window > 1:
                smoothed_intercept = np.convolve(intercept_rates, np.ones(window)/window, mode='valid')
                smoothed_episodes = episodes[window-1:]
            else:
                smoothed_intercept = intercept_rates
                smoothed_episodes = episodes
            
            ax4.plot(smoothed_episodes, smoothed_intercept, 
                    label=alg, color=self.colors[alg], linewidth=2)
        
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Intercept Rate')
        ax4.set_title('Target Intercept Rate')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_comparison(self, save_path='figures/performance_comparison.png'):
        """绘制性能对比图"""
        try:
            results = self.load_results('comparative_experiment_results.json')
            eval_results = results['evaluation_results']
        except FileNotFoundError:
            print("找不到对比实验结果文件")
            return
        
        # 准备数据
        algorithms = list(eval_results.keys())
        metrics = ['avg_reward', 'success_rate', 'coverage_rate', 'intercept_rate']
        metric_names = ['Average Reward', 'Success Rate', 'Coverage Rate', 'Intercept Rate']
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i // 2, i % 2]
            
            values = [eval_results[alg][metric] for alg in algorithms]
            colors = [self.colors[alg] for alg in algorithms]
            
            bars = ax.bar(algorithms, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel(name)
            ax.set_title(f'{name} Comparison')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 设置y轴范围
            if metric in ['success_rate', 'coverage_rate', 'intercept_rate']:
                ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_attention_analysis(self, save_path='figures/attention_analysis.png'):
        """绘制注意力机制分析"""
        # 创建环境和算法实例
        env = MultiDroneEnvironment(num_friendly_drones=9, num_enemy_drones=3)
        algorithm = HAMDTANMADDPGAlgorithm()
        
        # 尝试加载训练好的模型
        try:
            algorithm.load('models/ham_dtan_maddpg.pth')
            print("成功加载训练好的模型")
        except FileNotFoundError:
            print("未找到训练好的模型，使用随机初始化的模型进行演示")
        
        # 运行一个episode收集注意力数据
        obs, _ = env.reset()
        algorithm.reset_memory()
        
        attention_data = []
        spatial_weights = []
        temporal_weights = []
        task_weights = []
        
        for step in range(50):
            # 获取动作和注意力信息
            actions, infos = algorithm.select_actions(obs, add_noise=False, return_info=True)
            
            # 收集注意力数据
            if infos and len(infos) > 0:
                info = infos[0]  # 使用第一个智能体的信息
                if 'attention_info' in info:
                    att_info = info['attention_info']
                    
                    if att_info['spatial_weights'] is not None:
                        spatial_weights.append(att_info['spatial_weights'][0].detach().cpu().numpy())
                    if att_info['temporal_weights'] is not None:
                        temporal_weights.append(att_info['temporal_weights'][0].detach().cpu().numpy())
                    if att_info['task_weights'] is not None:
                        task_weights.append(att_info['task_weights'][0].detach().cpu().numpy())
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(actions)
            
            if terminated or truncated:
                break
        
        # 绘制注意力权重演化
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Attention Mechanism Analysis', fontsize=16, fontweight='bold')
        
        # 空间注意力
        if spatial_weights:
            spatial_array = np.array(spatial_weights)
            im1 = axes[0].imshow(spatial_array.T, cmap='Blues', aspect='auto')
            axes[0].set_title('Spatial Attention Weights')
            axes[0].set_xlabel('Time Step')
            axes[0].set_ylabel('Object Index')
            plt.colorbar(im1, ax=axes[0])
        
        # 时间注意力
        if temporal_weights:
            temporal_array = np.array(temporal_weights)
            im2 = axes[1].imshow(temporal_array.T, cmap='Greens', aspect='auto')
            axes[1].set_title('Temporal Attention Weights')
            axes[1].set_xlabel('Time Step')
            axes[1].set_ylabel('History Index')
            plt.colorbar(im2, ax=axes[1])
        
        # 任务注意力
        if task_weights:
            task_array = np.array(task_weights)
            im3 = axes[2].imshow(task_array.T, cmap='Reds', aspect='auto')
            axes[2].set_title('Task Attention Weights')
            axes[2].set_xlabel('Time Step')
            axes[2].set_ylabel('Object Index')
            plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        env.close()
    
    def create_algorithm_demo_video(self, algorithm_path='models/ham_dtan_maddpg.pth', 
                                   save_path='figures/algorithm_demo.gif'):
        """创建算法演示视频"""
        # 创建环境和算法
        env = MultiDroneEnvironment(num_friendly_drones=9, num_enemy_drones=3)
        algorithm = HAMDTANMADDPGAlgorithm()
        
        # 加载模型
        try:
            algorithm.load(algorithm_path)
            print("成功加载训练好的模型")
        except FileNotFoundError:
            print("未找到训练好的模型，使用随机模型演示")
        
        # 设置记录
        frames = []
        obs, _ = env.reset()
        algorithm.reset_memory()
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        def animate(frame):
            nonlocal obs
            
            ax1.clear()
            ax2.clear()
            
            # 获取动作
            actions = algorithm.select_actions(obs, add_noise=False)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(actions)
            
            # 绘制2D视图
            ax1.set_xlim(0, 500)
            ax1.set_ylim(0, 500)
            ax1.set_xlabel('X Position')
            ax1.set_ylabel('Y Position')
            ax1.set_title(f'Multi-Drone Coordination (Step {frame})')
            ax1.grid(True, alpha=0.3)
            
            # 绘制我方无人机
            for i, drone in enumerate(env.friendly_drones):
                circle = plt.Circle((drone.x, drone.y), 8, color='blue', alpha=0.7)
                ax1.add_patch(circle)
                ax1.text(drone.x, drone.y, str(i), ha='center', va='center', 
                        fontsize=8, color='white', weight='bold')
            
            # 绘制敌方无人机
            colors = ['red', 'orange', 'purple']
            for i, drone in enumerate(env.enemy_drones):
                circle = plt.Circle((drone.x, drone.y), 12, color=colors[i], alpha=0.8)
                ax1.add_patch(circle)
                
                # 覆盖范围
                coverage_circle = plt.Circle((drone.x, drone.y), env.coverage_threshold, 
                                           fill=False, color=colors[i], linestyle='--', alpha=0.5)
                ax1.add_patch(coverage_circle)
                
                ax1.text(drone.x, drone.y-20, f'{chr(65+i)}\n{drone.current_coverage}/{drone.required_coverage}',
                        ha='center', va='center', fontsize=10, weight='bold')
            
            # 绘制性能指标
            ax2.set_xlim(0, 100)
            ax2.set_ylim(0, 1)
            ax2.set_xlabel('Metric')
            ax2.set_ylabel('Value')
            ax2.set_title('Performance Metrics')
            
            # 计算当前指标
            coverage_satisfied = sum(status['satisfied'] for status in info['coverage_status'])
            intercept_satisfied = sum(status['intercepted'] for status in info['intercept_status'])
            
            coverage_rate = coverage_satisfied / len(info['coverage_status'])
            intercept_rate = intercept_satisfied / len(info['intercept_status'])
            
            metrics = ['Coverage\nRate', 'Intercept\nRate', 'Overall\nSuccess']
            values = [coverage_rate, intercept_rate, (coverage_rate + intercept_rate) / 2]
            colors_bar = ['skyblue', 'lightgreen', 'gold']
            
            bars = ax2.bar(metrics, values, color=colors_bar, alpha=0.8)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
            
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            if terminated or truncated:
                return
        
        # 创建动画
        anim = FuncAnimation(fig, animate, frames=100, interval=200, repeat=False)
        
        # 保存动画
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            anim.save(save_path, writer='pillow', fps=5)
            print(f"动画已保存到: {save_path}")
        except Exception as e:
            print(f"保存动画时出错: {e}")
        
        plt.show()
        env.close()
    
    def generate_comprehensive_report(self):
        """生成综合实验报告"""
        print("生成综合实验报告...")
        
        # 创建figures目录
        os.makedirs('figures', exist_ok=True)
        
        # 1. 训练曲线对比
        print("1. 生成训练曲线对比图...")
        self.plot_training_curves()
        
        # 2. 性能对比
        print("2. 生成性能对比图...")
        self.plot_performance_comparison()
        
        # 3. 注意力机制分析
        print("3. 生成注意力机制分析图...")
        self.plot_attention_analysis()
        
        # 4. 算法演示视频
        print("4. 生成算法演示动画...")
        self.create_algorithm_demo_video()
        
        print("综合实验报告生成完成！")
        print("所有图表已保存到 figures/ 目录")


def main():
    """主函数"""
    print("多无人机协同作战实验可视化")
    print("=" * 50)
    
    visualizer = ExperimentVisualizer()
    
    try:
        # 生成所有可视化结果
        visualizer.generate_comprehensive_report()
        
    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()