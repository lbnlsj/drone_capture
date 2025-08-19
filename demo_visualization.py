#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from multi_drone_env import MultiDroneEnvironment
import time


def run_demo():
    """运行可视化演示"""
    print("多无人机协同作战环境演示")
    print("=" * 40)
    
    # 创建环境
    env = MultiDroneEnvironment(num_friendly_drones=9, num_enemy_drones=3)
    obs, info = env.reset()
    
    print("环境信息:")
    print(f"- 我方无人机: {env.num_friendly_drones} 架")
    print(f"- 敌方无人机: {env.num_enemy_drones} 架")
    print(f"- 空间大小: {env.map_size}")
    print(f"- 观测维度: {obs.shape[0]}")
    
    print("\n目标覆盖需求:")
    for i, enemy in enumerate(env.enemy_drones):
        print(f"- 目标 {chr(65+i)}: 需要 {enemy.required_coverage} 架无人机")
    
    print("\n开始演示...")
    print("使用智能策略：优先分配到需求最大的目标")
    
    total_reward = 0
    step_count = 0
    
    for step in range(100):
        # 智能策略：基于需求优先级分配无人机
        action = np.zeros((env.num_friendly_drones, 3))
        
        # 为每架友军无人机分配目标
        for i, friendly in enumerate(env.friendly_drones):
            best_target = None
            best_priority = -1
            
            for enemy in env.enemy_drones:
                # 计算优先级：需求缺口 / 距离
                demand_gap = max(0, enemy.required_coverage - enemy.current_coverage)
                distance = friendly.distance_to(enemy)
                
                if distance > 0 and demand_gap > 0:
                    priority = demand_gap / (distance + 1)  # +1避免除零
                    
                    if priority > best_priority:
                        best_priority = priority
                        best_target = enemy
            
            # 如果没有需要覆盖的目标，选择最近的目标进行追踪
            if best_target is None:
                min_dist = float('inf')
                for enemy in env.enemy_drones:
                    dist = friendly.distance_to(enemy)
                    if dist < min_dist:
                        min_dist = dist
                        best_target = enemy
            
            # 计算移动方向
            if best_target:
                dx = best_target.x - friendly.x
                dy = best_target.y - friendly.y
                dz = best_target.z - friendly.z
                
                norm = np.sqrt(dx**2 + dy**2 + dz**2)
                if norm > 0:
                    # 添加一些随机性避免聚集
                    noise = np.random.normal(0, 0.1, 3)
                    direction = np.array([dx/norm, dy/norm, dz/norm]) + noise
                    direction = direction / np.linalg.norm(direction)
                    action[i] = direction
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # 每10步打印一次状态
        if step % 10 == 0:
            print(f"\n--- 步骤 {step} ---")
            print(f"奖励: {reward:.2f} (累计: {total_reward:.2f})")
            
            all_satisfied = True
            for status in info['coverage_status']:
                satisfied = "✓" if status['satisfied'] else "✗"
                print(f"目标 {chr(65 + status['enemy_id'])}: {status['current']}/{status['required']} {satisfied}")
                if not status['satisfied']:
                    all_satisfied = False
            
            print("拦截状态:")
            for intercept in info['intercept_status']:
                intercepted = "✓" if intercept['intercepted'] else "✗"
                print(f"目标 {chr(65 + intercept['enemy_id'])}: 距离 {intercept['min_distance']:.1f}m {intercepted}")
            
            if all_satisfied:
                print("🎉 所有目标覆盖达标!")
        
        # 渲染（每5步渲染一次以提高性能）
        if step % 5 == 0:
            env.render()
            plt.pause(0.1)
        
        if terminated:
            print(f"\n🎉 任务完成! 总步数: {step_count}, 总奖励: {total_reward:.2f}")
            break
        
        if truncated:
            print(f"\n⏰ 达到最大步数限制")
            break
    
    # 最终统计
    print(f"\n=== 最终统计 ===")
    print(f"总步数: {step_count}")
    print(f"总奖励: {total_reward:.2f}")
    print(f"平均奖励: {total_reward/step_count:.2f}")
    
    # 最终状态
    final_coverage = sum(status['satisfied'] for status in info['coverage_status'])
    final_intercept = sum(intercept['intercepted'] for intercept in info['intercept_status'])
    
    print(f"覆盖完成率: {final_coverage}/{len(info['coverage_status'])} ({final_coverage/len(info['coverage_status'])*100:.1f}%)")
    print(f"拦截完成率: {final_intercept}/{len(info['intercept_status'])} ({final_intercept/len(info['intercept_status'])*100:.1f}%)")
    
    # 保持图像显示
    print("\n按 Ctrl+C 退出...")
    try:
        while True:
            env.render()
            plt.pause(0.5)
    except KeyboardInterrupt:
        print("\n演示结束")
    
    env.close()


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()