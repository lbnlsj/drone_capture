#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from multi_drone_env import MultiDroneEnvironment
import time


def test_basic_functionality():
    """测试环境基本功能"""
    print("=== 测试环境基本功能 ===")
    
    env = MultiDroneEnvironment(num_friendly_drones=9, num_enemy_drones=3)
    
    # 测试重置
    obs, info = env.reset()
    print(f"观测空间维度: {obs.shape}")
    print(f"动作空间维度: {env.action_space.shape}")
    print(f"初始状态信息: {len(info['coverage_status'])} 个敌方目标")
    
    # 测试动作
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\n步骤 {step + 1}:")
        print(f"奖励: {reward:.2f}")
        print(f"是否结束: {terminated}")
        
        # 打印覆盖状态
        for status in info['coverage_status']:
            print(f"目标 {chr(65 + status['enemy_id'])}: {status['current']}/{status['required']} "
                  f"({'✓' if status['satisfied'] else '✗'})")
        
        if terminated:
            print("任务完成!")
            break
    
    env.close()
    print("基本功能测试完成!\n")


def test_visualization():
    """测试可视化功能"""
    print("=== 测试可视化功能 ===")
    
    env = MultiDroneEnvironment(num_friendly_drones=9, num_enemy_drones=3)
    obs, info = env.reset()
    
    print("开始可视化演示(20步)...")
    
    for step in range(20):
        # 使用简单策略：让无人机向最近的敌方目标移动
        action = np.zeros((env.num_friendly_drones, 3))
        
        for i, friendly in enumerate(env.friendly_drones):
            # 找到最近的敌方无人机
            min_dist = float('inf')
            target_enemy = None
            
            for enemy in env.enemy_drones:
                dist = friendly.distance_to(enemy)
                if dist < min_dist:
                    min_dist = dist
                    target_enemy = enemy
            
            if target_enemy:
                # 计算移动方向
                dx = target_enemy.x - friendly.x
                dy = target_enemy.y - friendly.y
                dz = target_enemy.z - friendly.z
                
                # 归一化
                norm = np.sqrt(dx**2 + dy**2 + dz**2)
                if norm > 0:
                    action[i] = [dx/norm, dy/norm, dz/norm]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 渲染环境
        env.render()
        
        if step % 5 == 0:
            print(f"步骤 {step}: 奖励 {reward:.2f}")
            for status in info['coverage_status']:
                print(f"  目标 {chr(65 + status['enemy_id'])}: {status['current']}/{status['required']}")
        
        time.sleep(0.1)
        
        if terminated:
            print(f"任务在第 {step + 1} 步完成!")
            break
    
    print("可视化测试完成!\n")
    input("按回车键继续...")
    env.close()


def test_reward_system():
    """测试奖励系统"""
    print("=== 测试奖励系统 ===")
    
    env = MultiDroneEnvironment(num_friendly_drones=9, num_enemy_drones=3)
    obs, info = env.reset()
    
    print("测试不同策略的奖励:")
    
    # 策略1：随机动作
    env.reset()
    total_reward_random = 0
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward_random += reward
        if terminated:
            break
    
    print(f"随机策略总奖励: {total_reward_random:.2f}")
    
    # 策略2：向目标移动
    env.reset()
    total_reward_directed = 0
    for step in range(50):
        action = np.zeros((env.num_friendly_drones, 3))
        
        for i, friendly in enumerate(env.friendly_drones):
            # 优先向需要更多覆盖的目标移动
            best_target = None
            best_score = -1
            
            for enemy in env.enemy_drones:
                if enemy.current_coverage < enemy.required_coverage:
                    score = (enemy.required_coverage - enemy.current_coverage) / friendly.distance_to(enemy)
                    if score > best_score:
                        best_score = score
                        best_target = enemy
            
            if best_target:
                dx = best_target.x - friendly.x
                dy = best_target.y - friendly.y
                dz = best_target.z - friendly.z
                norm = np.sqrt(dx**2 + dy**2 + dz**2)
                if norm > 0:
                    action[i] = [dx/norm, dy/norm, dz/norm]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward_directed += reward
        if terminated:
            break
    
    print(f"有向策略总奖励: {total_reward_directed:.2f}")
    
    # 分析奖励组成
    env.reset()
    action = np.zeros((env.num_friendly_drones, 3))
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\n奖励系统分析:")
    print(f"当前总奖励: {reward:.2f}")
    for i, status in enumerate(info['coverage_status']):
        coverage_diff = status['current'] - status['required']
        print(f"目标 {chr(65 + i)}: 需要{status['required']}, 当前{status['current']}, 差值{coverage_diff}")
    
    env.close()
    print("奖励系统测试完成!\n")


def performance_test():
    """性能测试"""
    print("=== 性能测试 ===")
    
    env = MultiDroneEnvironment(num_friendly_drones=9, num_enemy_drones=3)
    
    # 测试环境重置时间
    start_time = time.time()
    for _ in range(100):
        env.reset()
    reset_time = (time.time() - start_time) / 100
    print(f"平均重置时间: {reset_time*1000:.2f} ms")
    
    # 测试步骤执行时间
    env.reset()
    start_time = time.time()
    for _ in range(1000):
        action = env.action_space.sample()
        env.step(action)
    step_time = (time.time() - start_time) / 1000
    print(f"平均步骤时间: {step_time*1000:.2f} ms")
    
    # 测试观测空间大小
    obs, _ = env.reset()
    print(f"观测向量大小: {len(obs)} 维")
    print(f"动作空间大小: {env.action_space.shape}")
    
    env.close()
    print("性能测试完成!\n")


def main():
    """主测试函数"""
    print("多无人机协同作战环境测试")
    print("=" * 50)
    
    try:
        # 运行所有测试
        test_basic_functionality()
        test_reward_system()
        performance_test()
        
        # 最后运行可视化测试
        choice = input("是否运行可视化测试? (y/n): ").lower()
        if choice == 'y':
            test_visualization()
        
        print("所有测试完成!")
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()