#!/usr/bin/env python3

import numpy as np
import torch
import time

from multi_drone_env import MultiDroneEnvironment
from algorithms.ham_dtan_maddpg import HAMDTANMADDPGAlgorithm


def test_core_algorithms():
    """测试核心算法功能"""
    print("=" * 60)
    print("多无人机协同强化学习算法核心功能测试")
    print("=" * 60)
    
    # 1. 环境测试
    print("1. 测试环境...")
    env = MultiDroneEnvironment(num_friendly_drones=9, num_enemy_drones=3)
    obs, info = env.reset()
    
    print(f"✓ 环境创建成功")
    print(f"  - 观测维度: {obs.shape}")
    print(f"  - 我方无人机: {env.num_friendly_drones}架")
    print(f"  - 敌方无人机: {env.num_enemy_drones}架")
    print(f"  - 状态空间: {env.map_size}")
    
    # 2. 算法实例化测试
    print("\n2. 测试HAM-DTAN-MADDPG算法...")
    algorithm = HAMDTANMADDPGAlgorithm(
        num_agents=9,
        obs_dim=78,
        action_dim=3,
        num_enemies=3
    )
    
    print(f"✓ 算法实例化成功")
    print(f"  - 总参数量: {algorithm.get_algorithm_info()['total_parameters']:,}")
    
    # 3. 动作选择测试
    print("\n3. 测试动作选择...")
    start_time = time.time()
    
    actions = algorithm.select_actions(obs, add_noise=False)
    action_time = time.time() - start_time
    
    print(f"✓ 动作选择成功")
    print(f"  - 动作维度: {actions.shape}")
    print(f"  - 选择时间: {action_time*1000:.2f}ms")
    
    # 4. 详细信息获取测试
    print("\n4. 测试注意力和任务分配...")
    start_time = time.time()
    
    actions, infos = algorithm.select_actions(obs, add_noise=False, return_info=True)
    info_time = time.time() - start_time
    
    print(f"✓ 详细信息获取成功")
    print(f"  - 信息获取时间: {info_time*1000:.2f}ms")
    print(f"  - 返回信息数量: {len(infos)}")
    
    # 检查第一个智能体的信息
    if infos and len(infos) > 0:
        agent_info = infos[0]
        if 'attention_info' in agent_info:
            att_info = agent_info['attention_info']
            print(f"  - 主要任务: {att_info.get('primary_tasks', 'N/A')}")
            
        if 'dtan_results' in agent_info:
            dtan_info = agent_info['dtan_results']
            print(f"  - 当前需求: {dtan_info.get('current_demands', 'N/A')}")
            print(f"  - 分配矩阵形状: {dtan_info.get('allocation_matrix', torch.tensor([])).shape}")
    
    # 5. 环境交互测试
    print("\n5. 测试环境交互...")
    total_reward = 0
    steps = 0
    
    for step in range(10):
        actions = algorithm.select_actions(obs, add_noise=True)
        next_obs, reward, terminated, truncated, step_info = env.step(actions)
        
        total_reward += reward
        steps += 1
        obs = next_obs
        
        if terminated or truncated:
            break
    
    print(f"✓ 环境交互测试完成")
    print(f"  - 交互步数: {steps}")
    print(f"  - 累计奖励: {total_reward:.2f}")
    print(f"  - 平均奖励: {total_reward/steps:.2f}")
    
    # 6. 性能基准测试
    print("\n6. 性能基准测试...")
    
    # 测试选择动作的性能
    obs_batch = [obs for _ in range(100)]
    start_time = time.time()
    
    for test_obs in obs_batch:
        _ = algorithm.select_actions(test_obs, add_noise=False)
    
    batch_time = time.time() - start_time
    avg_time = batch_time / 100
    
    print(f"✓ 性能测试完成")
    print(f"  - 100次动作选择总时间: {batch_time:.3f}s")
    print(f"  - 平均单次时间: {avg_time*1000:.2f}ms")
    print(f"  - 理论FPS: {1/avg_time:.1f}")
    
    # 7. 算法组件测试
    print("\n7. 测试算法组件...")
    
    # 测试HAM
    print("  测试层次化注意力机制(HAM)...")
    ham_module = algorithm.agents[0].actor.ham
    ham_features, ham_info = ham_module(torch.FloatTensor(obs).unsqueeze(0))
    print(f"    ✓ HAM输出维度: {ham_features.shape}")
    print(f"    ✓ 空间注意力权重: {ham_info['spatial_weights'].shape if ham_info['spatial_weights'] is not None else 'None'}")
    
    # 测试DTAN
    print("  测试动态任务分配网络(DTAN)...")
    dtan_module = algorithm.agents[0].actor.dtan
    dtan_results = dtan_module(torch.FloatTensor(obs).unsqueeze(0))
    print(f"    ✓ DTAN分配矩阵: {dtan_results['allocation_matrix'].shape}")
    print(f"    ✓ 当前需求: {dtan_results['current_demands'].shape}")
    print(f"    ✓ 预测需求: {dtan_results['predicted_demands'].shape}")
    
    # 8. 内存使用测试
    print("\n8. 内存使用测试...")
    
    # 重置算法内存
    algorithm.reset_memory()
    print(f"  ✓ 内存重置完成")
    
    # 噪声重置
    algorithm.reset_noise()
    print(f"  ✓ 噪声重置完成")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("🎉 所有核心功能测试通过！")
    print("=" * 60)
    
    return True


def test_training_components():
    """测试训练相关组件"""
    print("\n" + "=" * 60)
    print("训练组件测试")
    print("=" * 60)
    
    # 创建算法实例
    algorithm = HAMDTANMADDPGAlgorithm()
    env = MultiDroneEnvironment()
    
    # 1. 经验存储测试
    print("1. 测试经验存储...")
    
    obs, _ = env.reset()
    actions = algorithm.select_actions(obs)
    next_obs, reward, terminated, truncated, info = env.step(actions)
    
    # 存储经验
    algorithm.store_transition(obs, actions.flatten(), reward, next_obs, terminated or truncated)
    
    print(f"  ✓ 经验存储成功")
    print(f"  ✓ 经验池大小: {len(algorithm.replay_buffer)}")
    
    # 2. 批量经验生成
    print("\n2. 生成批量经验...")
    
    for _ in range(100):
        obs, _ = env.reset()
        algorithm.reset_noise()
        
        for step in range(10):
            actions = algorithm.select_actions(obs, add_noise=True)
            next_obs, reward, terminated, truncated, info = env.step(actions)
            
            algorithm.store_transition(obs, actions.flatten(), reward, next_obs, terminated or truncated)
            
            obs = next_obs
            if terminated or truncated:
                break
    
    print(f"  ✓ 批量经验生成完成")
    print(f"  ✓ 最终经验池大小: {len(algorithm.replay_buffer)}")
    
    # 3. 网络更新测试
    print("\n3. 测试网络更新...")
    
    if len(algorithm.replay_buffer) >= 256:
        start_time = time.time()
        losses = algorithm.update(batch_size=256)
        update_time = time.time() - start_time
        
        print(f"  ✓ 网络更新成功")
        print(f"  ✓ 更新时间: {update_time:.3f}s")
        print(f"  ✓ 损失信息数量: {len(losses)}")
        
        # 打印部分损失信息
        sample_losses = list(losses.items())[:3]
        for key, value in sample_losses:
            print(f"    - {key}: {value:.4f}")
    else:
        print(f"  ⚠ 经验不足，跳过更新测试")
    
    env.close()
    print("\n✅ 训练组件测试完成")


def main():
    """主测试函数"""
    print("开始HAM-DTAN-MADDPG算法测试...")
    
    try:
        # 核心功能测试
        success = test_core_algorithms()
        
        if success:
            # 训练组件测试
            test_training_components()
            
            print("\n🎊 所有测试完成！算法已准备就绪。")
            print("\n📋 下一步操作建议:")
            print("  1. 运行 python experiments/training_script.py 开始训练")
            print("  2. 运行 python experiments/visualization.py 生成可视化结果")
            print("  3. 运行 python demo_visualization.py 查看实时演示")
            
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()