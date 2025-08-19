import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SpatialAttention(nn.Module):
    """空间注意力：关注3D空间中的重要区域"""
    def __init__(self, feature_dim, spatial_dim=3, hidden_dim=128):
        super(SpatialAttention, self).__init__()
        self.feature_dim = feature_dim
        self.spatial_dim = spatial_dim
        
        # 位置编码
        self.position_encoder = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # 空间注意力权重计算
        self.spatial_attention = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features, positions):
        """
        features: [batch_size, num_objects, feature_dim]
        positions: [batch_size, num_objects, spatial_dim] (x, y, z)
        """
        # 位置编码
        pos_encoding = self.position_encoder(positions)  # [batch_size, num_objects, feature_dim]
        
        # 特征与位置编码融合
        fused_features = torch.cat([features, pos_encoding], dim=-1)  # [batch_size, num_objects, feature_dim*2]
        
        # 计算空间注意力权重
        spatial_weights = self.spatial_attention(fused_features)  # [batch_size, num_objects, 1]
        
        # 应用注意力权重
        attended_features = features * spatial_weights  # [batch_size, num_objects, feature_dim]
        
        return attended_features, spatial_weights


class TemporalAttention(nn.Module):
    """时间注意力：关注历史信息的重要性"""
    def __init__(self, feature_dim, sequence_length=10, hidden_dim=128):
        super(TemporalAttention, self).__init__()
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        
        # 时间编码
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # 时间注意力权重计算
        self.temporal_attention = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features, time_steps):
        """
        features: [batch_size, sequence_length, feature_dim]
        time_steps: [batch_size, sequence_length, 1] 时间步信息
        """
        # 时间编码
        time_encoding = self.time_encoder(time_steps)  # [batch_size, sequence_length, feature_dim]
        
        # 特征与时间编码融合
        fused_features = torch.cat([features, time_encoding], dim=-1)
        
        # 计算时间注意力权重
        temporal_weights = self.temporal_attention(fused_features)  # [batch_size, sequence_length, 1]
        
        # 应用注意力权重
        attended_features = features * temporal_weights
        
        # 聚合时间信息
        aggregated_features = torch.sum(attended_features, dim=1)  # [batch_size, feature_dim]
        
        return aggregated_features, temporal_weights


class TaskAttention(nn.Module):
    """任务注意力：根据当前任务需求动态调整关注点"""
    def __init__(self, feature_dim, num_tasks=2, hidden_dim=128):
        super(TaskAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_tasks = num_tasks  # 0: 覆盖任务, 1: 追踪任务
        
        # 任务嵌入
        self.task_embedding = nn.Embedding(num_tasks, feature_dim)
        
        # 任务特定的查询、键、值
        self.query_nets = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(num_tasks)
        ])
        self.key_nets = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(num_tasks)
        ])
        self.value_nets = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(num_tasks)
        ])
        
        self.scale = math.sqrt(feature_dim)
        
    def forward(self, features, task_ids):
        """
        features: [batch_size, num_objects, feature_dim]
        task_ids: [batch_size] 当前主要任务ID
        """
        batch_size = features.size(0)
        num_objects = features.size(1)
        
        # 获取任务嵌入
        task_emb = self.task_embedding(task_ids)  # [batch_size, feature_dim]
        task_emb = task_emb.unsqueeze(1).expand(-1, num_objects, -1)  # [batch_size, num_objects, feature_dim]
        
        # 为每个任务计算注意力
        task_features = []
        task_weights = []
        
        for task_id in range(self.num_tasks):
            # 计算查询、键、值
            task_mask = (task_ids == task_id).float().view(-1, 1, 1)
            
            if task_mask.sum() > 0:  # 如果有这个任务
                q = self.query_nets[task_id](task_emb)  # [batch_size, num_objects, feature_dim]
                k = self.key_nets[task_id](features)    # [batch_size, num_objects, feature_dim]
                v = self.value_nets[task_id](features)  # [batch_size, num_objects, feature_dim]
                
                # 计算注意力权重
                attention_scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # [batch_size, num_objects, num_objects]
                attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_objects, num_objects]
                
                # 应用注意力
                attended_features = torch.bmm(attention_weights, v)  # [batch_size, num_objects, feature_dim]
                
                # 只保留当前任务的结果
                attended_features = attended_features * task_mask
                
                task_features.append(attended_features)
                task_weights.append(attention_weights)
        
        # 组合所有任务的结果
        if task_features:
            final_features = torch.stack(task_features, dim=0).sum(dim=0)  # [batch_size, num_objects, feature_dim]
            final_weights = torch.stack(task_weights, dim=0).sum(dim=0) if task_weights else None
        else:
            final_features = features
            final_weights = None
        
        return final_features, final_weights


class HierarchicalAttentionModule(nn.Module):
    """层次化注意力模块：整合空间、时间和任务注意力"""
    def __init__(self, obs_dim, num_agents=9, num_enemies=3, 
                 feature_dim=256, hidden_dim=128, sequence_length=10):
        super(HierarchicalAttentionModule, self).__init__()
        
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.num_enemies = num_enemies
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU()
        )
        
        # 三级注意力机制
        self.spatial_attention = SpatialAttention(feature_dim, spatial_dim=3, hidden_dim=hidden_dim)
        self.temporal_attention = TemporalAttention(feature_dim, sequence_length=sequence_length, hidden_dim=hidden_dim)
        self.task_attention = TaskAttention(feature_dim, num_tasks=2, hidden_dim=hidden_dim)
        
        # 特征融合
        self.fusion_net = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 历史信息缓存
        self.register_buffer('history_features', torch.zeros(1, sequence_length, feature_dim))
        self.register_buffer('history_time', torch.zeros(1, sequence_length, 1))
        self.step_count = 0
        
    def parse_observation(self, obs):
        """解析观测向量"""
        batch_size = obs.size(0)
        
        # 解析我方无人机位置 (前27维)
        friendly_positions = obs[:, :self.num_agents * 3].view(batch_size, self.num_agents, 3)
        
        # 解析敌方无人机信息 (接下来18维位置 + 6维速度)
        enemy_start = self.num_agents * 3
        enemy_positions = obs[:, enemy_start:enemy_start + self.num_enemies * 3].view(batch_size, self.num_enemies, 3)
        enemy_velocities = obs[:, enemy_start + self.num_enemies * 3:enemy_start + self.num_enemies * 6].view(batch_size, self.num_enemies, 3)
        
        # 解析覆盖需求信息
        coverage_start = enemy_start + self.num_enemies * 6
        coverage_info = obs[:, coverage_start:coverage_start + self.num_enemies * 2].view(batch_size, self.num_enemies, 2)
        
        return {
            'friendly_positions': friendly_positions,
            'enemy_positions': enemy_positions,
            'enemy_velocities': enemy_velocities,
            'coverage_info': coverage_info
        }
    
    def determine_primary_task(self, parsed_obs):
        """根据当前状态确定主要任务"""
        batch_size = parsed_obs['coverage_info'].size(0)
        
        # 检查覆盖需求是否满足
        required_coverage = parsed_obs['coverage_info'][:, :, 0]  # [batch_size, num_enemies]
        current_coverage = parsed_obs['coverage_info'][:, :, 1]  # [batch_size, num_enemies]
        
        coverage_deficit = required_coverage - current_coverage
        coverage_satisfied = (coverage_deficit <= 0).all(dim=1)  # [batch_size]
        
        # 如果覆盖未满足，主要任务是覆盖(0)；否则是追踪(1)
        primary_tasks = coverage_satisfied.long()
        
        return primary_tasks
    
    def forward(self, obs):
        """
        obs: [batch_size, obs_dim]
        返回: 层次化注意力处理后的特征
        """
        batch_size = obs.size(0)
        
        # 1. 特征提取
        features = self.feature_extractor(obs)  # [batch_size, feature_dim]
        
        # 2. 解析观测
        parsed_obs = self.parse_observation(obs)
        
        # 3. 空间注意力
        # 构造空间对象（我方+敌方无人机）
        all_positions = torch.cat([
            parsed_obs['friendly_positions'],
            parsed_obs['enemy_positions']
        ], dim=1)  # [batch_size, num_agents + num_enemies, 3]
        
        # 为每个对象创建特征表示
        num_objects = self.num_agents + self.num_enemies
        object_features = features.unsqueeze(1).expand(-1, num_objects, -1)  # [batch_size, num_objects, feature_dim]
        
        spatial_features, spatial_weights = self.spatial_attention(object_features, all_positions)
        
        # 聚合空间特征
        spatial_aggregated = spatial_features.mean(dim=1)  # [batch_size, feature_dim]
        
        # 4. 时间注意力
        # 更新历史信息
        self.step_count += 1
        current_time = torch.tensor([[self.step_count]], dtype=torch.float32, device=obs.device)
        current_time = current_time.expand(batch_size, 1, 1)
        
        if self.step_count == 1 or self.history_features.size(0) != batch_size:
            # 初始化历史或批量大小改变时重新初始化
            self.history_features = features.unsqueeze(1).expand(-1, self.sequence_length, -1).clone()
            self.history_time = current_time.expand(-1, self.sequence_length, -1).clone()
        else:
            # 滚动更新历史（只更新对应批量大小的部分）
            if batch_size <= self.history_features.size(0):
                new_history = torch.cat([
                    self.history_features[:batch_size, 1:].clone(),
                    features.unsqueeze(1)
                ], dim=1)
                new_time = torch.cat([
                    self.history_time[:batch_size, 1:].clone(),
                    current_time
                ], dim=1)
                self.history_features[:batch_size] = new_history
                self.history_time[:batch_size] = new_time
            else:
                # 批量大小增大，重新初始化
                self.history_features = features.unsqueeze(1).expand(-1, self.sequence_length, -1).clone()
                self.history_time = current_time.expand(-1, self.sequence_length, -1).clone()
        
        temporal_features, temporal_weights = self.temporal_attention(
            self.history_features[:batch_size], 
            self.history_time[:batch_size]
        )
        
        # 5. 任务注意力
        primary_tasks = self.determine_primary_task(parsed_obs)
        task_features, task_weights = self.task_attention(object_features, primary_tasks)
        task_aggregated = task_features.mean(dim=1)  # [batch_size, feature_dim]
        
        # 6. 特征融合
        fused_features = torch.cat([
            spatial_aggregated,
            temporal_features,
            task_aggregated
        ], dim=-1)  # [batch_size, feature_dim * 3]
        
        final_features = self.fusion_net(fused_features)  # [batch_size, feature_dim]
        
        # 返回注意力特征和权重（用于可视化和分析）
        attention_info = {
            'spatial_weights': spatial_weights,
            'temporal_weights': temporal_weights,
            'task_weights': task_weights,
            'primary_tasks': primary_tasks
        }
        
        return final_features, attention_info
    
    def reset_history(self):
        """重置历史信息"""
        self.step_count = 0
        self.history_features.zero_()
        self.history_time.zero_()


# 使用示例和测试
if __name__ == "__main__":
    # 测试层次化注意力模块
    batch_size = 32
    obs_dim = 78  # 从环境中获取的观测维度
    
    ham = HierarchicalAttentionModule(obs_dim)
    
    # 创建随机观测
    obs = torch.randn(batch_size, obs_dim)
    
    # 前向传播
    features, attention_info = ham(obs)
    
    print(f"输入观测维度: {obs.shape}")
    print(f"输出特征维度: {features.shape}")
    print(f"主要任务: {attention_info['primary_tasks']}")
    print(f"空间注意力权重形状: {attention_info['spatial_weights'].shape}")
    print(f"时间注意力权重形状: {attention_info['temporal_weights'].shape}")
    
    print("层次化注意力模块测试完成！")