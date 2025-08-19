import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
import math


class TaskDemandPredictor(nn.Module):
    """任务需求预测器：预测未来的任务需求变化"""
    def __init__(self, input_dim, hidden_dim=128, prediction_horizon=5):
        super(TaskDemandPredictor, self).__init__()
        self.prediction_horizon = prediction_horizon
        
        # LSTM用于时序预测
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        
        # 预测头：预测未来每个时刻的需求
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prediction_horizon * 3),  # 3个敌方目标的需求
            nn.Sigmoid()
        )
        
    def forward(self, history_obs, history_demands):
        """
        history_obs: [batch_size, seq_len, obs_dim] 历史观测
        history_demands: [batch_size, seq_len, 3] 历史需求
        """
        # 结合观测和需求信息
        combined_input = torch.cat([history_obs, history_demands], dim=-1)
        
        # LSTM处理
        lstm_out, _ = self.lstm(combined_input)  # [batch_size, seq_len, hidden_dim]
        
        # 使用最后一个时刻的输出进行预测
        last_hidden = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # 预测未来需求
        predictions = self.predictor(last_hidden)  # [batch_size, prediction_horizon * 3]
        predictions = predictions.view(-1, self.prediction_horizon, 3)
        
        return predictions


class AgentCapabilityEstimator(nn.Module):
    """智能体能力评估器：评估每个智能体完成不同任务的能力"""
    def __init__(self, obs_dim, num_agents=9, hidden_dim=128):
        super(AgentCapabilityEstimator, self).__init__()
        self.num_agents = num_agents
        
        # 能力特征提取器
        self.capability_extractor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 任务特定的能力评估
        self.coverage_capability = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.tracking_capability = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 协作能力评估
        self.cooperation_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, obs, agent_positions):
        """
        obs: [batch_size, obs_dim] 全局观测
        agent_positions: [batch_size, num_agents, 3] 智能体位置
        """
        batch_size = obs.size(0)
        
        # 特征提取
        global_features = self.capability_extractor(obs)  # [batch_size, hidden_dim]
        
        # 为每个智能体评估能力
        coverage_caps = []
        tracking_caps = []
        cooperation_scores = []
        
        for i in range(self.num_agents):
            # 单个智能体的能力评估
            coverage_cap = self.coverage_capability(global_features)  # [batch_size, 1]
            tracking_cap = self.tracking_capability(global_features)   # [batch_size, 1]
            
            coverage_caps.append(coverage_cap)
            tracking_caps.append(tracking_cap)
            
            # 与其他智能体的协作能力
            agent_coop_scores = []
            for j in range(self.num_agents):
                if i != j:
                    # 计算协作特征
                    coop_input = torch.cat([global_features, global_features], dim=-1)
                    coop_score = self.cooperation_net(coop_input)  # [batch_size, 1]
                    agent_coop_scores.append(coop_score)
            
            if agent_coop_scores:
                avg_coop = torch.stack(agent_coop_scores, dim=1).mean(dim=1)  # [batch_size, 1]
            else:
                avg_coop = torch.ones(batch_size, 1, device=obs.device)
            
            cooperation_scores.append(avg_coop)
        
        # 组合结果
        coverage_capabilities = torch.stack(coverage_caps, dim=1)      # [batch_size, num_agents, 1]
        tracking_capabilities = torch.stack(tracking_caps, dim=1)      # [batch_size, num_agents, 1]
        cooperation_capabilities = torch.stack(cooperation_scores, dim=1)  # [batch_size, num_agents, 1]
        
        return {
            'coverage': coverage_capabilities.squeeze(-1),     # [batch_size, num_agents]
            'tracking': tracking_capabilities.squeeze(-1),     # [batch_size, num_agents]
            'cooperation': cooperation_capabilities.squeeze(-1) # [batch_size, num_agents]
        }


class DynamicTaskAllocationNetwork(nn.Module):
    """动态任务分配网络：核心创新算法"""
    def __init__(self, obs_dim, num_agents=9, num_enemies=3, hidden_dim=256, prediction_horizon=5):
        super(DynamicTaskAllocationNetwork, self).__init__()
        
        self.num_agents = num_agents
        self.num_enemies = num_enemies
        self.hidden_dim = hidden_dim
        
        # 子模块
        self.demand_predictor = TaskDemandPredictor(
            input_dim=obs_dim + num_enemies, 
            hidden_dim=hidden_dim, 
            prediction_horizon=prediction_horizon
        )
        
        self.capability_estimator = AgentCapabilityEstimator(
            obs_dim=obs_dim, 
            num_agents=num_agents, 
            hidden_dim=hidden_dim
        )
        
        # 任务分配决策网络
        input_dim = num_agents + num_agents * num_enemies + num_agents * num_enemies  # 能力 + 需求扩展 + 优先级扩展
        self.allocation_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents * num_enemies),  # 分配矩阵
            nn.Softmax(dim=-1)
        )
        
        # 优先级网络
        self.priority_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_enemies),
            nn.Softmax(dim=-1)
        )
        
        # 协调网络：处理冲突和重分配
        coord_input_dim = num_agents * num_enemies + obs_dim  # 神经分配 + 观测
        self.coordination_net = nn.Sequential(
            nn.Linear(coord_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents * num_enemies),
            nn.Sigmoid()
        )
        
        # 历史缓存
        self.register_buffer('obs_history', torch.zeros(1, 10, obs_dim))
        self.register_buffer('demand_history', torch.zeros(1, 10, num_enemies))
        self.history_length = 0
        
    def parse_current_demands(self, obs):
        """从观测中解析当前任务需求"""
        batch_size = obs.size(0)
        
        # 覆盖需求信息位置：obs[60:66] -> [required, current] for 3 enemies
        coverage_start = self.num_agents * 3 + self.num_enemies * 6
        coverage_info = obs[:, coverage_start:coverage_start + self.num_enemies * 2]
        coverage_info = coverage_info.view(batch_size, self.num_enemies, 2)
        
        # 当前需求 = 要求覆盖 - 当前覆盖
        required = coverage_info[:, :, 0]  # [batch_size, num_enemies]
        current = coverage_info[:, :, 1]   # [batch_size, num_enemies]
        demands = torch.clamp(required - current, min=0)  # [batch_size, num_enemies]
        
        return demands
    
    def update_history(self, obs, demands):
        """更新历史信息"""
        batch_size = obs.size(0)
        
        if self.history_length == 0 or self.obs_history.size(0) != batch_size:
            # 初始化或批量大小改变
            self.obs_history = obs.unsqueeze(1).expand(-1, 10, -1).clone()
            self.demand_history = demands.unsqueeze(1).expand(-1, 10, -1).clone()
            self.history_length = 1
        else:
            # 滚动更新（只更新对应批量大小的部分）
            if batch_size <= self.obs_history.size(0):
                new_obs_history = torch.cat([
                    self.obs_history[:batch_size, 1:].clone(),
                    obs.unsqueeze(1)
                ], dim=1)
                new_demand_history = torch.cat([
                    self.demand_history[:batch_size, 1:].clone(),
                    demands.unsqueeze(1)
                ], dim=1)
                self.obs_history[:batch_size] = new_obs_history
                self.demand_history[:batch_size] = new_demand_history
            else:
                # 批量大小增大，重新初始化
                self.obs_history = obs.unsqueeze(1).expand(-1, 10, -1).clone()
                self.demand_history = demands.unsqueeze(1).expand(-1, 10, -1).clone()
            
            self.history_length = min(self.history_length + 1, 10)
    
    def optimal_assignment(self, cost_matrix):
        """使用匈牙利算法进行最优分配"""
        batch_size = cost_matrix.size(0)
        assignments = []
        
        for b in range(batch_size):
            cost = cost_matrix[b].detach().cpu().numpy()
            
            # 扩展成方阵（如果需要）
            if cost.shape[0] != cost.shape[1]:
                max_dim = max(cost.shape)
                square_cost = np.full((max_dim, max_dim), cost.max() + 1)
                square_cost[:cost.shape[0], :cost.shape[1]] = cost
                cost = square_cost
            
            # 匈牙利算法
            row_indices, col_indices = linear_sum_assignment(cost)
            
            # 创建分配矩阵
            assignment = torch.zeros_like(cost_matrix[b])
            valid_assignments = (row_indices < self.num_agents) & (col_indices < self.num_enemies)
            valid_rows = row_indices[valid_assignments]
            valid_cols = col_indices[valid_assignments]
            
            if len(valid_rows) > 0:
                assignment[valid_rows, valid_cols] = 1.0
            
            assignments.append(assignment)
        
        return torch.stack(assignments, dim=0).to(cost_matrix.device)
    
    def compute_assignment_cost(self, capabilities, demands, distances, priorities):
        """计算分配成本矩阵"""
        batch_size = demands.size(0)
        
        # 基础成本：距离成本
        distance_cost = distances  # [batch_size, num_agents, num_enemies]
        
        # 能力成本：能力越强成本越低
        capability_cost = 1 - capabilities['coverage'].unsqueeze(-1)  # [batch_size, num_agents, 1]
        capability_cost = capability_cost.expand(-1, -1, self.num_enemies)
        
        # 需求成本：需求越高优先级越高
        demand_cost = (1 - demands.unsqueeze(1))  # [batch_size, 1, num_enemies]
        demand_cost = demand_cost.expand(-1, self.num_agents, -1)
        
        # 优先级成本
        priority_cost = (1 - priorities.unsqueeze(1))  # [batch_size, 1, num_enemies]
        priority_cost = priority_cost.expand(-1, self.num_agents, -1)
        
        # 综合成本
        total_cost = (distance_cost * 0.4 + 
                     capability_cost * 0.3 + 
                     demand_cost * 0.2 + 
                     priority_cost * 0.1)
        
        return total_cost
    
    def forward(self, obs):
        """
        obs: [batch_size, obs_dim]
        返回: 动态任务分配结果
        """
        batch_size = obs.size(0)
        
        # 1. 解析当前状态
        demands = self.parse_current_demands(obs)  # [batch_size, num_enemies]
        
        # 2. 更新历史
        self.update_history(obs, demands)
        
        # 3. 预测未来需求
        if self.history_length >= 3:  # 需要足够的历史数据
            future_demands = self.demand_predictor(
                self.obs_history[:batch_size, -self.history_length:],
                self.demand_history[:batch_size, -self.history_length:]
            )  # [batch_size, prediction_horizon, num_enemies]
        else:
            future_demands = demands.unsqueeze(1).expand(-1, 5, -1)
        
        # 4. 评估智能体能力
        # 解析智能体位置
        agent_positions = obs[:, :self.num_agents * 3].view(batch_size, self.num_agents, 3)
        capabilities = self.capability_estimator(obs, agent_positions)
        
        # 5. 计算任务优先级
        priorities = self.priority_net(obs)  # [batch_size, num_enemies]
        
        # 6. 计算距离矩阵
        enemy_positions = obs[:, self.num_agents * 3:self.num_agents * 3 + self.num_enemies * 3]
        enemy_positions = enemy_positions.view(batch_size, self.num_enemies, 3)
        
        # 计算智能体到敌方的距离
        distances = torch.cdist(agent_positions, enemy_positions, p=2)  # [batch_size, num_agents, num_enemies]
        
        # 7. 计算分配成本并进行最优分配
        cost_matrix = self.compute_assignment_cost(capabilities, demands, distances, priorities)
        optimal_allocation = self.optimal_assignment(cost_matrix)  # [batch_size, num_agents, num_enemies]
        
        # 8. 神经网络辅助分配（软分配）
        allocation_features = torch.cat([
            capabilities['coverage'],  # [batch_size, num_agents]
            demands.unsqueeze(1).expand(-1, self.num_agents, -1).flatten(1),  # [batch_size, num_agents * num_enemies]
            priorities.unsqueeze(1).expand(-1, self.num_agents, -1).flatten(1)  # [batch_size, num_agents * num_enemies]
        ], dim=-1)
        
        neural_allocation = self.allocation_net(allocation_features)  # [batch_size, num_agents * num_enemies]
        neural_allocation = neural_allocation.view(batch_size, self.num_agents, self.num_enemies)
        
        # 9. 协调冲突
        conflict_features = torch.cat([
            neural_allocation.flatten(1),
            obs  # 使用完整观测作为上下文
        ], dim=-1)
        
        coordination_weights = self.coordination_net(conflict_features)  # [batch_size, num_agents * num_enemies]
        coordination_weights = coordination_weights.view(batch_size, self.num_agents, self.num_enemies)
        
        # 10. 最终分配：结合最优分配和神经网络分配
        alpha = 0.7  # 最优分配的权重
        final_allocation = (alpha * optimal_allocation + 
                          (1 - alpha) * neural_allocation * coordination_weights)
        
        # 11. 生成任务指令
        task_assignments = self.generate_task_assignments(final_allocation, demands, priorities)
        
        return {
            'allocation_matrix': final_allocation,           # [batch_size, num_agents, num_enemies]
            'task_assignments': task_assignments,            # [batch_size, num_agents]
            'predicted_demands': future_demands,             # [batch_size, prediction_horizon, num_enemies]
            'current_demands': demands,                      # [batch_size, num_enemies]
            'priorities': priorities,                        # [batch_size, num_enemies]
            'capabilities': capabilities,                    # dict
            'cost_matrix': cost_matrix                       # [batch_size, num_agents, num_enemies]
        }
    
    def generate_task_assignments(self, allocation_matrix, demands, priorities):
        """根据分配矩阵生成具体的任务指令"""
        batch_size = allocation_matrix.size(0)
        
        # 为每个智能体分配主要目标
        assignments = torch.argmax(allocation_matrix, dim=-1)  # [batch_size, num_agents]
        
        # 考虑需求和优先级调整分配
        for b in range(batch_size):
            # 如果某个目标需求很高，优先分配更多智能体
            high_demand_targets = torch.where(demands[b] > demands[b].mean())[0]
            
            if len(high_demand_targets) > 0:
                # 重新分配部分智能体到高需求目标
                available_agents = torch.arange(self.num_agents)
                for target in high_demand_targets:
                    # 找到距离最近的未分配智能体
                    target_agents = torch.where(assignments[b] == target)[0]
                    if len(target_agents) < demands[b, target]:
                        # 需要更多智能体
                        remaining_agents = available_agents[~torch.isin(available_agents, target_agents)]
                        if len(remaining_agents) > 0:
                            # 分配最近的智能体
                            assignments[b, remaining_agents[0]] = target
        
        return assignments
    
    def reset_history(self):
        """重置历史信息"""
        self.obs_history.zero_()
        self.demand_history.zero_()
        self.history_length = 0


# 测试和可视化
if __name__ == "__main__":
    # 测试动态任务分配网络
    batch_size = 16
    obs_dim = 78
    
    dtan = DynamicTaskAllocationNetwork(obs_dim)
    
    # 创建随机观测
    obs = torch.randn(batch_size, obs_dim)
    
    # 前向传播
    results = dtan(obs)
    
    print(f"输入观测维度: {obs.shape}")
    print(f"分配矩阵维度: {results['allocation_matrix'].shape}")
    print(f"任务分配维度: {results['task_assignments'].shape}")
    print(f"预测需求维度: {results['predicted_demands'].shape}")
    print(f"当前需求: {results['current_demands'][0]}")
    print(f"优先级: {results['priorities'][0]}")
    print(f"任务分配: {results['task_assignments'][0]}")
    
    print("动态任务分配网络测试完成！")