import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from collections import deque
import random

from .hierarchical_attention import HierarchicalAttentionModule
from .dynamic_task_allocation import DynamicTaskAllocationNetwork
from .maddpg_baseline import ReplayBuffer, OUNoise


class EnhancedActor(nn.Module):
    """增强的Actor网络，集成HAM和DTAN"""
    def __init__(self, obs_dim, action_dim, num_agents=9, num_enemies=3, hidden_dim=256):
        super(EnhancedActor, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 层次化注意力模块
        self.ham = HierarchicalAttentionModule(
            obs_dim=obs_dim,
            num_agents=num_agents,
            num_enemies=num_enemies,
            feature_dim=hidden_dim
        )
        
        # 动态任务分配网络
        self.dtan = DynamicTaskAllocationNetwork(
            obs_dim=obs_dim,
            num_agents=num_agents,
            num_enemies=num_enemies,
            hidden_dim=hidden_dim
        )
        
        # 任务感知的策略网络
        self.task_policy_net = nn.Sequential(
            nn.Linear(hidden_dim + num_enemies, hidden_dim),  # HAM特征 + 任务分配
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 协作感知的策略网络
        self.collab_policy_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),  # 任务特征 + 协作信息
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 最终动作输出
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()
        )
        
        # 注意力权重预测（用于可解释性）
        self.attention_predictor = nn.Sequential(
            nn.Linear(hidden_dim, num_agents + num_enemies),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, obs, agent_id):
        """
        obs: [batch_size, obs_dim] 全局观测
        agent_id: int 当前智能体ID
        """
        batch_size = obs.size(0)
        
        # 1. 层次化注意力处理
        ham_features, attention_info = self.ham(obs)  # [batch_size, hidden_dim]
        
        # 2. 动态任务分配
        dtan_results = self.dtan(obs)
        
        # 获取当前智能体的任务分配
        agent_allocation = dtan_results['allocation_matrix'][:, agent_id, :]  # [batch_size, num_enemies]
        
        # 3. 任务感知策略
        task_input = torch.cat([ham_features, agent_allocation], dim=-1)
        task_features = self.task_policy_net(task_input)  # [batch_size, hidden_dim]
        
        # 4. 协作感知：考虑其他智能体的分配
        # 计算协作上下文
        other_agents_allocation = dtan_results['allocation_matrix'].sum(dim=1)  # [batch_size, num_enemies]
        collab_context = self.compute_collaboration_context(
            ham_features, 
            other_agents_allocation, 
            agent_allocation
        )
        
        collab_input = torch.cat([task_features, collab_context], dim=-1)
        collab_features = self.collab_policy_net(collab_input)  # [batch_size, hidden_dim]
        
        # 5. 生成动作
        action = self.action_head(collab_features)  # [batch_size, action_dim]
        
        # 6. 预测注意力权重（可解释性）
        predicted_attention = self.attention_predictor(collab_features)
        
        return action, {
            'ham_features': ham_features,
            'attention_info': attention_info,
            'dtan_results': dtan_results,
            'task_features': task_features,
            'collab_features': collab_features,
            'predicted_attention': predicted_attention
        }
    
    def compute_collaboration_context(self, global_features, team_allocation, agent_allocation):
        """计算协作上下文"""
        # 团队总分配强度
        team_intensity = team_allocation.sum(dim=-1, keepdim=True)  # [batch_size, 1]
        
        # 个体与团队的分配差异
        allocation_diff = (agent_allocation - team_allocation / 9.0).abs().sum(dim=-1, keepdim=True)
        
        # 协作特征
        collab_features = torch.cat([
            global_features * team_intensity,
            global_features * allocation_diff
        ], dim=-1)
        
        # 降维到隐藏维度
        collab_context = F.adaptive_avg_pool1d(
            collab_features.unsqueeze(1), 
            self.hidden_dim
        ).squeeze(1)
        
        return collab_context


class EnhancedCritic(nn.Module):
    """增强的Critic网络"""
    def __init__(self, total_obs_dim, total_action_dim, hidden_dim=256):
        super(EnhancedCritic, self).__init__()
        
        # 全局状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(total_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 动作编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 状态-动作融合
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, obs, actions):
        """
        obs: [batch_size, total_obs_dim]
        actions: [batch_size, total_action_dim]
        """
        state_features = self.state_encoder(obs)
        action_features = self.action_encoder(actions)
        
        fused_features = torch.cat([state_features, action_features], dim=-1)
        q_value = self.fusion_net(fused_features)
        
        return q_value


class HAMDTANMADDPGAgent:
    """集成HAM和DTAN的MADDPG智能体"""
    def __init__(self, agent_id, obs_dim, action_dim, num_agents, num_enemies=3,
                 total_obs_dim=None, total_action_dim=None, 
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.95, tau=0.01, hidden_dim=256):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.num_enemies = num_enemies
        self.gamma = gamma
        self.tau = tau
        
        if total_obs_dim is None:
            total_obs_dim = obs_dim
        if total_action_dim is None:
            total_action_dim = action_dim * num_agents
            
        self.total_obs_dim = total_obs_dim
        self.total_action_dim = total_action_dim
        
        # 增强的网络
        self.actor = EnhancedActor(
            obs_dim, action_dim, num_agents, num_enemies, hidden_dim
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        self.critic = EnhancedCritic(
            total_obs_dim, total_action_dim, hidden_dim
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 噪声
        self.noise = OUNoise(action_dim)
        
        # 训练统计
        self.update_count = 0
        
    def select_action(self, obs, add_noise=True, return_info=False):
        """选择动作"""
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, info = self.actor(obs, self.agent_id)
            action = action.cpu().numpy()[0]
        
        if add_noise:
            noise = self.noise.sample()
            action += noise
            action = np.clip(action, -1, 1)
        
        if return_info:
            return action, info
        return action
    
    def soft_update(self, target, source, tau):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def update(self, replay_buffer, other_agents, batch_size=256):
        """更新网络"""
        if len(replay_buffer) < batch_size:
            return {}
        
        # 采样经验
        obs, actions, rewards, next_obs, dones = replay_buffer.sample(batch_size)
        
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # 当前智能体的奖励
        if len(rewards.shape) > 1:
            agent_rewards = rewards[:, self.agent_id]
            agent_dones = dones[:, self.agent_id]
        else:
            agent_rewards = rewards
            agent_dones = dones
        
        # 计算目标Q值
        with torch.no_grad():
            # 获取所有智能体在下一状态的动作
            next_actions = []
            for i, agent in enumerate(other_agents + [self]):
                if i == len(other_agents):  # 当前智能体
                    next_action, _ = agent.actor_target(next_obs, agent.agent_id)
                else:
                    next_action, _ = agent.actor_target(next_obs, agent.agent_id)
                next_actions.append(next_action)
            
            next_actions = torch.cat(next_actions, dim=1)
            target_q = self.critic_target(next_obs, next_actions)
            target_q = agent_rewards.unsqueeze(1) + (1 - agent_dones.unsqueeze(1).float()) * self.gamma * target_q
        
        # 更新Critic
        current_q = self.critic(obs, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # 更新Actor
        actions_pred = actions.clone()
        agent_action, actor_info = self.actor(obs, self.agent_id)
        actions_pred[:, self.agent_id * self.action_dim:(self.agent_id + 1) * self.action_dim] = agent_action
        
        actor_loss = -self.critic(obs, actions_pred).mean()
        
        # 添加注意力正则化损失
        attention_reg_loss = self.compute_attention_regularization(actor_info)
        
        # 添加任务一致性损失
        task_consistency_loss = self.compute_task_consistency_loss(actor_info)
        
        total_actor_loss = actor_loss + 0.01 * attention_reg_loss + 0.01 * task_consistency_loss
        
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)
        
        self.update_count += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'attention_reg_loss': attention_reg_loss.item(),
            'task_consistency_loss': task_consistency_loss.item(),
            'q_value': current_q.mean().item(),
            'update_count': self.update_count
        }
    
    def compute_attention_regularization(self, actor_info):
        """计算注意力正则化损失"""
        # 鼓励注意力分布的稀疏性
        predicted_attention = actor_info['predicted_attention']
        
        # 熵正则化：鼓励注意力集中
        entropy = -(predicted_attention * torch.log(predicted_attention + 1e-8)).sum(dim=-1)
        entropy_loss = entropy.mean()
        
        # 稀疏性正则化：鼓励只关注少数对象
        sparsity_loss = predicted_attention.max(dim=-1)[0].mean()
        
        return entropy_loss - sparsity_loss  # 最小化熵，最大化稀疏性
    
    def compute_task_consistency_loss(self, actor_info):
        """计算任务一致性损失"""
        dtan_results = actor_info['dtan_results']
        
        # 当前智能体的分配
        agent_allocation = dtan_results['allocation_matrix'][:, self.agent_id, :]
        
        # 预测的注意力应该与任务分配一致
        predicted_attention = actor_info['predicted_attention']
        
        # 只考虑敌方目标的注意力（最后num_enemies维）
        enemy_attention = predicted_attention[:, -self.num_enemies:]
        
        # 一致性损失：注意力应该与分配概率成正比
        consistency_loss = F.mse_loss(enemy_attention, agent_allocation)
        
        return consistency_loss


class HAMDTANMADDPGAlgorithm:
    """完整的HAM-DTAN-MADDPG算法"""
    def __init__(self, num_agents=9, obs_dim=78, action_dim=3, num_enemies=3, **kwargs):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_enemies = num_enemies
        
        # 创建所有智能体
        self.agents = []
        for i in range(num_agents):
            agent = HAMDTANMADDPGAgent(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=action_dim,
                num_agents=num_agents,
                num_enemies=num_enemies,
                total_obs_dim=obs_dim,
                total_action_dim=action_dim * num_agents,
                **kwargs
            )
            self.agents.append(agent)
        
        # 共享经验池
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        # 训练统计
        self.training_step = 0
        
    def select_actions(self, obs, add_noise=True, return_info=False):
        """所有智能体选择动作"""
        actions = []
        infos = []
        
        obs_tensor = torch.FloatTensor(obs)
        
        for i, agent in enumerate(self.agents):
            if return_info:
                action, info = agent.select_action(obs, add_noise, return_info=True)
                infos.append(info)
            else:
                action = agent.select_action(obs, add_noise)
            actions.append(action)
        
        if return_info:
            return np.array(actions), infos
        return np.array(actions)
    
    def store_transition(self, obs, actions, rewards, next_obs, dones):
        """存储转移"""
        self.replay_buffer.push(obs, actions, rewards, next_obs, dones)
    
    def update(self, batch_size=256):
        """更新所有智能体"""
        if len(self.replay_buffer) < batch_size:
            return {}
        
        losses = {}
        for i, agent in enumerate(self.agents):
            other_agents = [self.agents[j] for j in range(self.num_agents) if j != i]
            agent_losses = agent.update(self.replay_buffer, other_agents, batch_size)
            
            for key, value in agent_losses.items():
                losses[f'agent_{i}_{key}'] = value
        
        self.training_step += 1
        losses['training_step'] = self.training_step
        
        return losses
    
    def reset_noise(self):
        """重置所有智能体的噪声"""
        for agent in self.agents:
            agent.noise.reset()
    
    def reset_memory(self):
        """重置所有智能体的记忆"""
        for agent in self.agents:
            # 重置HAM的历史
            agent.actor.ham.reset_history()
            agent.actor_target.ham.reset_history()
            
            # 重置DTAN的历史
            agent.actor.dtan.reset_history()
            agent.actor_target.dtan.reset_history()
    
    def save(self, path):
        """保存所有智能体"""
        checkpoint = {}
        for i, agent in enumerate(self.agents):
            checkpoint[f'agent_{i}_actor'] = agent.actor.state_dict()
            checkpoint[f'agent_{i}_critic'] = agent.critic.state_dict()
            checkpoint[f'agent_{i}_actor_optimizer'] = agent.actor_optimizer.state_dict()
            checkpoint[f'agent_{i}_critic_optimizer'] = agent.critic_optimizer.state_dict()
            checkpoint[f'agent_{i}_update_count'] = agent.update_count
        
        checkpoint['training_step'] = self.training_step
        torch.save(checkpoint, path)
    
    def load(self, path):
        """加载所有智能体"""
        checkpoint = torch.load(path, map_location=self.agents[0].device)
        
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(checkpoint[f'agent_{i}_actor'])
            agent.critic.load_state_dict(checkpoint[f'agent_{i}_critic'])
            agent.actor_optimizer.load_state_dict(checkpoint[f'agent_{i}_actor_optimizer'])
            agent.critic_optimizer.load_state_dict(checkpoint[f'agent_{i}_critic_optimizer'])
            agent.update_count = checkpoint[f'agent_{i}_update_count']
            
            # 同步目标网络
            agent.actor_target.load_state_dict(agent.actor.state_dict())
            agent.critic_target.load_state_dict(agent.critic.state_dict())
        
        self.training_step = checkpoint['training_step']
    
    def get_algorithm_info(self):
        """获取算法信息"""
        return {
            'algorithm': 'HAM-DTAN-MADDPG',
            'num_agents': self.num_agents,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'num_enemies': self.num_enemies,
            'training_step': self.training_step,
            'replay_buffer_size': len(self.replay_buffer),
            'total_parameters': sum(
                sum(p.numel() for p in agent.actor.parameters()) + 
                sum(p.numel() for p in agent.critic.parameters())
                for agent in self.agents
            )
        }


# 测试代码
if __name__ == "__main__":
    print("测试HAM-DTAN-MADDPG算法...")
    
    # 创建算法实例
    algorithm = HAMDTANMADDPGAlgorithm()
    
    # 创建随机观测
    obs = np.random.randn(78)
    
    # 选择动作
    actions, infos = algorithm.select_actions(obs, return_info=True)
    
    print(f"动作维度: {actions.shape}")
    print(f"信息数量: {len(infos)}")
    print(f"算法信息: {algorithm.get_algorithm_info()}")
    
    print("HAM-DTAN-MADDPG算法测试完成！")