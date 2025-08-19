import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import copy


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    def __init__(self, total_obs_dim, total_action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=-1)
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, actions, rewards, next_obs, dones):
        experience = (obs, actions, rewards, next_obs, dones)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        
        return (
            np.array(obs),
            np.array(actions),
            np.array(rewards),
            np.array(next_obs),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class OUNoise:
    """Ornstein-Uhlenbeck噪声，用于连续动作探索"""
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


class MADDPGAgent:
    def __init__(self, agent_id, obs_dim, action_dim, num_agents, total_obs_dim=None, total_action_dim=None,
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.95, tau=0.01, hidden_dim=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        
        # 如果没有提供全局观测和动作维度，使用默认值
        if total_obs_dim is None:
            total_obs_dim = obs_dim * num_agents
        if total_action_dim is None:
            total_action_dim = action_dim * num_agents
            
        self.total_obs_dim = total_obs_dim
        self.total_action_dim = total_action_dim
        
        # Actor网络（只使用局部观测）
        self.actor = Actor(obs_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic网络（使用全局观测和动作）
        self.critic = Critic(total_obs_dim, total_action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 噪声
        self.noise = OUNoise(action_dim)
        
    def select_action(self, obs, add_noise=True):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()[0]
        
        if add_noise:
            noise = self.noise.sample()
            action += noise
            action = np.clip(action, -1, 1)
        
        return action
    
    def soft_update(self, target, source, tau):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def update(self, replay_buffer, other_agents, batch_size=256):
        if len(replay_buffer) < batch_size:
            return {}
        
        # 采样经验
        obs, actions, rewards, next_obs, dones = replay_buffer.sample(batch_size)
        
        # 转换为tensor
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # 获取当前智能体的奖励
        agent_rewards = rewards[:, self.agent_id]
        agent_dones = dones[:, self.agent_id]
        
        # 计算目标Q值
        with torch.no_grad():
            # 获取所有智能体在下一状态的动作
            next_actions = []
            for i, agent in enumerate(other_agents + [self]):
                if i == len(other_agents):  # 当前智能体
                    agent_obs = next_obs[:, i * self.obs_dim:(i + 1) * self.obs_dim]
                    next_action = agent.actor_target(agent_obs)
                else:
                    agent_obs = next_obs[:, i * self.obs_dim:(i + 1) * self.obs_dim]
                    next_action = agent.actor_target(agent_obs)
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
        # 获取当前智能体的观测
        agent_obs = obs[:, self.agent_id * self.obs_dim:(self.agent_id + 1) * self.obs_dim]
        
        # 计算当前策略的动作
        actions_pred = actions.clone()
        actions_pred[:, self.agent_id * self.action_dim:(self.agent_id + 1) * self.action_dim] = self.actor(agent_obs)
        
        # Actor损失
        actor_loss = -self.critic(obs, actions_pred).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_value': current_q.mean().item()
        }


class MADDPG:
    def __init__(self, num_agents, obs_dim, action_dim, total_obs_dim=None, total_action_dim=None, **kwargs):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 计算总观测和动作维度
        if total_obs_dim is None:
            total_obs_dim = obs_dim * num_agents
        if total_action_dim is None:
            total_action_dim = action_dim * num_agents
        
        # 创建所有智能体
        self.agents = []
        for i in range(num_agents):
            agent = MADDPGAgent(
                agent_id=i,
                obs_dim=obs_dim,
                action_dim=action_dim,
                num_agents=num_agents,
                total_obs_dim=total_obs_dim,
                total_action_dim=total_action_dim,
                **kwargs
            )
            self.agents.append(agent)
        
        # 共享经验池
        self.replay_buffer = ReplayBuffer()
        
    def select_actions(self, observations, add_noise=True):
        """所有智能体选择动作"""
        actions = []
        for i, agent in enumerate(self.agents):
            obs = observations[i * self.obs_dim:(i + 1) * self.obs_dim]
            action = agent.select_action(obs, add_noise)
            actions.append(action)
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
        
        return losses
    
    def reset_noise(self):
        """重置所有智能体的噪声"""
        for agent in self.agents:
            agent.noise.reset()
    
    def save(self, path):
        """保存所有智能体"""
        checkpoint = {}
        for i, agent in enumerate(self.agents):
            checkpoint[f'agent_{i}_actor'] = agent.actor.state_dict()
            checkpoint[f'agent_{i}_critic'] = agent.critic.state_dict()
            checkpoint[f'agent_{i}_actor_optimizer'] = agent.actor_optimizer.state_dict()
            checkpoint[f'agent_{i}_critic_optimizer'] = agent.critic_optimizer.state_dict()
        
        torch.save(checkpoint, path)
    
    def load(self, path):
        """加载所有智能体"""
        checkpoint = torch.load(path, map_location=self.agents[0].device)
        
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(checkpoint[f'agent_{i}_actor'])
            agent.critic.load_state_dict(checkpoint[f'agent_{i}_critic'])
            agent.actor_optimizer.load_state_dict(checkpoint[f'agent_{i}_actor_optimizer'])
            agent.critic_optimizer.load_state_dict(checkpoint[f'agent_{i}_critic_optimizer'])
            
            # 同步目标网络
            agent.actor_target.load_state_dict(agent.actor.state_dict())
            agent.critic_target.load_state_dict(agent.critic.state_dict())