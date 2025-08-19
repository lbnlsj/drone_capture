import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from collections import deque
import random


class PPONetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(PPONetwork, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 共享特征提取层
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor网络
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, obs):
        shared_features = self.shared_net(obs)
        
        # Actor输出
        action_mean = self.actor_mean(shared_features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        # Critic输出
        value = self.critic(shared_features)
        
        return action_mean, action_std, value
    
    def get_action(self, obs, deterministic=False):
        action_mean, action_std, value = self.forward(obs)
        
        if deterministic:
            action = action_mean
            log_prob = None
        else:
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value
    
    def evaluate_actions(self, obs, actions):
        action_mean, action_std, value = self.forward(obs)
        
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, value.squeeze(), entropy


class PPOBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, obs, action, reward, next_obs, done, log_prob, value):
        experience = (obs, action, reward, next_obs, done, log_prob, value)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done, log_prob, value = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(obs)),
            torch.FloatTensor(np.array(action)),
            torch.FloatTensor(np.array(reward)),
            torch.FloatTensor(np.array(next_obs)),
            torch.BoolTensor(np.array(done)),
            torch.FloatTensor(np.array(log_prob)),
            torch.FloatTensor(np.array(value))
        )
    
    def get_all(self):
        obs, action, reward, next_obs, done, log_prob, value = zip(*self.buffer)
        
        return (
            torch.FloatTensor(np.array(obs)),
            torch.FloatTensor(np.array(action)),
            torch.FloatTensor(np.array(reward)),
            torch.FloatTensor(np.array(next_obs)),
            torch.BoolTensor(np.array(done)),
            torch.FloatTensor(np.array(log_prob)),
            torch.FloatTensor(np.array(value))
        )
    
    def clear(self):
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


class PPOAgent:
    def __init__(self, obs_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2,
                 k_epochs=10, entropy_coef=0.01, value_coef=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # 网络
        self.policy = PPONetwork(obs_dim, action_dim).to(self.device)
        self.policy_old = PPONetwork(obs_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 经验池
        self.buffer = PPOBuffer()
        
        # 训练统计
        self.update_count = 0
        
    def select_action(self, obs, deterministic=False):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.policy_old.get_action(obs, deterministic)
        
        return action.cpu().numpy()[0], log_prob.cpu().item() if log_prob is not None else None, value.cpu().item()
    
    def store_transition(self, obs, action, reward, next_obs, done, log_prob, value):
        self.buffer.push(obs, action, reward, next_obs, done, log_prob, value)
    
    def compute_returns_and_advantages(self, rewards, values, dones, next_values):
        """计算回报和优势函数"""
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # 计算折扣回报
        running_return = next_values[-1]
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        # 计算优势函数 (GAE)
        gae = 0
        lambda_gae = 0.95
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * next_value - values[t]
                gae = delta + self.gamma * lambda_gae * gae
            
            advantages[t] = gae
        
        return returns, advantages
    
    def update(self):
        if len(self.buffer) < 1000:  # 等待足够的经验
            return {}
        
        # 获取所有经验
        obs, actions, rewards, next_obs, dones, old_log_probs, old_values = self.buffer.get_all()
        
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        old_values = old_values.to(self.device)
        
        # 计算next_values
        with torch.no_grad():
            _, _, next_values = self.policy_old(next_obs)
            next_values = next_values.squeeze()
        
        # 计算回报和优势
        returns, advantages = self.compute_returns_and_advantages(rewards, old_values, dones, next_values)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for _ in range(self.k_epochs):
            # 评估当前策略
            log_probs, values, entropy = self.policy.evaluate_actions(obs, actions)
            
            # 计算重要性采样比率
            ratios = torch.exp(log_probs - old_log_probs)
            
            # 计算surrogate损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值函数损失
            value_loss = F.mse_loss(values, returns)
            
            # 熵损失
            entropy_loss = -entropy.mean()
            
            # 总损失
            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
        
        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 清空缓存
        self.buffer.clear()
        
        self.update_count += 1
        
        return {
            'policy_loss': total_policy_loss / self.k_epochs,
            'value_loss': total_value_loss / self.k_epochs,
            'entropy_loss': total_entropy_loss / self.k_epochs,
            'total_loss': (total_policy_loss + total_value_loss + total_entropy_loss) / self.k_epochs
        }
    
    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint['update_count']