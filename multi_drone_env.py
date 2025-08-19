import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import math


class Drone:
    def __init__(self, drone_id: int, x: float, y: float, z: float = 0, is_enemy: bool = False):
        self.id = drone_id
        self.x = x
        self.y = y
        self.z = z
        self.is_enemy = is_enemy
        self.is_active = True
        self.speed = 5.0 if not is_enemy else 3.0
        self.required_coverage = 0 if not is_enemy else np.random.randint(2, 5)
        self.current_coverage = 0
        self.intercept_radius = 20.0
        
        if is_enemy:
            self.velocity_x = np.random.uniform(-2, 2)
            self.velocity_y = np.random.uniform(-2, 2)
            self.velocity_z = np.random.uniform(-1, 1)
    
    def update_position(self, action: Optional[np.ndarray] = None):
        if self.is_enemy:
            self.x += self.velocity_x
            self.y += self.velocity_y
            self.z += self.velocity_z
            
            if self.x <= 0 or self.x >= 500:
                self.velocity_x *= -1
            if self.y <= 0 or self.y >= 500:
                self.velocity_y *= -1
            if self.z <= 0 or self.z >= 60:
                self.velocity_z *= -1
                
            self.x = np.clip(self.x, 0, 500)
            self.y = np.clip(self.y, 0, 500)
            self.z = np.clip(self.z, 0, 60)
        else:
            if action is not None:
                dx, dy, dz = action
                self.x = np.clip(self.x + dx * self.speed, 0, 500)
                self.y = np.clip(self.y + dy * self.speed, 0, 500)
                self.z = np.clip(self.z + dz * self.speed, 0, 60)
    
    def distance_to(self, other_drone) -> float:
        return math.sqrt((self.x - other_drone.x)**2 + 
                        (self.y - other_drone.y)**2 + 
                        (self.z - other_drone.z)**2)


class MultiDroneEnvironment(gym.Env):
    def __init__(self, 
                 num_friendly_drones: int = 9,
                 num_enemy_drones: int = 3,
                 map_size: Tuple[int, int, int] = (500, 500, 60)):
        super().__init__()
        
        self.map_size = map_size
        self.num_friendly_drones = num_friendly_drones
        self.num_enemy_drones = num_enemy_drones
        
        self.action_space = spaces.Box(
            low=-1, high=1, 
            shape=(num_friendly_drones, 3), 
            dtype=np.float32
        )
        
        obs_size = (
            num_friendly_drones * 3 +  # 我方无人机位置
            num_enemy_drones * 6 +     # 敌方无人机位置和速度
            num_enemy_drones * 2 +     # 敌方需求覆盖数和当前覆盖数
            num_friendly_drones * num_enemy_drones  # 距离矩阵
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        self.friendly_drones = []
        self.enemy_drones = []
        self.max_steps = 1000
        self.current_step = 0
        
        self.coverage_threshold = 30.0
        self.intercept_threshold = 20.0
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.friendly_drones = []
        self.enemy_drones = []
        
        for i in range(self.num_friendly_drones):
            x = np.random.uniform(50, 150)
            y = np.random.uniform(50, 150)
            z = np.random.uniform(10, 30)
            self.friendly_drones.append(Drone(i, x, y, z, is_enemy=False))
        
        for i in range(self.num_enemy_drones):
            x = np.random.uniform(300, 450)
            y = np.random.uniform(300, 450)
            z = np.random.uniform(20, 50)
            enemy_drone = Drone(i, x, y, z, is_enemy=True)
            enemy_drone.required_coverage = [3, 2, 4][i]  # A需3架、B需2架、C需4架
            self.enemy_drones.append(enemy_drone)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        self.current_step += 1
        
        for i, drone in enumerate(self.friendly_drones):
            drone.update_position(action[i])
        
        for drone in self.enemy_drones:
            drone.update_position()
        
        self._update_coverage()
        
        observation = self._get_observation()
        reward = self._calculate_reward()
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _update_coverage(self):
        for enemy_drone in self.enemy_drones:
            enemy_drone.current_coverage = 0
            for friendly_drone in self.friendly_drones:
                distance = friendly_drone.distance_to(enemy_drone)
                if distance <= self.coverage_threshold:
                    enemy_drone.current_coverage += 1
    
    def _get_observation(self):
        obs = []
        
        for drone in self.friendly_drones:
            obs.extend([drone.x / 500, drone.y / 500, drone.z / 60])
        
        for drone in self.enemy_drones:
            obs.extend([
                drone.x / 500, drone.y / 500, drone.z / 60,
                drone.velocity_x / 5, drone.velocity_y / 5, drone.velocity_z / 5
            ])
        
        for drone in self.enemy_drones:
            obs.extend([
                drone.required_coverage / 10,
                drone.current_coverage / 10
            ])
        
        for friendly in self.friendly_drones:
            for enemy in self.enemy_drones:
                distance = friendly.distance_to(enemy)
                obs.append(distance / 707)  # 归一化最大距离
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(self):
        reward = 0
        
        coverage_reward = 0
        tracking_reward = 0
        efficiency_penalty = 0
        collision_penalty = 0
        
        for enemy_drone in self.enemy_drones:
            coverage_diff = enemy_drone.current_coverage - enemy_drone.required_coverage
            if coverage_diff >= 0:
                coverage_reward += 100
                if coverage_diff == 0:
                    coverage_reward += 50  # 精确覆盖奖励
            else:
                coverage_reward -= abs(coverage_diff) * 20
        
        min_distances = []
        for enemy_drone in self.enemy_drones:
            distances = [f.distance_to(enemy_drone) for f in self.friendly_drones]
            min_distance = min(distances)
            min_distances.append(min_distance)
            
            if min_distance <= self.intercept_threshold:
                tracking_reward += 50
            else:
                tracking_reward -= (min_distance - self.intercept_threshold) * 0.5
        
        total_friendly_in_range = sum(
            1 for f in self.friendly_drones 
            for e in self.enemy_drones 
            if f.distance_to(e) <= self.coverage_threshold
        )
        total_required = sum(e.required_coverage for e in self.enemy_drones)
        if total_friendly_in_range > total_required:
            efficiency_penalty = -(total_friendly_in_range - total_required) * 10
        
        for i, drone1 in enumerate(self.friendly_drones):
            for drone2 in self.friendly_drones[i+1:]:
                if drone1.distance_to(drone2) < 10:
                    collision_penalty -= 30
        
        reward = coverage_reward + tracking_reward + efficiency_penalty + collision_penalty
        
        return reward
    
    def _is_terminated(self):
        all_covered = all(
            enemy.current_coverage >= enemy.required_coverage 
            for enemy in self.enemy_drones
        )
        
        all_intercepted = all(
            min(f.distance_to(enemy) for f in self.friendly_drones) <= self.intercept_threshold
            for enemy in self.enemy_drones
        )
        
        return all_covered and all_intercepted
    
    def _get_info(self):
        return {
            'coverage_status': [
                {
                    'enemy_id': i,
                    'required': enemy.required_coverage,
                    'current': enemy.current_coverage,
                    'satisfied': enemy.current_coverage >= enemy.required_coverage
                }
                for i, enemy in enumerate(self.enemy_drones)
            ],
            'intercept_status': [
                {
                    'enemy_id': i,
                    'min_distance': min(f.distance_to(enemy) for f in self.friendly_drones),
                    'intercepted': min(f.distance_to(enemy) for f in self.friendly_drones) <= self.intercept_threshold
                }
                for i, enemy in enumerate(self.enemy_drones)
            ],
            'step': self.current_step
        }
    
    def render(self, mode='human'):
        if not hasattr(self, 'fig'):
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 7))
            plt.ion()
        
        self.ax1.clear()
        self.ax2.clear()
        
        # 2D 视图 (x-y平面)
        self.ax1.set_xlim(0, 500)
        self.ax1.set_ylim(0, 500)
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        self.ax1.set_title('Multi-Drone Environment - Top View')
        self.ax1.grid(True, alpha=0.3)
        
        # 绘制我方无人机
        for drone in self.friendly_drones:
            circle = patches.Circle((drone.x, drone.y), 8, color='blue', alpha=0.7)
            self.ax1.add_patch(circle)
            self.ax1.text(drone.x, drone.y, str(drone.id), ha='center', va='center', 
                         fontsize=8, color='white', weight='bold')
        
        # 绘制敌方无人机及其需求
        colors = ['red', 'orange', 'purple']
        for i, drone in enumerate(self.enemy_drones):
            circle = patches.Circle((drone.x, drone.y), 12, color=colors[i], alpha=0.8)
            self.ax1.add_patch(circle)
            
            coverage_circle = patches.Circle((drone.x, drone.y), self.coverage_threshold, 
                                           fill=False, color=colors[i], linestyle='--', alpha=0.5)
            self.ax1.add_patch(coverage_circle)
            
            self.ax1.text(drone.x, drone.y-20, f'{chr(65+i)}\n{drone.current_coverage}/{drone.required_coverage}',
                         ha='center', va='center', fontsize=10, weight='bold')
        
        # 侧视图 (x-z平面)
        self.ax2.set_xlim(0, 500)
        self.ax2.set_ylim(0, 60)
        self.ax2.set_xlabel('X')
        self.ax2.set_ylabel('Z (Height)')
        self.ax2.set_title('Multi-Drone Environment - Side View')
        self.ax2.grid(True, alpha=0.3)
        
        # 绘制我方无人机高度
        for drone in self.friendly_drones:
            self.ax2.scatter(drone.x, drone.z, c='blue', s=80, alpha=0.7, marker='o')
            self.ax2.text(drone.x, drone.z+2, str(drone.id), ha='center', va='bottom', fontsize=8)
        
        # 绘制敌方无人机高度
        for i, drone in enumerate(self.enemy_drones):
            self.ax2.scatter(drone.x, drone.z, c=colors[i], s=120, alpha=0.8, marker='s')
            self.ax2.text(drone.x, drone.z+3, chr(65+i), ha='center', va='bottom', 
                         fontsize=10, weight='bold')
        
        plt.tight_layout()
        plt.pause(0.01)
        
        if mode == 'rgb_array':
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return image
    
    def close(self):
        if hasattr(self, 'fig'):
            plt.close(self.fig)