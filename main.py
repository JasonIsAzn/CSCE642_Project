import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import random

# -----------------------
# Environment
# -----------------------
class SimpleDispatchEnv(gym.Env):
    def __init__(self, grid_size=10, num_vehicles=1):
        super().__init__()
        self.grid_size = grid_size
        self.num_vehicles = num_vehicles

        # Observation: vehicle positions + request positions
        self.observation_space = spaces.Box(low=0, high=grid_size-1,
                                            shape=(num_vehicles*2 + 2,), dtype=np.int32)

        # Action: move vehicle in 4 directions or stay
        self.action_space = spaces.Discrete(5)

        self.reset()

    def reset(self, seed=None, options=None):
        self.vehicle_pos = np.array([0, 0])
        self.request_pos = np.array([np.random.randint(0, self.grid_size),
                                     np.random.randint(0, self.grid_size)])
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.vehicle_pos, self.request_pos])

    def step(self, action):
        # Move vehicle
        if action == 0:   # up
            self.vehicle_pos[1] = min(self.grid_size-1, self.vehicle_pos[1]+1)
        elif action == 1: # down
            self.vehicle_pos[1] = max(0, self.vehicle_pos[1]-1)
        elif action == 2: # left
            self.vehicle_pos[0] = max(0, self.vehicle_pos[0]-1)
        elif action == 3: # right
            self.vehicle_pos[0] = min(self.grid_size-1, self.vehicle_pos[0]+1)
        # action 4 = stay

        self.steps += 1

        # Reward: negative Manhattan distance to request
        dist = np.sum(np.abs(self.vehicle_pos - self.request_pos))
        reward = -dist

        done = dist == 0 or self.steps >= 50

        return self._get_obs(), reward, done, False, {}

# -----------------------
# Simple DQN network
# -----------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# -----------------------
# Main test loop
# -----------------------
if __name__ == "__main__":
    # Create environment
    env = SimpleDispatchEnv()
    obs, _ = env.reset()

    # Create network
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    net = DQN(state_dim, action_dim)

    print("Initial observation:", obs)
    print("Initial network output:", net(torch.tensor(obs, dtype=torch.float32)))

    # Run a random test episode
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()  # random action
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        print(f"Action: {action}, Obs: {obs}, Reward: {reward}")

    print("Total reward for this episode:", total_reward)
