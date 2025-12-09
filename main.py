import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt

# -----------------------
# Experience Replay Buffer
# -----------------------
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

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
                                            shape=(num_vehicles*2 + 2,), dtype=np.float32)

        # Action: move vehicle in 4 directions or stay
        self.action_space = spaces.Discrete(5)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.vehicle_pos = np.array([0, 0], dtype=np.float32)
        self.request_pos = np.array([np.random.randint(0, self.grid_size),
                                     np.random.randint(0, self.grid_size)], dtype=np.float32)
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
        
        # Better reward shaping
        reward = -dist * 0.1  # Small penalty for distance
        
        # Big bonus for reaching the request
        if dist == 0:
            reward += 10.0
        
        done = dist == 0 or self.steps >= 50
        
        # Penalty for timeout
        if self.steps >= 50 and dist > 0:
            reward -= 5.0

        return self._get_obs(), reward, done, False, {}

# -----------------------
# DQN Network (Enhanced)
# -----------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# -----------------------
# DQN Agent
# -----------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-network and target network
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity=10000)
        
    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def store_transition(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)
    
    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None
        
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state))
        action_batch = torch.LongTensor(batch.action).unsqueeze(1)
        reward_batch = torch.FloatTensor(batch.reward)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state))
        done_batch = torch.FloatTensor(batch.done)
        
        # Current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# -----------------------
# Training Loop
# -----------------------
def train_dqn(env, agent, num_episodes=1000, batch_size=64, 
              target_update_freq=10, print_freq=50):
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Select and perform action
            action = agent.select_action(state, training=True)
            next_state, reward, done, _, _ = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, next_state, reward, done)
            
            # Train the agent
            loss = agent.train_step(batch_size)
            if loss is not None:
                losses.append(loss)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # Update epsilon
        agent.update_epsilon()
        
        # Update target network
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Print progress
        if (episode + 1) % print_freq == 0:
            avg_reward = np.mean(episode_rewards[-print_freq:])
            avg_length = np.mean(episode_lengths[-print_freq:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}")
    
    return episode_rewards, episode_lengths, losses

# -----------------------
# Evaluation
# -----------------------
def evaluate_agent(env, agent, num_episodes=100):
    total_rewards = []
    success_count = 0
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, done, _, _ = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        if episode_reward > 5:  # Consider it success if reward is positive
            success_count += 1
    
    avg_reward = np.mean(total_rewards)
    success_rate = success_count / num_episodes * 100
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    return avg_reward, success_rate

# -----------------------
# Visualization
# -----------------------
def plot_training_results(episode_rewards, episode_lengths):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot rewards
    ax1.plot(episode_rewards, alpha=0.6, linewidth=0.5)
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label=f'{window}-episode MA')
        ax1.legend()
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Rewards')
    ax1.grid(True, alpha=0.3)
    
    # Plot episode lengths
    ax2.plot(episode_lengths, alpha=0.6, linewidth=0.5)
    if len(episode_lengths) >= window:
        moving_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_lengths)), moving_avg, 'r-', linewidth=2, label=f'{window}-episode MA')
        ax2.legend()
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Episode Lengths')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    print("\nTraining plot saved as 'training_results.png'")

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create environment
    env = SimpleDispatchEnv(grid_size=10, num_vehicles=1)
    
    # Create agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, lr=1e-3, gamma=0.99)
    
    print("=" * 60)
    print("Starting DQN Training for Dispatch Environment")
    print("=" * 60)
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Grid size: {env.grid_size}x{env.grid_size}")
    print("=" * 60)
    
    # Train the agent
    episode_rewards, episode_lengths, losses = train_dqn(
        env, agent, 
        num_episodes=1000, 
        batch_size=64,
        target_update_freq=10,
        print_freq=50
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Evaluate the trained agent
    evaluate_agent(env, agent, num_episodes=100)
    
    # Plot results
    plot_training_results(episode_rewards, episode_lengths)
    
    # Save the model
    torch.save(agent.policy_net.state_dict(), 'dqn_dispatch_model.pth')
    print("\nModel saved as 'dqn_dispatch_model.pth'")
    
    # Demonstrate a trained episode
    print("\n" + "=" * 60)
    print("Demonstrating Trained Agent:")
    print("=" * 60)
    state, _ = env.reset()
    done = False
    step = 0
    total_reward = 0
    
    print(f"Initial - Vehicle: {env.vehicle_pos}, Request: {env.request_pos}")
    
    while not done and step < 20:
        action = agent.select_action(state, training=False)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        step += 1
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']
        print(f"Step {step}: Action={action_names[action]}, Vehicle={env.vehicle_pos}, Reward={reward:.2f}")
    
    print(f"\nFinal Total Reward: {total_reward:.2f}")
    print("Demo complete!")