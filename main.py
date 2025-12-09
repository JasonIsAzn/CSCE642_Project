import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Patch
from IPython.display import HTML
import time

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
# Multi-Vehicle/Request Environment
# -----------------------
class MultiDispatchEnv(gym.Env):
    def __init__(self, grid_size=10, num_vehicles=3, num_requests=3):
        super().__init__()
        self.grid_size = grid_size
        self.num_vehicles = num_vehicles
        self.num_requests = num_requests

        # Observation: all vehicle positions + all request positions + assignment status
        obs_size = num_vehicles * 2 + num_requests * 3  # positions + (x, y, active)
        self.observation_space = spaces.Box(low=-1, high=grid_size,
                                            shape=(obs_size,), dtype=np.float32)

        # Action: which vehicle to move and in which direction
        # Actions: vehicle_id * 5 directions (up, down, left, right, stay)
        self.action_space = spaces.Discrete(num_vehicles * 5)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize vehicles at random positions
        self.vehicle_positions = np.array([
            [np.random.randint(0, self.grid_size), 
             np.random.randint(0, self.grid_size)]
            for _ in range(self.num_vehicles)
        ], dtype=np.float32)
        
        # Initialize requests at random positions
        self.request_positions = np.array([
            [np.random.randint(0, self.grid_size), 
             np.random.randint(0, self.grid_size)]
            for _ in range(self.num_requests)
        ], dtype=np.float32)
        
        # Track which requests are still active
        self.request_active = np.ones(self.num_requests, dtype=bool)
        
        # Track assignments (which vehicle is assigned to which request, -1 = unassigned)
        self.assignments = np.full(self.num_vehicles, -1, dtype=int)
        
        self.steps = 0
        self.completed_requests = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Flatten vehicle positions
        vehicle_obs = self.vehicle_positions.flatten()
        
        # Request observations: (x, y, active)
        request_obs = []
        for i in range(self.num_requests):
            if self.request_active[i]:
                request_obs.extend([self.request_positions[i, 0], 
                                   self.request_positions[i, 1], 1.0])
            else:
                request_obs.extend([-1.0, -1.0, 0.0])  # Inactive request marker
        
        return np.concatenate([vehicle_obs, request_obs])

    def _assign_requests(self):
        """Simple greedy assignment: assign each vehicle to nearest unassigned request"""
        for v_idx in range(self.num_vehicles):
            if self.assignments[v_idx] == -1:  # Vehicle not assigned
                # Find nearest active request
                min_dist = float('inf')
                best_req = -1
                
                for r_idx in range(self.num_requests):
                    if self.request_active[r_idx] and r_idx not in self.assignments:
                        dist = np.sum(np.abs(self.vehicle_positions[v_idx] - 
                                            self.request_positions[r_idx]))
                        if dist < min_dist:
                            min_dist = dist
                            best_req = r_idx
                
                if best_req != -1:
                    self.assignments[v_idx] = best_req

    def step(self, action):
        # Decode action: which vehicle and which direction
        vehicle_id = action // 5
        direction = action % 5
        
        # Move the selected vehicle
        if direction == 0:   # up
            self.vehicle_positions[vehicle_id, 1] = min(self.grid_size-1, 
                                                        self.vehicle_positions[vehicle_id, 1]+1)
        elif direction == 1: # down
            self.vehicle_positions[vehicle_id, 1] = max(0, 
                                                        self.vehicle_positions[vehicle_id, 1]-1)
        elif direction == 2: # left
            self.vehicle_positions[vehicle_id, 0] = max(0, 
                                                        self.vehicle_positions[vehicle_id, 0]-1)
        elif direction == 3: # right
            self.vehicle_positions[vehicle_id, 0] = min(self.grid_size-1, 
                                                        self.vehicle_positions[vehicle_id, 0]+1)
        # direction 4 = stay

        self.steps += 1
        
        # Update assignments
        self._assign_requests()
        
        # Calculate reward
        reward = 0
        
        # Check if any vehicle reached their assigned request
        for v_idx in range(self.num_vehicles):
            if self.assignments[v_idx] != -1:
                assigned_req = self.assignments[v_idx]
                dist = np.sum(np.abs(self.vehicle_positions[v_idx] - 
                                    self.request_positions[assigned_req]))
                
                # Distance penalty
                reward -= dist * 0.05
                
                # Check if vehicle reached request
                if dist == 0:
                    reward += 20.0  # Big reward for completing request
                    self.request_active[assigned_req] = False
                    self.assignments[v_idx] = -1
                    self.completed_requests += 1
        
        # Small penalty for each step to encourage efficiency
        reward -= 0.1
        
        # Check if done
        done = (not np.any(self.request_active)) or self.steps >= 100
        
        # Penalty for timeout
        if self.steps >= 100 and np.any(self.request_active):
            reward -= 10.0

        return self._get_obs(), reward, done, False, {'completed': self.completed_requests}

# -----------------------
# DQN Network (Enhanced for larger state space)
# -----------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
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
        self.memory = ReplayBuffer(capacity=50000)  # Larger buffer for complex problem
        
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
    
    def train_step(self, batch_size=128):
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
# Visualization Functions
# -----------------------
def visualize_episode_static(env, agent, save_path='multi_episode_steps.png'):
    """Create a step-by-step visualization of a multi-vehicle episode"""
    state, _ = env.reset()
    done = False
    
    # Record the episode
    trajectory = [{
        'vehicles': env.vehicle_positions.copy(),
        'requests': env.request_positions.copy(),
        'active': env.request_active.copy(),
        'action': None,
        'reward': 0,
        'completed': 0
    }]
    
    while not done:
        action = agent.select_action(state, training=False)
        state, reward, done, _, info = env.step(action)
        
        trajectory.append({
            'vehicles': env.vehicle_positions.copy(),
            'requests': env.request_positions.copy(),
            'active': env.request_active.copy(),
            'action': action,
            'reward': reward,
            'completed': info['completed']
        })
    
    # Create visualization
    num_steps = min(20, len(trajectory))  # Limit to 20 frames
    cols = 5
    rows = (num_steps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    colors = ['blue', 'green', 'purple', 'orange', 'cyan']
    
    for idx in range(num_steps):
        ax = axes[idx]
        data = trajectory[idx]
        
        # Create empty grid
        ax.set_xlim(-0.5, env.grid_size-0.5)
        ax.set_ylim(-0.5, env.grid_size-0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(env.grid_size))
        ax.set_yticks(range(env.grid_size))
        
        # Plot requests (active ones as stars)
        for r_idx in range(env.num_requests):
            if data['active'][r_idx]:
                ax.plot(data['requests'][r_idx, 0], data['requests'][r_idx, 1], 
                       'r*', markersize=25, markeredgecolor='darkred', markeredgewidth=2)
        
        # Plot vehicles (circles with different colors)
        for v_idx in range(env.num_vehicles):
            color = colors[v_idx % len(colors)]
            ax.plot(data['vehicles'][v_idx, 0], data['vehicles'][v_idx, 1], 
                   'o', color=color, markersize=15, markeredgecolor='black', 
                   markeredgewidth=2, label=f'V{v_idx+1}' if idx == 0 else '')
        
        # Title
        action_str = f"A{data['action']}" if data['action'] is not None else "START"
        ax.set_title(f"Step {idx}\n{action_str} | R={data['reward']:.1f}\nCompleted: {data['completed']}", 
                    fontsize=9)
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=7)
    
    # Hide unused subplots
    for idx in range(num_steps, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Multi-Vehicle Dispatch Episode', fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Multi-vehicle episode visualization saved as '{save_path}'")
    plt.close()

def visualize_episode_animated(env, agent, save_path='multi_episode_animation.gif', fps=2):
    """Create an animated GIF of a multi-vehicle episode"""
    state, _ = env.reset()
    done = False
    
    # Record the episode
    trajectory = []
    step = 0
    
    trajectory.append({
        'vehicles': env.vehicle_positions.copy(),
        'requests': env.request_positions.copy(),
        'active': env.request_active.copy(),
        'action': None,
        'reward': 0,
        'step': 0,
        'completed': 0
    })
    
    while not done:
        action = agent.select_action(state, training=False)
        state, reward, done, _, info = env.step(action)
        step += 1
        
        trajectory.append({
            'vehicles': env.vehicle_positions.copy(),
            'requests': env.request_positions.copy(),
            'active': env.request_active.copy(),
            'action': action,
            'reward': reward,
            'step': step,
            'completed': info['completed']
        })
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ['blue', 'green', 'purple', 'orange', 'cyan']
    
    def update(frame):
        ax.clear()
        data = trajectory[frame]
        
        ax.set_xlim(-0.5, env.grid_size-0.5)
        ax.set_ylim(-0.5, env.grid_size-0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_xticks(range(env.grid_size))
        ax.set_yticks(range(env.grid_size))
        
        # Plot active requests
        for r_idx in range(env.num_requests):
            if data['active'][r_idx]:
                ax.plot(data['requests'][r_idx, 0], data['requests'][r_idx, 1], 
                       'r*', markersize=40, markeredgecolor='darkred', 
                       markeredgewidth=3, label='Request' if r_idx == 0 else '')
        
        # Plot vehicles
        for v_idx in range(env.num_vehicles):
            color = colors[v_idx % len(colors)]
            ax.plot(data['vehicles'][v_idx, 0], data['vehicles'][v_idx, 1], 
                   'o', color=color, markersize=25, markeredgecolor='black', 
                   markeredgewidth=2, label=f'Vehicle {v_idx+1}')
            
            # Add vehicle number
            ax.text(data['vehicles'][v_idx, 0], data['vehicles'][v_idx, 1], 
                   str(v_idx+1), ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white')
        
        # Title
        vehicle_id = data['action'] // 5 if data['action'] is not None else 0
        direction = data['action'] % 5 if data['action'] is not None else 0
        direction_names = ['UP ↑', 'DOWN ↓', 'LEFT ←', 'RIGHT →', 'STAY ⊗']
        action_str = f"V{vehicle_id+1}: {direction_names[direction]}" if data['action'] is not None else 'START'
        
        title = f"Step {data['step']}/{len(trajectory)-1} | Action: {action_str}\n"
        title += f"Completed: {data['completed']}/{env.num_requests} | Reward: {data['reward']:.2f}"
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlabel('X Position', fontsize=11)
        ax.set_ylabel('Y Position', fontsize=11)
    
    anim = FuncAnimation(fig, update, frames=len(trajectory), repeat=True, interval=1000//fps)
    anim.save(save_path, writer='pillow', fps=fps)
    print(f"Multi-vehicle animated episode saved as '{save_path}'")
    plt.close()

# -----------------------
# Training Loop
# -----------------------
def train_dqn(env, agent, num_episodes=2000, batch_size=128, 
              target_update_freq=10, print_freq=100):
    episode_rewards = []
    episode_lengths = []
    episode_completions = []
    losses = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, done, _, info = env.step(action)
            
            agent.store_transition(state, action, next_state, reward, done)
            
            loss = agent.train_step(batch_size)
            if loss is not None:
                losses.append(loss)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        agent.update_epsilon()
        
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_completions.append(info['completed'])
        
        if (episode + 1) % print_freq == 0:
            avg_reward = np.mean(episode_rewards[-print_freq:])
            avg_length = np.mean(episode_lengths[-print_freq:])
            avg_completions = np.mean(episode_completions[-print_freq:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Completed: {avg_completions:.2f}/{env.num_requests} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}")
    
    return episode_rewards, episode_lengths, episode_completions, losses

# -----------------------
# Evaluation
# -----------------------
def evaluate_agent(env, agent, num_episodes=100):
    total_rewards = []
    total_completions = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, done, _, info = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        total_completions.append(info['completed'])
    
    avg_reward = np.mean(total_rewards)
    avg_completions = np.mean(total_completions)
    success_rate = (np.array(total_completions) == env.num_requests).sum() / num_episodes * 100
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Completions: {avg_completions:.2f}/{env.num_requests}")
    print(f"Full Success Rate: {success_rate:.1f}%")
    
    return avg_reward, avg_completions, success_rate

# -----------------------
# Training Visualization
# -----------------------
def plot_training_results(episode_rewards, episode_lengths, episode_completions, num_requests):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    window = 50
    
    # Plot rewards
    ax1.plot(episode_rewards, alpha=0.4, linewidth=0.5)
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label=f'{window}-ep MA')
        ax1.legend()
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Rewards')
    ax1.grid(True, alpha=0.3)
    
    # Plot completions
    ax2.plot(episode_completions, alpha=0.4, linewidth=0.5)
    if len(episode_completions) >= window:
        moving_avg = np.convolve(episode_completions, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_completions)), moving_avg, 'g-', linewidth=2, label=f'{window}-ep MA')
        ax2.legend()
    ax2.axhline(y=num_requests, color='r', linestyle='--', label='All Requests')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Completed Requests')
    ax2.set_title('Requests Completed per Episode')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot episode lengths
    ax3.plot(episode_lengths, alpha=0.4, linewidth=0.5)
    if len(episode_lengths) >= window:
        moving_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(episode_lengths)), moving_avg, 'b-', linewidth=2, label=f'{window}-ep MA')
        ax3.legend()
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Episode Length')
    ax3.set_title('Episode Lengths')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_training_results.png', dpi=150)
    print("\nTraining plot saved as 'multi_training_results.png'")
    plt.close()

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Create multi-vehicle environment
    NUM_VEHICLES = 3
    NUM_REQUESTS = 3
    env = MultiDispatchEnv(grid_size=10, num_vehicles=NUM_VEHICLES, num_requests=NUM_REQUESTS)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, lr=5e-4, gamma=0.99)
    
    print("=" * 60)
    print("Multi-Vehicle DQN Training for Dispatch Environment")
    print("=" * 60)
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Grid size: {env.grid_size}x{env.grid_size}")
    print(f"Number of vehicles: {NUM_VEHICLES}")
    print(f"Number of requests: {NUM_REQUESTS}")
    print("=" * 60)
    
    # Train
    episode_rewards, episode_lengths, episode_completions, losses = train_dqn(
        env, agent, 
        num_episodes=2000,
        batch_size=128,
        target_update_freq=10,
        print_freq=100
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Evaluate
    evaluate_agent(env, agent, num_episodes=100)
    
    # Plot results
    plot_training_results(episode_rewards, episode_lengths, episode_completions, NUM_REQUESTS)
    
    # Save model
    torch.save(agent.policy_net.state_dict(), 'dqn_multi_dispatch_model.pth')
    print("\nModel saved as 'dqn_multi_dispatch_model.pth'")
    
    # Visualizations
    print("\n" + "=" * 60)
    print("Creating Multi-Vehicle Visualizations...")
    print("=" * 60)
    
    print("\n1. Creating static visualization...")
    visualize_episode_static(env, agent, save_path='multi_episode_steps.png')
    
    print("\n2. Creating animated GIF...")
    visualize_episode_animated(env, agent, save_path='multi_episode_animation.gif', fps=2)
    
    print("\n" + "=" * 60)
    print("All Done! Files created:")
    print("  - multi_training_results.png")
    print("  - multi_episode_steps.png")
    print("  - multi_episode_animation.gif")
    print("  - dqn_multi_dispatch_model.pth")
    print("=" * 60)