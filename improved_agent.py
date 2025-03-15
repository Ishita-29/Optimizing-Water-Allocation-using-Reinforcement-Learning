import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt
from collections import deque
import random
import os
import time

def configure_logging():
    """
    Configure logging for the DQN agent
    
    Returns:
    --------
    logging.Logger: Configured logger
    """
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

# Check for CUDA availability
def get_device():
    """
    Get the available device (CUDA or CPU)
    
    Returns:
    --------
    torch.device: Device for PyTorch tensors
    """
    logger = configure_logging()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.warning("No GPU available. Using CPU.")
    return device

class DQN(nn.Module):
    """
    Deep Q-Network model implemented in PyTorch
    """
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128, 64]):
        super(DQN, self).__init__()
        
        # Input normalization (learn mean and std)
        self.input_bn = nn.BatchNorm1d(state_dim)
        
        # First hidden layer
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        
        # Second hidden layer
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        
        # Third hidden layer
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        
        # Output layer
        self.fc4 = nn.Linear(hidden_dims[2], action_dim)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network
        """
        # Input normalization
        if x.size(0) > 1:  # Only apply batch norm for batch size > 1
            x = self.input_bn(x)
        
        # Layer 1
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # Layer 3
        x = self.fc3(x)
        if x.size(0) > 1:
            x = self.bn3(x)
        x = self.leaky_relu(x)
        
        # Output layer - no activation for Q-values
        x = self.fc4(x)
        
        return x

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for more efficient learning
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment  # Beta annealing
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0  # Max priority for new experiences
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer with max priority"""
        max_priority = self.max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        # New experiences get max priority to ensure they're sampled
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample batch based on priorities"""
        if len(self.buffer) < self.capacity:
            priorities = self.priorities[:len(self.buffer)]
        else:
            priorities = self.priorities
        
        # Probabilities based on priorities
        probs = priorities ** self.alpha
        probs = probs / np.sum(probs)
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)  # Normalize
        
        # Increment beta for annealing
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get samples
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            state, action, reward, next_state, done = self.buffer[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        return (states, actions, rewards, next_states, dones), indices, weights
    
    def update_priorities(self, indices, errors):
        """Update priorities based on TD errors"""
        for i, error in zip(indices, errors):
            # Add small value to prevent zero priority
            self.priorities[i] = (abs(error) + 1e-5) ** self.alpha
            self.max_priority = max(self.max_priority, self.priorities[i])
    
    def __len__(self):
        """Return current buffer size"""
        return len(self.buffer)

class WaterAllocationDQNAgent:
    """
    Double DQN Agent with Prioritized Experience Replay
    """
    def __init__(self, state_dim, action_dim):
        """
        Initialize DQN Agent
        
        Parameters:
        -----------
        state_dim : int
            Dimension of the state space
        action_dim : int
            Number of possible actions
        """
        # Get device (GPU or CPU)
        self.device = get_device()
        
        # Logging
        self.logger = configure_logging()
        
        # Environment parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Hyperparameters
        self.learning_rate = 0.001
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Start with full exploration
        self.epsilon_decay = 0.995  # Slower decay
        self.epsilon_min = 0.05  # Higher minimum for continued exploration
        self.batch_size = 64
        self.tau = 0.005  # Soft update parameter
        self.update_target_every = 10  # Update target network every N episodes
        
        # Memory for prioritized experience replay
        self.memory = PrioritizedReplayBuffer(capacity=20000, alpha=0.6, beta=0.4)
        
        # Build neural network models
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in evaluation mode
        
        # Optimizer with learning rate scheduling
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.5)
        
        # Training tracking
        self.training_history = {
            'episode_rewards': [],
            'epsilon_values': [],
            'loss_values': [],
            'avg_q_values': [],
            'learning_rates': []
        }
        
        # Training steps
        self.training_steps = 0
        
        self.logger.info(f"Improved Double DQN agent initialized on {self.device}")
        self.logger.info(f"Network architecture: {self.policy_net}")
    
    def update_target_model(self):
        """
        Hard update of target network parameters
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def soft_update_target_model(self):
        """
        Soft update of target network parameters:
        θ_target = τ*θ_policy + (1-τ)*θ_target
        """
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : numpy.ndarray
            Next state
        done : bool
            Episode termination flag
        """
        # Ensure state and next_state are flattened
        state = np.array(state, dtype=np.float32).flatten()
        next_state = np.array(next_state, dtype=np.float32).flatten() if next_state is not None else None
        
        # Add to prioritized replay buffer
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state, evaluate=False):
        """
        Choose action using epsilon-greedy strategy
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state
        evaluate : bool
            Whether to use greedy policy (for evaluation)
            
        Returns:
        --------
        int: Selected action index
        """
        # Exploration (skip during evaluation)
        if not evaluate and np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)
        
        # Ensure state is flattened
        state = np.array(state, dtype=np.float32).flatten()
        
        # Convert state to PyTorch tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Exploitation - use policy network
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        self.policy_net.train()
        
        return q_values.argmax().item()
    
    def replay(self):
        """
        Experience replay with prioritized sampling and Double DQN
        
        Returns:
        --------
        float: Loss value
        """
        # Check if enough experiences are available
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample batch from memory with priorities
        (states, actions, rewards, next_states, dones), indices, weights = self.memory.sample(self.batch_size)
        
        # Convert to PyTorch tensors
        state_batch = torch.FloatTensor(np.array(states)).to(self.device)
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        weights_tensor = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        # Handle None values in next_states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), 
                                     device=self.device, dtype=torch.bool)
        non_final_next_states = torch.FloatTensor([s for s in next_states if s is not None]).to(self.device)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Double DQN: use policy net to select actions, target net to evaluate them
        next_state_values = torch.zeros(self.batch_size, 1, device=self.device)
        
        with torch.no_grad():
            # Select actions using policy network
            if sum(non_final_mask) > 0:  # Check if there are any non-final states
                next_actions = self.policy_net(non_final_next_states).max(1, keepdim=True)[1]
                
                # Evaluate Q-values using target network
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, next_actions)
        
        # Compute the expected Q values: Q'(s,a) = r + γ * max_a' Q(s',a')
        expected_state_action_values = reward_batch + (self.gamma * next_state_values * (~torch.tensor(dones, dtype=torch.bool, device=self.device).unsqueeze(1)))
        
        # Compute Huber loss (more robust than MSE)
        td_errors = torch.abs(state_action_values - expected_state_action_values).detach().cpu().numpy()
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction='none')
        
        # Apply importance sampling weights
        weighted_loss = (loss * weights_tensor).mean()
        
        # Update priorities in buffer
        self.memory.update_priorities(indices, td_errors)
        
        # Optimize the model
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update training steps and perform target network update
        self.training_steps += 1
        
        # Track average Q-values
        avg_q = self.policy_net(state_batch).mean().item()
        
        # Soft update target network
        self.soft_update_target_model()
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return weighted_loss.item(), avg_q
    
    def train(self, env, episodes=500):
        """
        Train the agent in the given environment
        
        Parameters:
        -----------
        env : WaterAllocationEnvironment
            Environment to train in
        episodes : int, default=500
            Number of training episodes
        
        Returns:
        --------
        dict: Training history
        """
        self.logger.info(f"Starting training for {episodes} episodes on {self.device}")
        
        # Training tracking
        start_time = time.time()
        last_update_time = start_time
        episode_durations = []
        
        # Initialize variables for tracking improvement
        best_avg_reward = float('-inf')
        no_improvement_count = 0
        patience = 50  # Episodes to wait before early stopping
        
        for episode in range(episodes):
            # Reset environment
            state = env.reset()
            total_reward = 0
            done = False
            episode_losses = []
            episode_q_values = []
            episode_start_time = time.time()
            
            # For episode analysis
            states = []
            actions = []
            rewards = []
            step_count = 0
            
            while not done:
                # Choose and execute action
                action = self.act(state)
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                self.remember(state, action, reward, next_state, done)
                
                # Store for analysis
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                step_count += 1
                
                # Update state
                state = next_state
                total_reward += reward
                
                # Learn from experiences
                if len(self.memory) >= self.batch_size:
                    loss, avg_q = self.replay()
                    if loss > 0:
                        episode_losses.append(loss)
                        episode_q_values.append(avg_q)
            
            # Update learning rate scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Track episode duration
            episode_duration = time.time() - episode_start_time
            episode_durations.append(episode_duration)
            
            # Track training progress
            self.training_history['episode_rewards'].append(total_reward)
            self.training_history['epsilon_values'].append(self.epsilon)
            self.training_history['learning_rates'].append(current_lr)
            
            # Track average loss and Q-values
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            avg_q_value = np.mean(episode_q_values) if episode_q_values else 0
            self.training_history['loss_values'].append(avg_loss)
            self.training_history['avg_q_values'].append(avg_q_value)
            
            # Periodically update target network
            if episode % self.update_target_every == 0:
                self.update_target_model()
                self.logger.info(f"Target network updated at episode {episode+1}")
            
            # Calculate rolling average of rewards for early stopping
            if episode >= 19:  # Start calculating after 20 episodes
                last_20_avg = np.mean(self.training_history['episode_rewards'][-20:])
                
                # Check for improvement
                if last_20_avg > best_avg_reward:
                    best_avg_reward = last_20_avg
                    no_improvement_count = 0
                    
                    # Save best model
                    if not os.path.exists('models'):
                        os.makedirs('models')
                    self.save('models/water_allocation_best_model.pt')
                    self.logger.info(f"New best model saved with average reward: {best_avg_reward:.2f}")
                else:
                    no_improvement_count += 1
            
            # Periodic logging
            current_time = time.time()
            if (episode + 1) % 10 == 0:
                elapsed = current_time - last_update_time
                avg_duration = np.mean(episode_durations[-10:])
                remaining_episodes = episodes - episode - 1
                eta_minutes = (remaining_episodes * avg_duration) / 60
                
                self.logger.info(
                    f"Episode {episode+1}/{episodes} " +
                    f"({episode+1/episodes*100:.1f}%) | " +
                    f"Reward: {total_reward:.2f} | " +
                    f"Avg(20): {np.mean(self.training_history['episode_rewards'][-20:]):.2f} | " +
                    f"Best: {best_avg_reward:.2f} | " +
                    f"Steps: {step_count} | " +
                    f"Epsilon: {self.epsilon:.4f} | " +
                    f"Loss: {avg_loss:.6f} | " +
                    f"Avg Q: {avg_q_value:.4f} | " +
                    f"LR: {current_lr:.6f} | " +
                    f"Mem: {len(self.memory)} | " +
                    f"Time: {avg_duration:.2f}s/ep | " +
                    f"ETA: {eta_minutes:.1f}m"
                )
                
                last_update_time = current_time
                
                # Create periodic checkpoint
                if (episode + 1) % 50 == 0:
                    checkpoint_path = f'models/checkpoint_ep{episode+1}.pt'
                    self.save(checkpoint_path)
            
            # Early stopping check
            if no_improvement_count >= patience:
                self.logger.info(f"Early stopping triggered after {episode+1} episodes (no improvement for {patience} episodes)")
                break
            
            # Analyze action distribution periodically
            if (episode + 1) % 25 == 0:
                action_counts = np.bincount(actions, minlength=self.action_dim)
                action_percents = action_counts / len(actions) * 100
                action_str = " | ".join([f"A{i}: {p:.1f}%" for i, p in enumerate(action_percents)])
                self.logger.info(f"Action distribution: {action_str}")
        
        # Final target model update
        self.update_target_model()
        
        # Report final training time
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        # Save final model
        if not os.path.exists('models'):
            os.makedirs('models')
            
        self.save('models/water_allocation_final_model.pt')
        self.logger.info("Final model saved to 'models/water_allocation_final_model.pt'")
        
        # Visualize training progress
        self._plot_training_progress()
        
        return self.training_history
    
    def evaluate(self, env, episodes=100):
        """
        Evaluate agent performance
        
        Parameters:
        -----------
        env : WaterAllocationEnvironment
            Environment to evaluate in
        episodes : int, default=100
            Number of evaluation episodes
            
        Returns:
        --------
        dict: Performance metrics
        """
        self.logger.info(f"Evaluating agent for {episodes} episodes...")
        
        rewards = []
        steps = []
        water_allocations = []
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_water = []
            done = False
            step = 0
            
            while not done:
                # Use greedy policy (no exploration)
                action = self.act(state, evaluate=True)
                next_state, reward, done, info = env.step(action)
                
                # Track metrics
                episode_reward += reward
                episode_water.append(env.action_space[action])
                step += 1
                
                # Update state
                state = next_state
            
            rewards.append(episode_reward)
            steps.append(step)
            water_allocations.append(episode_water)
            
            if (episode + 1) % 20 == 0:
                self.logger.info(f"Evaluated {episode+1}/{episodes} episodes...")
        
        # Calculate metrics
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        mean_steps = np.mean(steps)
        mean_water = np.mean([np.mean(w) for w in water_allocations])
        std_water = np.std([np.mean(w) for w in water_allocations])
        
        # Action distribution
        all_actions = [a for ep_actions in water_allocations for a in ep_actions]
        action_counts = {}
        for a in env.action_space:
            action_counts[a] = all_actions.count(a) / len(all_actions) * 100
        
        metrics = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_steps': mean_steps,
            'mean_water_allocation': mean_water,
            'std_water_allocation': std_water,
            'action_distribution': action_counts
        }
        
        self.logger.info(f"Evaluation results: Mean reward: {mean_reward:.2f}±{std_reward:.2f}, "
                         f"Mean water allocation: {mean_water:.2f}±{std_water:.2f}")
        
        # Log action distribution
        action_str = " | ".join([f"{a}: {p:.1f}%" for a, p in action_counts.items()])
        self.logger.info(f"Action distribution: {action_str}")
        
        return metrics
    
    def save(self, filepath):
        """
        Save model to file
        """
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load model from file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.target_net.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon']
            
        if 'training_steps' in checkpoint:
            self.training_steps = checkpoint['training_steps']
            
        self.logger.info(f"Model loaded from {filepath}")
    
    def _plot_training_progress(self, filename='water_allocation_training_progress.png'):
        """
        Visualize training rewards, epsilon decay, and loss
        """
        # Create visualizations directory if it doesn't exist
        if not os.path.exists('visualizations'):
            os.makedirs('visualizations')
            
        plt.figure(figsize=(20, 15))
        
        # Plot rewards
        plt.subplot(3, 2, 1)
        plt.plot(self.training_history['episode_rewards'])
        
        # Add moving average line
        if len(self.training_history['episode_rewards']) > 20:
            window_size = 20
            smoothed = np.convolve(
                self.training_history['episode_rewards'], 
                np.ones(window_size)/window_size, 
                mode='valid'
            )
            plt.plot(
                range(window_size-1, len(self.training_history['episode_rewards'])), 
                smoothed, 
                'r-', 
                linewidth=2,
                label=f'Moving Avg (window={window_size})'
            )
            plt.legend()
            
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(alpha=0.3)
        
        # Plot epsilon decay
        plt.subplot(3, 2, 2)
        plt.plot(self.training_history['epsilon_values'])
        plt.title('Exploration Rate Decay')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid(alpha=0.3)
        
        # Plot loss
        plt.subplot(3, 2, 3)
        plt.plot(self.training_history['loss_values'])
        plt.title('Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.grid(alpha=0.3)
        
        # Plot average Q-values
        plt.subplot(3, 2, 4)
        if 'avg_q_values' in self.training_history and self.training_history['avg_q_values']:
            plt.plot(self.training_history['avg_q_values'])
            plt.title('Average Q-Values')
            plt.xlabel('Episode')
            plt.ylabel('Avg Q-Value')
            plt.grid(alpha=0.3)
        
        # Plot learning rate
        plt.subplot(3, 2, 5)
        if 'learning_rates' in self.training_history and self.training_history['learning_rates']:
            plt.plot(self.training_history['learning_rates'])
            plt.title('Learning Rate')
            plt.xlabel('Episode')
            plt.ylabel('Learning Rate')
            plt.grid(alpha=0.3)
        
        # Plot reward distribution
        plt.subplot(3, 2, 6)
        plt.hist(self.training_history['episode_rewards'], bins=20, alpha=0.7)
        plt.title('Reward Distribution')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join('visualizations', filename))
        plt.close()