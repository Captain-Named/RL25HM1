from __future__ import annotations

from typing import Callable, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.distributions import Normal
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
print(f"Using {device} device")

import torch.nn.functional as F


class Config:
    """
    Configuration class for the DQN-based portfolio optimization model.
    Contains all hyperparameters for the financial environment, 
    reinforcement learning algorithm, and neural network training.
    """
    def __init__(self):
        # Utility function parameters
        self.utility_a: float = 1  # Risk aversion coefficient in exponential utility
        
        # Financial environment parameters
        self.T: int = 10  # Time horizon (number of trading periods)
        self.p: float = 0.749519253521917  # Probability of high return for risky asset
        self.r: float = 0.281093058305667  # Risk-free interest rate (annual)
        self.a: float = 0.479550138167822  # Risky asset return in favorable state
        self.b: float = -0.14637663947236  # Risky asset return in unfavorable state
        self.initial_wealth = 1  # Starting wealth, normalized to 1
        
        # DQN learning hyperparameters
        self.alpha: float = 1e-4  # Learning rate for Adam optimizer
        self.num_episodes_per_epoch: int = 20  # Number of episodes to collect per training epoch
        self.num_actions: int = 21  # Action space granularity (discretized allocation options)
        self.epsilon: float = 0.3  # Exploration rate for epsilon-greedy policy
        self.num_epoch: int = 15  # Total number of training epochs
        
        # Experience replay parameters
        self.batch_size: int = 4  # Number of experiences per mini-batch update
        # Calculate total batches per epoch based on episode length and count
        self.num_batch_per_epoch: int = int(self.T * self.num_episodes_per_epoch * 2 / self.batch_size)
        
        # Utility function definition: exponential utility with constant risk aversion
        # Represents investor's risk preferences (risk-averse for utility_a > 0)
        self.utility_func: Callable[[float], float] = lambda x: (-np.exp(-self.utility_a * x)) / self.utility_a


class ReplayBuffer:
    """
    Experience replay buffer for DQN algorithm implementation.
    
    This class provides storage and sampling functionality for transition experiences,
    which are used to train the neural network in a stable manner. By storing and randomly
    sampling past experiences, it breaks the correlation between consecutive training samples
    and improves learning stability.
    
    Attributes:
        replay_buffer (list[Experience]): List that stores experience tuples for training
    """
    def __init__(self):
        """Initialize an empty replay buffer."""
        self.replay_buffer: list[Experience] = []

    def _push(self, experiences: list[Experience]):
        """
        Add a list of experiences to the replay buffer.
        
        Args:
            experiences (list[Experience]): List of experience tuples to be added
        """
        self.replay_buffer.extend(experiences)

    def sample(self, batch_size: int) -> list[Experience]:
        """
        Sample a batch of experiences randomly from the replay buffer.
        
        Random sampling helps break the correlation between consecutive
        training samples, which stabilizes the learning process.
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            list[Experience]: A randomly sampled batch of experiences
        """
        return random.choices(self.replay_buffer, k=batch_size)
    
    def _reset(self):
        """Clear all experiences from the replay buffer."""
        self.replay_buffer = []
    
    def get_experiences_and_push(self, episodes):
        """
        Extract transition experiences from episodes and add them to the buffer.
        
        This method:
        1. Resets the current buffer
        2. Processes each episode to extract (state, action, reward, next_state) tuples
        3. Shuffles the experiences to break sequential correlations
        4. Adds all experiences to the buffer
        
        Args:
            episodes (list[dict]): List of episode dictionaries containing "states", 
                                  "actions", and "rewards" sequences
        """
        # Clear the current replay buffer
        self._reset()
        
        # Container for all experiences extracted from episodes
        experiences: list[Experience] = []
        
        # Process each episode to extract transition experiences
        for episode in episodes:
            # For each step in the episode (except the terminal state)
            for i in range(len(episode["states"]) - 1):
                # Create an Experience object with state, action, reward, and next_state
                experiences.append(
                    Experience(
                        episode["states"][i],            # Current state
                        episode["actions"][i],           # Action taken
                        episode["rewards"][i],           # Reward received
                        episode["states"][i + 1],        # Next state
                    )
                )
        
        # Shuffle experiences to break temporal correlation
        random.shuffle(experiences)
        
        # Add all experiences to the replay buffer
        self._push(experiences)


class RiskyReturnDistribution:
    """
    Models a binary distribution for risky asset returns in financial simulations.
    
    Attributes:
        a (float): The higher return value, typically positive, representing favorable market conditions
        b (float): The lower return value, typically negative, representing unfavorable market conditions
        p (float): Probability of obtaining the higher return 'a', value between 0 and 1
    """
    def __init__(self, a, b, p):
        """
        Initialize the binary return distribution with specified parameters.
        """
        self.a = a  # Higher return value (favorable market condition)
        self.b = b  # Lower return value (unfavorable market condition)
        self.p = p  # Probability of favorable outcome
    
    def sample(self) -> float:
        """
        Generate a random sample from the binary return distribution.
        
        Returns a single random sample based on the configured probability distribution:
        - Returns value 'a' with probability p
        - Returns value 'b' with probability (1-p)
        
        Returns:
            float: A random return value (either a or b)
        """
        return self.a if np.random.uniform() < self.p else self.b


class State:
    """
    Represents a state in the portfolio optimization MDP (Markov Decision Process).
    """
    def __init__(self, t, w):
        """
        Initialize a State object with time step and wealth value.
        
        Args:
            t (int): Current time step in the investment horizon
            w (float): Current wealth value of the investor's portfolio
        """
        self.t = t  # Current time step in the investment horizon
        self.w = w  # Current wealth value at this time step



class Q(nn.Module):
    """
    Neural network for approximating the action-value function (Q-function) in DQN.

    Architecture:
        - Input layer (3 neurons): Takes wealth, time step, and action as input
        - 4 hidden layers (64 neurons each): Process features with LeakyReLU activation
        - Output layer (1 neuron): Produces the Q-value estimate
    """
    def __init__(self):
        """
        Initialize the Q-network architecture with fully connected layers.
        """
        super(Q, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # Input layer: takes (wealth, time step, action)
        self.fc2 = nn.Linear(64, 64)  # First hidden layer
        self.fc3 = nn.Linear(64, 64)  # Second hidden layer
        self.fc4 = nn.Linear(64, 64)  # Third hidden layer
        self.fc5 = nn.Linear(64, 1)   # Output layer: produces Q(state, action)

    def forward(self, wealth_time):
        """
        Forward pass through the Q-network.
        
        Args:
            wealth_time (torch.Tensor): Input tensor containing [wealth, time, action]
                                       with shape (batch_size, 3)
                                       
        Returns:
            torch.Tensor: Predicted Q-values with shape (batch_size, 1)
        """
        # Apply layers with LeakyReLU activation
        x = F.leaky_relu(self.fc1(wealth_time))  # First layer activation
        x = F.leaky_relu(self.fc2(x))            # Second layer activation
        x = F.leaky_relu(self.fc3(x))            # Third layer activation
        x = F.leaky_relu(self.fc4(x))            # Fourth layer activation
        
        # Final layer (no activation function)
        return self.fc5(x)  # Output Q-values
    
                


class AssetAllocDiscreteMDP:
    """
    Asset allocation Markov Decision Process (MDP) with discrete actions.
    
    This class implements the financial environment for portfolio optimization,
    modeling the sequential decision-making problem of allocating wealth between
    risky and risk-free assets. The environment evolves stochastically based on
    the agent's allocation decisions and random asset returns.
    
    Attributes:
        risky_return_distribution (RiskyReturnDistribution): Distribution model for risky asset returns
        riskless_returns (float): Fixed return rate for the risk-free asset
        state (State): Current state of the MDP (time step and wealth)
        T (int): Time horizon (total number of investment periods)
        num_actions (int): Number of discrete allocation actions available
        actions (list): Container for storing actions taken in an episode
        rewards (list): Container for storing rewards received in an episode
        states (list): Container for storing states visited in an episode
        qnn (Q): Q-network used for action selection
        utility_func (callable): Utility function for evaluating terminal wealth
        config (Config): Configuration parameters for the environment
    """
    def __init__(self, risky_return_distribution, initial_state, qnn, config: Config):
        """
        Initialize the asset allocation MDP environment.
        
        Args:
            risky_return_distribution (RiskyReturnDistribution): Model for generating risky asset returns
            initial_state (State): Starting state for the MDP
            qnn (Q): Q-network for action selection
            config (Config): Configuration parameters for the environment
        """
        self.risky_return_distribution: RiskyReturnDistribution = risky_return_distribution
        self.riskless_returns: float = config.r
        self.state: State = initial_state   # Current state of the environment
        self.T: int = config.T              # Time horizon
        self.num_actions: int = config.num_actions  # Number of discrete allocation choices
        
        # Containers for tracking episode trajectory
        self.actions = []  # Actions taken during the episode (T actions)
        self.rewards = []  # Rewards received during the episode (T rewards)
        self.states = []   # States visited during the episode (T+1 states including initial)
        
        self.qnn: Q = qnn  # Q-network for action selection
        self.utility_func = config.utility_func  # Utility function for reward calculation
        self.config = config  # Configuration parameters

    def step(self, state: State, action) -> Tuple[State, float]:
        """
        Execute a single step in the environment given a state and action.
        
        Args:
            state (State): Current state containing time and wealth
            action (int): Action index representing portfolio allocation decision
            
        Returns:
            Tuple[State, float]: Next state and reward tuple
        """
        # Convert action index to proportion of wealth allocated to risky asset
        risky_alloc = (action / (self.num_actions - 1)) * state.w
        
        # Calculate next wealth based on portfolio performance
        next_wealth: float = (
            risky_alloc * (1 + self.risky_return_distribution.sample()) +  # Risky asset component
            (state.w - risky_alloc) * (1 + self.riskless_returns)          # Risk-free asset component
        )
        
        # Terminal reward based on utility function, zero for non-terminal states
        reward: float = self.utility_func(next_wealth) if state.t == self.T - 1 else 0.0
        
        # Alternative reward formulations (commented out)
        # if state.t == self.T - 1:
        #     reward = 2 * self.utility_func(next_wealth) - self.utility_func(state.w)
        # else:
        #     reward = self.utility_func(next_wealth) - self.utility_func(state.w)
        
        # Create next state with incremented time step
        next_state: State = State(state.t + 1, next_wealth)
        
        return (next_state, reward)

    def get_episode(self, greedy: bool = False):
        """
        Generate a complete episode by simulating the investment process from start to finish.

        This method simulates the sequential decision-making process for an entire investment 
        horizon (T time steps). At each time step, it selects an action (portfolio allocation)
        using either an epsilon-greedy policy or a purely greedy policy, then executes that
        action and records the resulting state, action, and reward.

        Args:
            greedy (bool, optional): If True, always select the action with highest expected value.
                                    If False, use epsilon-greedy policy for exploration. 
                                    Defaults to False.

        Returns:
            dict: A dictionary containing the complete trajectory of the episode:
                - "states": List of T+1 states (including initial and terminal states)
                - "actions": List of T actions taken
                - "rewards": List of T rewards received
        """
        # Record initial state
        self.states.append(self.state)

        # Generate episode by simulating T time steps
        for t in range(self.T):
            action: int  # Action index to be selected

            # Action selection using epsilon-greedy or pure greedy policy
            if (not greedy) and (random.random() < self.config.epsilon):
                # Exploration: choose random action
                action = random.randint(0, self.config.num_actions)
            else:
                # Exploitation: choose best action according to Q-network
                with torch.no_grad():  # Disable gradient computation for inference
                    # Create tensor with all possible actions
                    all_actions = torch.arange(self.num_actions, dtype=torch.float32).to(device)  # Shape: (num_actions,)

                    # Create tensor with current state (wealth and time)
                    state_tensor = torch.tensor([self.state.w, self.state.t], dtype=torch.float32).to(device)  # Shape: (2,)

                    # Expand state tensor to match number of actions for batch processing
                    state_tensor = state_tensor.unsqueeze(0).expand(self.num_actions, -1)  # Shape: (num_actions, 2)

                    # Concatenate state and action tensors to form network inputs
                    input_tensor = torch.cat([state_tensor, all_actions.unsqueeze(1)], dim=1)  # Shape: (num_actions, 3)

                    # Calculate Q-values for all actions simultaneously
                    q_values = self.qnn(input_tensor).detach().cpu().numpy()  # Shape: (num_actions, 1)

                    # Flatten result to 1D array for easier processing
                    q4actions = q_values.flatten()

                    # Select action with highest Q-value
                    action = np.argmax(q4actions)

            # Execute selected action and observe next state and reward
            self.state, reward = self.step(self.state, action)

            # Record trajectory information
            self.states.append(self.state)
            self.actions.append(action)
            self.rewards.append(reward)

        # Return complete episode trajectory
        return {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards
        }

class Experience:
    """
    Container class that stores a single transition experience for reinforcement learning.
    """
    def __init__(self, state, action, reward, next_state):
        """
        Initialize an Experience object with the components of a transition.
        
        Args:
            state (State): Current state containing time step and wealth value
            action (int): Action index representing the portfolio allocation decision
            reward (float): Immediate reward received for this transition (typically non-zero only for terminal states)
            next_state (State): Next state resulting from taking the action in the current state
        """
        self.state: State = state          # Starting state of the transition
        self.action: int = action        # Action taken at the state
        self.reward: float = reward        # Reward received for the transition
        self.next_state: State = next_state  # Resulting state after taking the action


class DQNAgent:
    """
    Deep Q-Network (DQN) agent that learns optimal asset allocation strategies.
    
    This class implements the DQN algorithm for portfolio optimization, managing the
    neural network training process, experience collection, and evaluation of learned
    policies. It handles the core reinforcement learning loop of collecting experiences,
    updating the Q-network, and evaluating performance.
    
    Attributes:
        config (Config): Configuration parameters for the learning process
        qnn (Q): Neural network that approximates the Q-function
        replay_buffer (ReplayBuffer): Storage for transition experiences
        optimizer (torch.optim.Adam): Optimizer for neural network training
        loss_fn (torch.nn.MSELoss): Loss function for Q-value regression
        loss_history (list): History of training losses for monitoring convergence
        final_wealth (list): Record of terminal wealth values for performance evaluation
    """
    def __init__(self, qnn: Q, config: Config):
        """
        Initialize the DQN agent with neural network and configuration parameters.
        
        Args:
            qnn (Q): Neural network for Q-function approximation
            config (Config): Configuration object with environment and training parameters
        """
        self.config = config  # Configuration parameters
        self.qnn = qnn  # Q-network for action-value function approximation
        self.replay_buffer: ReplayBuffer = ReplayBuffer()  # Experience storage for training
        
        # Optimizer with weight decay for regularization to prevent overfitting
        self.optimizer = optim.Adam(qnn.parameters(), lr=config.alpha, weight_decay=1e-6)
        
        # Mean squared error loss for Q-value prediction
        self.loss_fn = nn.MSELoss()
        
        # Metrics tracking for performance monitoring
        self.loss_history = []  # Record of training losses
        self.final_wealth = []  # Record of terminal wealth values achieved

    def _get_episodes_from_mdp(self, num_episodes, greedy: bool = False):
        """
        Generate multiple episode trajectories by simulating the MDP.
        
        This method creates multiple independent MDP instances and collects
        complete episode trajectories for each. These trajectories can be used
        for training (with exploration) or evaluation (with greedy policy).
        
        Args:
            num_episodes (int): Number of episodes to generate
            greedy (bool, optional): Whether to use greedy action selection (True)
                                    or allow exploration (False). Defaults to False.
                                    
        Returns:
            list: List of episode dictionaries, each containing states, actions, and rewards
        """
        episodes = []  # Container for collected episodes
        
        # Generate the specified number of episodes
        for _ in range(num_episodes):
            # Create a new MDP instance with fresh random seed for independence
            mdp = AssetAllocDiscreteMDP(
                # Initialize with the binary distribution for risky asset returns
                risky_return_distribution=RiskyReturnDistribution(self.config.a, self.config.b, self.config.p),
                # Start from initial state with time=0 and specified initial wealth
                initial_state=State(0, self.config.initial_wealth),
                # Use the current Q-network for action selection
                qnn=qnn,
                # Pass configuration parameters
                config=self.config
            )
            
            # Generate and store a complete episode trajectory
            episodes.append(mdp.get_episode(greedy))
            
            # Explicitly delete MDP instance to free memory
            del mdp
            
        return episodes

    def train(self):
        """
        Train the DQN agent for one complete epoch.

        This method implements a full training cycle of the Deep Q-Network algorithm:
        1. Collects episodes using the current policy
        2. Extracts experiences from episodes into the replay buffer
        3. Performs multiple mini-batch updates on the Q-network using random samples
        4. Updates the exploration rate to gradually shift from exploration to exploitation

        The training follows standard DQN principles including:
        - Experience replay to break correlation between consecutive samples
        - Target Q-values computed using the Bellman equation
        - Batch gradient updates to optimize the neural network

        Returns:
            None: Updates are applied directly to the Q-network and metrics are stored internally
        """
        # Generate episodes using the current policy
        episodes = self._get_episodes_from_mdp(greedy=True, num_episodes=self.config.num_episodes_per_epoch)

        # Process episodes to extract experiences and update replay buffer
        self.replay_buffer.get_experiences_and_push(episodes)

        # Perform multiple mini-batch updates
        for i in range(self.config.num_batch_per_epoch):
            # Sample a random batch of experiences from the replay buffer
            batch_experiences = self.replay_buffer.sample(self.config.batch_size)

            # Prepare input features and target Q-values for neural network training
            input_batch = []  # Will contain [wealth, time, action] for each experience
            label_batch = []  # Will contain target Q-values for each experience

            # Process each experience to compute its target Q-value
            for experience in batch_experiences:
                # Initialize next_q value (for Bellman equation)
                next_q: float = 0

                # Terminal state handling: use utility of final wealth as target
                if experience.state.t == self.config.T - 1:
                    next_q = self.config.utility_func(experience.next_state.w)
                else:
                    # Non-terminal state: use max Q-value of next state as target (Q-learning)
                    with torch.no_grad():  # No gradient computation for target network
                        # Create tensors for all possible actions in the next state
                        actions = torch.arange(self.config.num_actions, dtype=torch.float32).to(device)  # Shape: (num_actions,)

                        # Create tensor for the next state (wealth and time)
                        state_tensor = torch.tensor([experience.next_state.w, experience.next_state.t], dtype=torch.float32).to(device)  # Shape: (2,)

                        # Expand state tensor to match number of actions for batch processing
                        state_tensor = state_tensor.unsqueeze(0).expand(self.config.num_actions, -1)  # Shape: (num_actions, 2)

                        # Concatenate state and action tensors to form network inputs
                        input_tensor = torch.cat([state_tensor, actions.unsqueeze(1)], dim=1)  # Shape: (num_actions, 3)

                        # Calculate Q-values for all next state actions simultaneously
                        q_values = self.qnn(input_tensor).detach().cpu().numpy()  # Shape: (num_actions, 1)

                        # Flatten results and find maximum Q-value (standard Q-learning update)
                        q4actions = q_values.flatten()
                        next_q = np.max(q4actions)  # Maximum Q-value for next state

                # Compute TD target using Bellman equation: reward + max_a' Q(s',a')
                td_target = experience.reward + next_q

                # Add state-action pair and target to training batches
                input_batch.append([experience.state.w, experience.state.t, experience.action])
                label_batch.append(td_target)

            # Perform neural network update using mean squared error loss
            loss = self.loss_fn(
                # Forward pass: predict Q-values for the input batch
                self.qnn(torch.tensor(input_batch, dtype=torch.float32).to(device)),
                # Target Q-values reshaped to match network output
                torch.tensor(label_batch, dtype=torch.float32).unsqueeze(1).to(device)
            )

            # Standard gradient descent update
            self.optimizer.zero_grad()  # Clear previous gradients
            loss.backward()             # Compute gradients
            self.optimizer.step()       # Apply gradient update

            # Record loss for monitoring convergence
            self.loss_history.append(loss.item())

            # Evaluate current policy performance
            self.get_final_wealth_evaluation()

        # Decay exploration rate to gradually shift from exploration to exploitation
        self.config.epsilon *= 0.92  # Multiplicative decay factor
    
    def get_final_wealth_evaluation(self):
        """
        Evaluate the agent's current policy by measuring terminal wealth.

        Process:
        1. Generate an episode using purely greedy action selection
        2. Extract the terminal wealth from the final state
        3. Add this value to the historical performance record

        Returns:
            None: Results are stored in the agent's final_wealth attribute
        """
        # Generate a complete episode using the current policy with greedy action selection
        episodes = self._get_episodes_from_mdp(greedy=True, num_episodes=1)

        # Extract final wealth values from the terminal states of each episode
        final_wealths = [episode["states"][-1].w for episode in episodes]

        # Add the results to the historical record for performance tracking
        self.final_wealth.extend(final_wealths)


    def plot_results(self):
        """
        Visualize training metrics with refined styling and structure
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Configure plot style
        plt.figure(figsize=(13, 5.5))
        plt.rcParams.update({'font.size': 12})

        # Common configuration
        plot_config = {
            'q_diff': {
                'data': self.loss_history,
                'title': 'Batch Loss Convergence Pattern',
                'colors': ('skyblue', 'navy'),
                'ylabel': 'Cumulative Q-Value Change'
            },
            'wealth': {
                'data': self.final_wealth,
                'title': 'Portfolio Value Development',
                'colors': ('lightgreen', 'darkgreen'),
                'ylabel': 'Terminal Wealth'
            }
        }

        # Plotting function
        def create_subplot(pos, cfg):
            plt.subplot(1, 2, pos)
            plt.plot(cfg['data'], label='Raw Values', 
                    color=cfg['colors'][0], alpha=0.4)

            # Smoothing logic
            window_size = 5
            if len(cfg['data']) >= window_size:
                ma = np.convolve(cfg['data'], np.ones(window_size)/window_size, mode='valid')
                x_vals = range(window_size-1, len(cfg['data']))
                plt.plot(x_vals, ma, label=f'{window_size}-episode MA', 
                        color=cfg['colors'][1], linewidth=1.8)

            plt.grid(True, alpha=0.3)
            plt.title(cfg['title'], pad=15)
            plt.xlabel('Training Episodes', labelpad=10)
            plt.ylabel(cfg['ylabel'], labelpad=10)
            plt.legend()

        # Generate plots
        create_subplot(1, plot_config['q_diff'])
        create_subplot(2, plot_config['wealth'])

        plt.tight_layout(pad=3.0)
        plt.show()




# ====================main====================

# Initialize the Q-network and place it on the appropriate device (CPU or GPU)
qnn = Q().to(device)

# Create configuration with default hyperparameters
config = Config()

# Initialize DQN agent with the Q-network and configuration
agent = DQNAgent(qnn, config)

# Generate initial episodes to observe pre-training policy behavior
# print("Initial policy behavior before training:")
# episodes = agent._get_episodes_from_mdp(10, greedy=True)
# for episode in episodes:
#     print(episode["actions"])

# Main training loop
print("\nStarting training process...")
for epoch in tqdm(range(config.num_epoch)):
    # Train the agent for one complete epoch
    agent.train()
    
    # Periodically check policy behavior during training
    # print(f"\nPolicy behavior after epoch {epoch+1}:")
    # episodes = agent._get_episodes_from_mdp(10, greedy=True)
    # for episode in episodes:
    #     print(episode["actions"])  # Display updated allocation decisions

# Visualize training results with plots of loss and wealth metrics
print("\nGenerating performance visualization...")
agent.plot_results()

# Evaluate final trained policy behavior
print("\nFinal policy behavior after complete training:")
episodes = agent._get_episodes_from_mdp(10, greedy=True)
for episode in episodes:
    print(episode["actions"])  # Display final allocation strategy