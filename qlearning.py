from __future__ import annotations

from typing import Callable, Tuple
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class State:
    """
    Represents a state in an asset allocation problem with discrete time and wealth levels.
    
    This class is used in a Markov Decision Process (MDP) to represent the agent's state
    at a specific point in time. Each state is defined by a time step and a discrete
    wealth level index.
    
    Attributes:
        t (int): Current time step, ranging from 0 to T-1, where T is the total time horizon
        w (int): Current discretized wealth level index, representing the relative level
                 of wealth compared to a maximum wealth value
    """
    
    def __init__(self, t, w):
        """
        Initialize a State object.
        
        Args:
            t (int): Current time step
            w (int): Current discretized wealth level index
        """
        self.t: int = t  # state's current time
        self.w: int = w  # wealth level @ time t

class RiskyReturnDistribution:
    """
    Represents a binary risky asset return distribution for financial modeling.
    
    This class models a simple binary distribution where the risky asset return
    can take one of two possible values (a or b) with probability p for value a
    and (1-p) for value b. This is commonly used in asset allocation problems
    to simulate risky investment outcomes.
    
    Attributes:
        a (float): The first possible return value, typically the higher return
        b (float): The second possible return value, typically the lower return
        p (float): The probability of return a occurring, between 0 and 1
    """
    
    def __init__(self, a, b, p):
        """
        Initialize a binary return distribution.
        
        Args:
            a (float): The first possible return value
            b (float): The second possible return value
            p (float): The probability of return a occurring (should be between 0 and 1)
        """
        self.a = a
        self.b = b
        self.p = p

    def sample(self):
        """
        Generate a random sample from the distribution.
        
        Returns a random value based on the distribution parameters:
        - Returns value 'a' with probability p
        - Returns value 'b' with probability (1-p)
        
        Returns:
            float: A random sample (either a or b) from the distribution
        """
        return self.a if np.random.uniform() < self.p else self.b

class Config:
    """
    Configuration parameters for the asset allocation reinforcement learning model.
    
    This class encapsulates all parameters needed to define the environment, agent behavior,
    and learning process for an asset allocation problem solved using Q-learning.
    
    Attributes:
        utility_a (float): Risk aversion parameter for the exponential utility function.
        T (int): Time horizon for the investment problem, representing the number of time steps.
        p (float): Probability of the risky asset returning value 'a' (higher return).
        r (float): Risk-free rate of return.
        a (float): Higher possible return from the risky asset.
        b (float): Lower possible return from the risky asset.
        alpha (float): Learning rate for Q-learning updates.
        num_actions (int): Number of discretization points for the action space (allocation choices).
        num_wealth (int): Number of discretization points for the wealth space.
        initial_wealth (float): Starting wealth value for each episode.
        max_wealth (float): Maximum possible wealth used for state space discretization.
        epsilon (float): Exploration parameter for epsilon-greedy action selection.
        num_episodes (int): Total number of episodes for training the Q-learning agent.
        utility_func (callable): Exponential utility function that measures risk-adjusted rewards.
    """
    
    def __init__(self):
        """
        Initialize the configuration with default parameters for the asset allocation problem.
        
        The defaults represent a moderately risk-averse investor in a market with positive
        expected returns from both risk-free and risky assets, with the risky asset offering
        higher expected returns but with uncertainty.
        """
        self.utility_a: float = 1                 # Risk aversion parameter
        self.T: int = 10                          # End time horizon
        self.p: float = 0.749519253521917  # Probability of high return for risky asset
        self.r: float = 0.281093058305667  # Risk-free interest rate (annual)
        self.a: float = 0.479550138167822  # Risky asset return in favorable state
        self.b: float = -0.14637663947236  # Risky asset return in unfavorable state
        self.alpha: float = 0.01                  # Learning rate
        self.num_actions: int = 20                # Number of actions discretized from continuous action space
        self.num_wealth: int = 500                 # Number of wealth levels
        self.initial_wealth: float = 1            # Initial wealth level
        higher_return: float = max(self.a, self.r)  # Higher return between a and r
        self.max_wealth: float = self.initial_wealth * ((1+higher_return)**10)  # Maximum possible wealth
        self.epsilon: float = 0.4                 # Exploration rate for epsilon-greedy policy
        self.num_episodes: int = 500000           # Number of episodes for training
        
        # Exponential utility function capturing risk aversion
        # Negative exponential utility function: -exp(-a*x)/a
        # where 'a' controls risk aversion level (higher a = more risk-averse)
        self.utility_func = lambda x: -np.exp(-self.utility_a * x) / self.utility_a

class QTable:
    """
    A three-dimensional Q-value table for reinforcement learning in asset allocation.
    
    This class implements a state-action value function (Q-function) represented as a 3D array,
    indexed by time step, wealth level, and action. It supports storing, retrieving, and
    updating Q-values for the Q-learning algorithm.
    
    Attributes:
        num_wealth (int): Number of discretized wealth levels
        num_time (int): Number of time steps in the investment horizon
        num_actions (int): Number of possible actions (allocation choices)
        q (numpy.ndarray): 3D array storing Q-values, with dimensions (time, wealth, action)
    """
    
    def __init__(self, num_wealth, num_time, num_actions):
        """
        Initialize the Q-table with slightly pessimistic values.
        
        Creates a 3D array with dimensions representing time steps, wealth levels, and
        actions. The Q-values are initialized with small negative random values to
        encourage exploration in the early stages of learning.
        
        Args:
            num_wealth (int): Number of discretized wealth levels
            num_time (int): Number of time steps in the investment horizon
            num_actions (int): Number of possible actions (allocation choices)
        """
        self.num_wealth = num_wealth
        self.num_time = num_time
        self.num_actions = num_actions
        # Initialize with slightly pessimistic values to encourage exploration
        self.q = np.random.uniform(-0.1, 0, (num_time, num_wealth, num_actions))
        # Alternative initialization methods (commented out):
        # self.q = np.zeros((num_time, num_wealth, num_actions))
        # self.q = np.full((num_time, num_wealth, num_actions),
    def update(self, state: State, action: int, qvalue: float):
        """
        Update the Q-value for a specific state-action pair.
        
        This method assigns a new Q-value to the specified state-action pair in the Q-table.
        It uses the state's time step and wealth level along with the action index as 
        indices into the 3D array.
        
        Args:
            state (State): The state object containing time step (t) and wealth level (w)
            action (int): The action index to update, ranging from 0 to num_actions-1
            qvalue (float): The new Q-value to store
        """
        self.q[state.t, state.w, action] = qvalue
    
    def get(self, state: State, action: int):
        """
        Retrieve the Q-value for a specific state-action pair.
        
        This method accesses the Q-table to retrieve the current value estimate for 
        the specified state-action pair. It uses the state's time step and wealth level
        along with the action index as indices into the 3D array.
        
        Args:
            state (State): The state object containing time step (t) and wealth level (w)
            action (int): The action index to query, ranging from 0 to num_actions-1
            
        Returns:
            float: The current Q-value for the specified state-action pair
        """
        return self.q[state.t, state.w, action]
    
    
class AssetAllocDiscreteMDP:
    """
    A discrete Markov Decision Process (MDP) for asset allocation problems.
    
    This class simulates a financial environment where an agent makes sequential
    asset allocation decisions between risky and risk-free investments. It implements
    the dynamics of wealth evolution based on investment allocation and random returns,
    and provides methods for generating episodes by performing actions under supervision
    of current policy represented by q-network.
    
    Attributes:
        risky_return_distribution (RiskyReturnDistribution): Distribution of risky asset returns
        riskless_returns (float): Constant risk-free rate of return
        num_actions (int): Number of discrete allocation choices available
        num_wealth (int): Number of discrete wealth levels in the state space
        max_wealth (float): Maximum possible wealth value for normalization
        state (State): Current state of the MDP (time step and wealth level)
        T (int): Time horizon (number of decision periods)
        epsilon (float): Exploration parameter for epsilon-greedy action selection
        actions (list): Storage for actions taken in an episode
        rewards (list): Storage for rewards received in an episode
        states (list): Storage for states visited in an episode
        qtable (QTable): Q-value table for determining optimal actions
        utility_func (callable): Utility function for terminal wealth evaluation
    """
    
    def __init__(self, config: Config, qtable: QTable):
        """
        Initialize the asset allocation MDP environment.
        
        Sets up the environment parameters based on the provided configuration
        and prepares containers to track the trajectory of an episode (states,
        actions, and rewards).
        
        Args:
            config (Config): Configuration object containing environment parameters
            qtable (QTable): Q-value table for action selection
        """
        self.risky_return_distribution: RiskyReturnDistribution = RiskyReturnDistribution(config.a, config.b, config.p)
        self.riskless_returns: float = config.r
        self.num_actions: int = config.num_actions
        self.num_wealth: int = config.num_wealth    # meaning a wealth level is in [0, num_wealth-1]
        self.max_wealth: float = config.max_wealth
        initial_wealth_level: int = min(config.num_wealth-1, round(config.initial_wealth / config.max_wealth * (config.num_wealth - 1)))
        self.state: State = State(0, initial_wealth_level)
        self.T: int = config.T
        self.epsilon: float = config.epsilon
        self.actions = []  # T actions
        self.rewards = []  # T rewards
        self.states = []  # T+1 states
        self.qtable = qtable
        self.utility_func = config.utility_func

    def step(self, state: State, action) -> Tuple[State, float]:
        """
        Execute one step in the MDP by taking an action from the current state.

        This method implements the core dynamics of the asset allocation problem:
        1. Converts the discrete state and action to continuous wealth and allocation values
        2. Determines the next wealth level based on the portfolio performance
        3. Assigns a reward based on terminal utility (at the final time step only)
        4. Creates and returns the next state and corresponding reward

        Args:
            state (State): Current state containing time step and wealth level
            action (int): Action index representing the allocation choice

        Returns:
            Tuple[State, float]: A tuple containing:
                - next_state: The resulting state after taking the action
                - reward: The reward received (non-zero only at terminal states)
        """
        # Convert discrete wealth level to continuous wealth value
        current_wealth: float = state.w / (self.num_wealth-1) * self.max_wealth

        # Convert discrete action to continuous allocation amount
        risky_alloc: float = action / (self.num_actions - 1) * current_wealth

        # Calculate next wealth based on portfolio performance
        # (risky asset portion × risky return) + (risk-free portion × risk-free return)
        next_wealth: float = min(self.max_wealth, 
                                risky_alloc * (1 + self.risky_return_distribution.sample()) + 
                                (current_wealth - risky_alloc) * (1 + self.riskless_returns))

        # Convert continuous wealth back to discrete wealth level
        next_wealth_level: int = round(next_wealth / self.max_wealth * (self.num_wealth - 1))

        # Terminal reward based on utility of final wealth, zero otherwise
        reward: float = self.utility_func(next_wealth) if state.t == self.T - 1 else 0.0

        # Create next state with incremented time step
        next_state: State = State(state.t + 1, next_wealth_level)

        return (next_state, reward)

    def get_episode(self, greedy: bool = False):
        """
        Generate a complete episode trajectory by simulating the agent's actions in the environment.

        This method runs the agent through a complete investment horizon (T time steps),
        selecting actions either greedily (using the best known Q-values) or with
        exploration (using epsilon-greedy policy). It tracks and stores the entire
        trajectory of states, actions, and rewards.

        The episode begins from the current state of the MDP and proceeds until reaching
        the terminal time step. At each step:
        1. An action is selected (either randomly or based on Q-values)
        2. The environment transitions to a new state based on the action
        3. The trajectory information is recorded

        Args:
            greedy (bool, optional): Whether to use pure greedy action selection (True)
                                    or (1-epsilon)-greedy exploration (False). Defaults to False.

        Returns:
            dict: A dictionary containing the complete episode trajectory with keys:
                - 'states': List of T+1 states visited (including initial and terminal states)
                - 'actions': List of T actions taken
                - 'rewards': List of T rewards received (typically only the last is non-zero)
        """
        # Record initial state
        self.states.append(self.state)

        # Simulate for T time steps
        for t in range(self.T):
            # Select action using epsilon-greedy or pure greedy policy
            action_index: int
            if (not greedy) and (random.random() < self.epsilon):
                # Exploration: choose random action
                action_index = random.randint(0, self.num_actions - 1)
            else:
                # Exploitation: choose best action according to Q-table
                action_index = np.argmax([self.qtable.get(self.state, action) 
                                        for action in range(self.num_actions)])

            # Execute selected action and observe next state and reward
            self.state, reward = self.step(self.state, action_index)

            # Record trajectory information
            self.states.append(self.state)
            self.actions.append(action_index)
            self.rewards.append(reward)

        # Return the complete episode trajectory
        return {
            "states": self.states,
            "actions": self.actions,
            "rewards": self.rewards
        }


class QAgent:
    """
    Q-learning agent that learns optimal asset allocation policies through interaction with the environment.
    
    This class implements the reinforcement learning agent that iteratively improves its allocation
    strategy by updating Q-values based on experience. It handles the training process, policy 
    evaluation, and visualization of learning metrics.
    
    Attributes:
        qtable (QTable): The Q-value table that represents the agent's knowledge of state-action values
        config (Config): Configuration parameters for the learning process
        num_episodes (int): Total number of training episodes to run
        T (int): Time horizon for each investment episode
        num_actions (int): Number of discrete allocation choices available
        alpha (float): Learning rate for updating Q-values
        qdelta (list): History of Q-value changes between episodes, used to monitor convergence
        final_wealth (list): History of terminal wealth values achieved during training
    """
    
    def __init__(self, q: QTable, config: Config):
        """
        Initialize the Q-learning agent with a Q-table and configuration.
        
        Args:
            q (QTable): Pre-initialized Q-table for storing and updating state-action values
            config (Config): Configuration object with learning parameters
        """
        self.qtable = q
        self.config = config
        self.num_episodes = config.num_episodes
        self.T = config.T
        self.num_actions = config.num_actions
        self.alpha = config.alpha   # learning rate
        self.qdelta = []  # q-value difference between two consecutive episodes
        self.final_wealth = []  # final wealth of each episode
    
    def _get_episode_from_mdp(self, greedy: bool = False):
        """
        Generate an episode by simulating the agent's interaction with the environment.
        
        Creates a new MDP instance for each episode to ensure independent sampling of
        investment returns, then runs a complete episode trajectory using the current
        Q-values to guide action selection.
        
        Args:
            greedy (bool, optional): Whether to use purely greedy action selection (True)
                                    or allow exploration (False). Defaults to False.
                                    
        Returns:
            dict: Episode trajectory containing states, actions, and rewards
        """
        mdp = AssetAllocDiscreteMDP(config=self.config, qtable=self.qtable)
        episode = mdp.get_episode(greedy)
        del mdp  # Explicitly free the MDP instance
        return episode

    def train(self):
        """
        Train the Q-learning agent over multiple episodes to learn the optimal asset allocation policy.

        This method implements the core Q-learning algorithm, which iteratively improves the 
        agent's policy by:
        1. Generating episodes using the current policy with exploration
        2. Computing TD-errors between predicted and target Q-values
        3. Updating Q-values using gradient descent with the learning rate alpha
        4. Tracking convergence metrics and final performance

        The learning process incorporates:
        - TD (Temporal Difference) learning with single-step backups
        - A gradually decaying learning rate to ensure convergence
        - Periodic policy visualization to monitor learning progress
        - Tracking of Q-value changes to assess convergence

        Returns:
            None: Results are stored in the agent's qdelta and final_wealth attributes
        """
        for j in tqdm(range(self.num_episodes)):
            # Store pre-update Q-values to measure changes
            qbefore = self.qtable.q.copy()

            # Generate a new episode using the current policy
            episode = self._get_episode_from_mdp()
            td_errors_this_episode = []

            # Process each transition in the episode for Q-learning updates
            for i in range(self.T):
                # Extract the i-th transition components
                state: State = episode["states"][i]
                reward = episode["rewards"][i]
                action = episode["actions"][i]
                next_state: State = episode["states"][i+1]

                # Compute the maximum Q-value for the next state (0 if terminal)
                if i == self.T-1:
                    max_next_q = 0  # Terminal state has no future value
                else:
                    max_next_q = max([self.qtable.get(next_state, action) for action in range(self.num_actions)])

                # Calculate TD target and TD error
                td_target = reward + max_next_q  # Bellman equation
                td_error = self.qtable.get(state, action) - td_target
                td_errors_this_episode.append(td_error)

                # Update Q-value using TD error and learning rate
                new_q = self.qtable.get(state, action) - self.alpha * td_error
                self.qtable.update(state, action, new_q)

            # Gradually decay learning rate for better convergence
            self.config.alpha *= 0.999999

            # Periodically display the current greedy policy
            # if j % 10000 == 0:
            #     self.show_policy()

            # Store post-update Q-values and calculate total change
            qafter = self.qtable.q.copy()
            self.qdelta.append(np.sum(np.abs(qbefore - qafter)))

            # Track terminal wealth achieved in this episode
            self.final_wealth.append(episode["states"][-1].w)
    
    def show_policy(self):
        """
        Display sample allocation policies from the current Q-function.
        """
        pass
        # for i in range(5):
        #     episode = self._get_episode_from_mdp(greedy=True)
        #     print(episode["actions"])

    def plot_results(self):
        """
        Visualize training metrics with refined styling and structure.

        This method creates a comprehensive visualization of the agent's learning progress
        with two key metrics:
        1. Q-value convergence - showing how rapidly the agent's value estimates are changing
        2. Terminal wealth development - showing how the agent's performance improves over time

        The plots include both raw data and a moving average to filter noise and highlight trends.
        Advanced styling is applied to improve readability and visual appeal.

        Returns:
            None: Displays the plots in the current figure window
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # Configure plot style
        plt.figure(figsize=(13, 5.5))
        plt.rcParams.update({'font.size': 12})

        # Common configuration for both subplots
        plot_config = {
            'q_diff': {
                'data': self.qdelta,
                'title': 'Q-Value Convergence Pattern',
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

        # Helper function to create individual subplots with consistent styling
        def create_subplot(pos, cfg):
            """
            Create a subplot with the specified configuration.

            Args:
                pos (int): Position of the subplot (1 or 2)
                cfg (dict): Configuration dictionary with data and styling parameters
            """
            plt.subplot(1, 2, pos)
            # Plot raw data with transparency
            plt.plot(cfg['data'], label='Raw Values', 
                    color=cfg['colors'][0], alpha=0.4)

            # Apply moving average smoothing when sufficient data is available
            window_size = 100
            if len(cfg['data']) >= window_size:
                ma = np.convolve(cfg['data'], np.ones(window_size)/window_size, mode='valid')
                x_vals = range(window_size-1, len(cfg['data']))
                plt.plot(x_vals, ma, label=f'{window_size}-episode MA', 
                        color=cfg['colors'][1], linewidth=1.8)

            # Add grid and labels for readability
            plt.grid(True, alpha=0.3)
            plt.title(cfg['title'], pad=15)
            plt.xlabel('Training Episodes', labelpad=10)
            plt.ylabel(cfg['ylabel'], labelpad=10)
            plt.legend()

        # Generate the two side-by-side plots
        create_subplot(1, plot_config['q_diff'])
        create_subplot(2, plot_config['wealth'])

        # Ensure proper spacing and display the figure
        plt.tight_layout(pad=3.0)
        plt.show()

config = Config()
qtable = QTable(num_wealth=config.num_wealth, num_time=config.T, num_actions=config.num_actions)
agent = QAgent(qtable, config)
agent.train()
agent.plot_results()


