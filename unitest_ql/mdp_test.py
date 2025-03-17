import numpy as np
import random
from tabulate import tabulate
from qlearning import AssetAllocDiscreteMDP, QTable, State, Config, RiskyReturnDistribution

def test_asset_alloc_discrete_mdp():
    print("\n" + "="*60)
    print("TESTING AssetAllocDiscreteMDP CLASS".center(60))
    print("="*60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create configuration with simplified parameters for testing
    config = Config()
    config.T = 4  # Shorter time horizon
    config.num_actions = 5  # Fewer actions
    config.num_wealth = 10  # Fewer wealth levels
    config.initial_wealth = 1.0
    config.max_wealth = 2.0
    
    # Initialize QTable
    qtable = QTable(config.num_wealth, config.T, config.num_actions)
    
    # Set some specific Q-values to influence action selection in greedy mode
    for t in range(config.T):
        for w in range(config.num_wealth):
            # Create a pattern where the optimal action increases with wealth
            optimal_action = min(w % config.num_actions, config.num_actions - 1)
            for a in range(config.num_actions):
                # Set higher Q-values for "optimal" actions
                if a == optimal_action:
                    qtable.update(State(t, w), a, 1.0)
                else:
                    qtable.update(State(t, w), a, -0.1)
    
    print("\n1. INITIALIZATION TEST")
    print(f"\nCreating AssetAllocDiscreteMDP with parameters:")
    for param, value in [
        ("Time horizon (T)", config.T),
        ("Initial wealth", config.initial_wealth),
        ("Max wealth", config.max_wealth),
        ("Num wealth levels", config.num_wealth),
        ("Num actions", config.num_actions),
        ("Risky returns", f"a={config.a:.3f}, b={config.b:.3f}, p={config.p:.2f}"),
        ("Risk-free rate", config.r)
    ]:
        print(f"  - {param}: {value}")
    
    # Create AssetAllocDiscreteMDP instance
    mdp = AssetAllocDiscreteMDP(config, qtable)
    
    # Verify initial state
    print(f"\nInitial state: Time={mdp.state.t}, Wealth level={mdp.state.w}")
    print(f"Initial discretized wealth: {mdp.state.w / (mdp.num_wealth-1) * mdp.max_wealth:.4f}")
    
    print("\n2. STEP METHOD TEST")
    print("\nTesting step() with different actions from initial state:")
    
    # Create a copy of the initial state for testing
    test_state = State(0, mdp.state.w)
    
    # Test actions
    test_actions = list(range(config.num_actions))
    step_results = []
    
    # Override sample method for deterministic testing
    original_sample = mdp.risky_return_distribution.sample
    mdp.risky_return_distribution.sample = lambda: mdp.risky_return_distribution.a
    
    for action in test_actions:
        # Calculate allocation percentage
        alloc_pct = action / (config.num_actions - 1)
        
        # Get next state and reward
        next_state, reward = mdp.step(test_state, action)
        
        # Calculate expected wealth (for verification)
        current_wealth = test_state.w / (mdp.num_wealth-1) * mdp.max_wealth
        risky_alloc = alloc_pct * current_wealth
        expected_next_wealth = min(mdp.max_wealth, 
                                  risky_alloc * (1 + mdp.risky_return_distribution.a) + 
                                  (current_wealth - risky_alloc) * (1 + mdp.riskless_returns))
        expected_wealth_level = round(expected_next_wealth / mdp.max_wealth * (mdp.num_wealth - 1))
        
        step_results.append([
            action,
            f"{alloc_pct*100:.1f}%",
            f"{current_wealth:.4f}",
            f"{next_state.t}",
            f"{next_state.w}",
            f"{expected_wealth_level}",
            f"{reward:.6f}"
        ])
    
    # Restore original sample method
    mdp.risky_return_distribution.sample = original_sample
    
    # Display step results
    print(tabulate(step_results, headers=[
        "Action", "Allocation", "Current Wealth", 
        "Next Time", "Next Wealth Level", "Expected Level", "Reward"], 
        tablefmt="grid"))
    
    print("\n3. GET_EPISODE METHOD TEST")
    print("\n3.1 Testing get_episode() with exploration (non-greedy):")
    
    # Reset MDP for testing get_episode with exploration
    mdp = AssetAllocDiscreteMDP(config, qtable)
    
    # Set high exploration rate for testing
    mdp.epsilon = 1.0  # 100% random actions
    
    # Generate episode with full exploration
    explore_episode = mdp.get_episode(greedy=False)
    
    # Format episode data
    episode_data = []
    for i in range(len(explore_episode["states"])-1):
        state = explore_episode["states"][i]
        action = explore_episode["actions"][i]
        next_state = explore_episode["states"][i+1]
        reward = explore_episode["rewards"][i]
        alloc_pct = action / (config.num_actions - 1)
        
        episode_data.append([
            f"{i}",
            f"{state.t}",
            f"{state.w}",
            f"{action}",
            f"{alloc_pct*100:.1f}%",
            f"{next_state.w}",
            f"{reward:.6f}"
        ])
    
    # Add terminal state
    terminal_state = explore_episode["states"][-1]
    episode_data.append([
        f"{len(explore_episode['states'])-1}",
        f"{terminal_state.t}",
        f"{terminal_state.w}",
        "N/A",
        "N/A",
        "N/A",
        "N/A"
    ])
    
    print("\nEpisode Trajectory with Exploration:")
    print(tabulate(episode_data, headers=[
        "Step", "Time", "Wealth Level", "Action", 
        "Allocation", "Next Wealth", "Reward"], 
        tablefmt="grid"))
    
    print("\n3.2 Testing get_episode() with greedy policy:")
    
    # Reset MDP for testing get_episode with greedy policy
    mdp = AssetAllocDiscreteMDP(config, qtable)
    
    # Generate episode with greedy policy
    greedy_episode = mdp.get_episode(greedy=True)
    
    # Format greedy episode data
    greedy_data = []
    for i in range(len(greedy_episode["states"])-1):
        state = greedy_episode["states"][i]
        action = greedy_episode["actions"][i]
        next_state = greedy_episode["states"][i+1]
        reward = greedy_episode["rewards"][i]
        alloc_pct = action / (config.num_actions - 1)
        
        # Get Q-values for current state to verify action selection
        q_values = [qtable.get(state, a) for a in range(config.num_actions)]
        optimal_action = np.argmax(q_values)
        
        greedy_data.append([
            f"{i}",
            f"{state.t}",
            f"{state.w}",
            f"{action}",
            f"{optimal_action}",
            f"{'Yes' if action == optimal_action else 'No'}",
            f"{alloc_pct*100:.1f}%",
            f"{next_state.w}",
            f"{reward:.6f}"
        ])
    
    # Add terminal state
    terminal_state = greedy_episode["states"][-1]
    greedy_data.append([
        f"{len(greedy_episode['states'])-1}",
        f"{terminal_state.t}",
        f"{terminal_state.w}",
        "N/A",
        "N/A",
        "N/A",
        "N/A",
        "N/A",
        "N/A"
    ])
    
    print("\nGreedy Episode Trajectory:")
    print(tabulate(greedy_data, headers=[
        "Step", "Time", "Wealth Level", "Action", "Optimal Action",
        "Followed Policy", "Allocation", "Next Wealth", "Reward"], 
        tablefmt="grid"))
    
    print("\n4. STOCHASTIC BEHAVIOR TEST")
    print("\nRunning multiple episodes to demonstrate stochastic environment:")
    
    # Run multiple episodes with fixed policy
    num_test_episodes = 5
    terminal_wealth_levels = []
    
    for i in range(num_test_episodes):
        # Set different random seed
        np.random.seed(42 + i)
        
        # Create new MDP
        test_mdp = AssetAllocDiscreteMDP(config, qtable)
        
        # Use fixed action sequence for all episodes
        fixed_actions = [2] * config.T  # Mid-level allocation for all steps
        
        # Track states and rewards
        test_states = [test_mdp.state]
        test_rewards = []
        
        # Manually step through environment with fixed actions
        for action in fixed_actions:
            next_state, reward = test_mdp.step(test_states[-1], action)
            test_states.append(next_state)
            test_rewards.append(reward)
        
        # Record terminal wealth
        terminal_wealth_levels.append(test_states[-1].w)
    
    # Display terminal wealth levels
    wealth_data = [[f"Episode {i+1}", level] for i, level in enumerate(terminal_wealth_levels)]
    print("\nTerminal Wealth Levels (with identical actions but different return samples):")
    print(tabulate(wealth_data, headers=["Episode", "Terminal Wealth Level"], tablefmt="grid"))
    
    # Calculate statistics
    unique_levels = len(set(terminal_wealth_levels))
    print(f"\nUnique terminal wealth levels: {unique_levels}")
    print(f"Demonstrates {'stochastic' if unique_levels > 1 else 'deterministic'} environment behavior")
    
    print("\n" + "="*60)
    print("AssetAllocDiscreteMDP TESTING COMPLETE".center(60))
    print("="*60 + "\n")

if __name__ == "__main__":    
    test_asset_alloc_discrete_mdp()