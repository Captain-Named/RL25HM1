import sys
import numpy as np
import torch
import tabulate

# Import required classes from your module
from deepQlearning import AssetAllocDiscreteMDP, RiskyReturnDistribution, State, Q, Config

def test_asset_alloc_discrete_mdp():
    print("\n" + "="*50)
    print("TESTING AssetAllocDiscreteMDP CLASS")
    print("="*50)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create configuration
    config = Config()
    config.T = 5  # Shorter horizon for testing
    config.num_actions = 11  # Fewer actions for simplicity
    
    # Initialize Q-network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qnn = Q().to(device)
    
    # Initialize MDP components
    risky_dist = RiskyReturnDistribution(config.a, config.b, config.p)
    initial_state = State(0, config.initial_wealth)
    
    print(f"\n1. INITIALIZING MDP with parameters:")
    print(f"   - Time horizon (T): {config.T}")
    print(f"   - Initial wealth: {config.initial_wealth}")
    print(f"   - Risky asset returns: a={config.a:.4f}, b={config.b:.4f}, p={config.p:.4f}")
    print(f"   - Risk-free rate: {config.r:.4f}")
    print(f"   - Number of actions: {config.num_actions}")
    
    # Create MDP instance
    mdp = AssetAllocDiscreteMDP(risky_dist, initial_state, qnn, config)
    
    print("\n2. TESTING step() METHOD")
    print("   Simulating single steps with different actions...")
    
    # Test different actions
    test_actions = [0, 5, 10]  # Test min, mid, and max allocations
    test_results = []
    
    for action in test_actions:
        # Get allocation proportion
        alloc_prop = action / (config.num_actions - 1)
        
        # Force specific return for reproducibility (use a high return)
        risky_dist.sample = lambda: risky_dist.a
        
        # Execute step
        next_state, reward = mdp.step(initial_state, action)
        
        # Calculate expected wealth manually to verify
        risky_alloc = alloc_prop * initial_state.w
        expected_wealth = (risky_alloc * (1 + risky_dist.a) + 
                          (initial_state.w - risky_alloc) * (1 + config.r))
        
        test_results.append([
            action,
            f"{alloc_prop:.2f}",
            f"{next_state.t}",
            f"{next_state.w:.6f}",
            f"{expected_wealth:.6f}",
            f"{reward:.6f}"
        ])
    
    # Display results
    headers = ["Action", "Allocation %", "Next Time", "Next Wealth", "Expected Wealth", "Reward"]
    print("\n   Single Step Results:")
    print(tabulate.tabulate(test_results, headers=headers, tablefmt="grid"))
    
    print("\n3. TESTING get_episode() METHOD")
    print("   Generating complete episode trajectory...")
    
    # Reset MDP
    mdp = AssetAllocDiscreteMDP(risky_dist, initial_state, qnn, config)
    
    # Generate episode using epsilon-greedy policy
    print("   Using epsilon-greedy policy (exploration enabled):")
    episode = mdp.get_episode(greedy=False)
    
    # Prepare data for display
    episode_data = []
    for i in range(len(episode["states"])-1):
        state = episode["states"][i]
        action = episode["actions"][i]
        next_state = episode["states"][i+1]
        reward = episode["rewards"][i]
        alloc_prop = action / (config.num_actions - 1)
        
        episode_data.append([
            f"{i}",
            f"{state.t}",
            f"{state.w:.6f}",
            f"{action}",
            f"{alloc_prop:.2f}",
            f"{next_state.w:.6f}",
            f"{reward:.6f}"
        ])
    
    # Add terminal state
    terminal_state = episode["states"][-1]
    episode_data.append([
        f"{len(episode['states'])-1}",
        f"{terminal_state.t}",
        f"{terminal_state.w:.6f}",
        "N/A",
        "N/A",
        "N/A",
        "N/A"
    ])
    
    # Display episode trajectory
    headers = ["Step", "Time", "Wealth", "Action", "Allocation %", "Next Wealth", "Reward"]
    print("\n   Episode Trajectory:")
    print(tabulate.tabulate(episode_data, headers=headers, tablefmt="grid"))
    
    print("\n4. TESTING greedy policy behavior")
    print("   Generating episode with greedy policy (no exploration)...")
    
    # Reset MDP
    mdp = AssetAllocDiscreteMDP(risky_dist, initial_state, qnn, config)
    
    # Generate episode using greedy policy
    greedy_episode = mdp.get_episode(greedy=True)
    
    # Extract actions for comparison
    greedy_actions = greedy_episode["actions"]
    
    print(f"   Greedy policy actions: {greedy_actions}")
    print(f"   Terminal wealth: {greedy_episode['states'][-1].w:.6f}")
    
    print("\n5. TESTING with different random seeds")
    print("   Running multiple episodes to observe stochastic behavior...")
    
    # Run multiple episodes
    wealth_results = []
    for seed in range(5):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        mdp = AssetAllocDiscreteMDP(risky_dist, initial_state, qnn, config)
        episode = mdp.get_episode(greedy=True)
        terminal_wealth = episode["states"][-1].w
        wealth_results.append([f"Seed {seed}", f"{terminal_wealth:.6f}"])
    
    # Display terminal wealth results
    print("\n   Terminal Wealth Results:")
    print(tabulate.tabulate(wealth_results, headers=["Random Seed", "Terminal Wealth"], tablefmt="grid"))
    
    print("\n" + "="*50)
    print("AssetAllocDiscreteMDP TEST COMPLETE")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_asset_alloc_discrete_mdp()