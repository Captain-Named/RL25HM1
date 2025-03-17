import numpy as np
import time
from tabulate import tabulate
from qlearning import QAgent, QTable, Config, AssetAllocDiscreteMDP

def test_qagent():
    print("\n" + "="*60)
    print("TESTING QAgent CLASS".center(60))
    print("="*60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Create configuration with simplified parameters for testing
    print("\n1. INITIALIZATION TEST")
    print("\nCreating test configuration with reduced parameters:")
    
    config = Config()
    # Modify config for faster testing
    config.T = 4  # Shorter time horizon
    config.num_episodes = 100  # Much fewer episodes for testing
    config.num_wealth = 10  # Fewer wealth levels
    config.num_actions = 5  # Fewer actions
    config.epsilon = 0.3  # Exploration rate
    
    # Display test configuration
    config_data = [
        ["Time horizon (T)", config.T],
        ["Number of episodes", config.num_episodes],
        ["Number of wealth levels", config.num_wealth],
        ["Number of actions", config.num_actions],
        ["Learning rate (α)", config.alpha],
        ["Exploration rate (ε)", config.epsilon],
        ["Risk aversion (utility_a)", config.utility_a],
        ["Initial wealth", config.initial_wealth],
        ["Max wealth", f"{config.max_wealth:.4f}"]
    ]
    print(tabulate(config_data, tablefmt="grid"))
    
    # Initialize Q-table
    qtable = QTable(num_wealth=config.num_wealth, num_time=config.T, num_actions=config.num_actions)
    
    # Create QAgent
    print("\nInitializing QAgent...")
    agent = QAgent(qtable, config)
    
    print("\n2. TESTING _get_episode_from_mdp() METHOD")
    print("\n2.1 Testing with exploration (non-greedy):")
    
    # Generate episode with exploration
    explore_episode = agent._get_episode_from_mdp(greedy=False)
    
    # Display episode data
    episode_states = explore_episode["states"]
    episode_actions = explore_episode["actions"]
    episode_rewards = explore_episode["rewards"]
    
    print(f"  Episode length: {len(episode_actions)} actions, {len(episode_states)} states")
    print(f"  Terminal state: (t={episode_states[-1].t}, w={episode_states[-1].w})")
    print(f"  Terminal wealth level: {episode_states[-1].w} (of {config.num_wealth-1} max)")
    print(f"  Actions taken: {episode_actions}")
    print(f"  Total reward: {sum(episode_rewards):.6f}")
    
    print("\n2.2 Testing with greedy policy:")
    
    # Generate episode with greedy policy
    greedy_episode = agent._get_episode_from_mdp(greedy=True)
    
    # Display episode data
    greedy_states = greedy_episode["states"]
    greedy_actions = greedy_episode["actions"]
    greedy_rewards = greedy_episode["rewards"]
    
    print(f"  Episode length: {len(greedy_actions)} actions, {len(greedy_states)} states")
    print(f"  Terminal state: (t={greedy_states[-1].t}, w={greedy_states[-1].w})")
    print(f"  Terminal wealth level: {greedy_states[-1].w} (of {config.num_wealth-1} max)")
    print(f"  Actions taken: {greedy_actions}")
    print(f"  Total reward: {sum(greedy_rewards):.6f}")
    
    # Compare with exploration episode
    print("\n2.3 Comparison between greedy and exploratory episodes:")
    compare_data = [
        ["Policy Type", "Terminal Wealth Level", "Actions Taken", "Total Reward"],
        ["Exploratory", f"{episode_states[-1].w}", f"{episode_actions}", f"{sum(episode_rewards):.6f}"],
        ["Greedy", f"{greedy_states[-1].w}", f"{greedy_actions}", f"{sum(greedy_rewards):.6f}"]
    ]
    print(tabulate(compare_data[1:], headers=compare_data[0], tablefmt="grid"))
    
    print("\n3. TESTING train() METHOD")
    print("\nRunning mini-training (100 episodes)...")
    
    # Capture Q-table state before training
    q_before = agent.qtable.q.copy()
    
    # Track time for performance measurement
    start_time = time.time()
    
    # Run training for a small number of episodes
    agent.train()
    
    # Calculate training time
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds ({train_time/config.num_episodes*1000:.2f} ms per episode)")
    
    # Compare Q-values before and after training
    q_after = agent.qtable.q.copy()
    q_diff = np.sum(np.abs(q_after - q_before))
    q_changed_pct = np.mean(np.abs(q_after - q_before) > 0.0001) * 100
    
    print(f"\nQ-value changes:")
    print(f"  Total absolute change: {q_diff:.6f}")
    print(f"  Percentage of Q-values changed: {q_changed_pct:.2f}%")
    
    # Display QDelta convergence pattern
    if agent.qdelta:
        qdelta_data = [
            ["Statistic", "Value"],
            ["Number of QDelta points", len(agent.qdelta)],
            ["Initial QDelta", f"{agent.qdelta[0]:.6f}"],
            ["Final QDelta", f"{agent.qdelta[-1]:.6f}"],
            ["Ratio (Final/Initial)", f"{agent.qdelta[-1]/agent.qdelta[0]:.6f}"],
            ["Mean QDelta", f"{np.mean(agent.qdelta):.6f}"],
            ["Min QDelta", f"{np.min(agent.qdelta):.6f}"],
            ["Max QDelta", f"{np.max(agent.qdelta):.6f}"]
        ]
        print("\nQDelta Convergence Statistics:")
        print(tabulate(qdelta_data[1:], headers=qdelta_data[0], tablefmt="grid"))
    
    # Test final wealth measurements
    if agent.final_wealth:
        wealth_data = [
            ["Statistic", "Value"],
            ["Mean Terminal Wealth Level", f"{np.mean(agent.final_wealth):.2f}"],
            ["Min Terminal Wealth Level", f"{np.min(agent.final_wealth)}"],
            ["Max Terminal Wealth Level", f"{np.max(agent.final_wealth)}"]
        ]
        print("\nTerminal Wealth Statistics:")
        print(tabulate(wealth_data[1:], headers=wealth_data[0], tablefmt="grid"))
    
    print("\n4. TESTING show_policy() METHOD")
    print("\nGenerating sample policy visualizations:")
    
    # Save original print function
    original_print = print
    
    # Capture policy output
    import io
    import sys
    
    policy_output = []
    for i in range(2):  # Capture two policy outputs
        # Redirect stdout to capture print output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Call show_policy with small modifications for testing
        agent.show_policy()
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        policy_output.append(captured_output.getvalue())
    
    # Display the captured policies
    print("First policy visualization:")
    print(policy_output[0])
    print("Second policy visualization:")
    print(policy_output[1])
    
    print("\n5. TESTING plot_results() METHOD")
    
    # Verify that we have data to plot
    has_qdelta = len(agent.qdelta) > 0
    has_final_wealth = len(agent.final_wealth) > 0
    
    print(f"\nQDelta data available: {'Yes' if has_qdelta else 'No'} ({len(agent.qdelta)} points)")
    print(f"Final wealth data available: {'Yes' if has_final_wealth else 'No'} ({len(agent.final_wealth)} points)")
    
    if has_qdelta and has_final_wealth:
        # Temporarily disable actual plotting
        import matplotlib.pyplot as plt
        original_show = plt.show
        plt.show = lambda: None
        
        # Call plot_results
        try:
            agent.plot_results()
            print("\nPlot generated successfully.")
            print("The plot would show:")
            print("  - Left panel: Q-Value Convergence Pattern")
            print("  - Right panel: Portfolio Value Development")
        except Exception as e:
            print(f"\nError generating plot: {str(e)}")
        finally:
            # Restore original show function
            plt.show = original_show
    else:
        print("\nInsufficient data for plotting.")
    
    # Test additional greedy episode performance after training
    print("\n6. EVALUATING TRAINED POLICY")
    
    print("\nRunning 10 greedy episodes with trained policy...")
    wealth_levels = []
    for _ in range(10):
        greedy_episode = agent._get_episode_from_mdp(greedy=True)
        wealth_levels.append(greedy_episode["states"][-1].w)
    
    # Display results
    eval_data = [
        ["Episode", "Terminal Wealth Level"],
        *[[i+1, level] for i, level in enumerate(wealth_levels)]
    ]
    print("\nEvaluation Results:")
    print(tabulate(eval_data[1:], headers=eval_data[0], tablefmt="grid"))
    
    # Calculate statistics
    mean_wealth = np.mean(wealth_levels)
    std_wealth = np.std(wealth_levels)
    
    print(f"\nMean terminal wealth level: {mean_wealth:.2f}")
    print(f"Standard deviation: {std_wealth:.2f}")
    
    print("\n" + "="*60)
    print("QAgent TESTING COMPLETE".center(60))
    print("="*60 + "\n")

if __name__ == "__main__":
    test_qagent()