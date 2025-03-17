import torch
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from copy import deepcopy
from tqdm import tqdm

# Import required classes from your module
from deepQlearning import DQNAgent, Q, Config, State, ReplayBuffer, RiskyReturnDistribution

def test_dqn_agent():
    print("\n" + "="*60)
    print("TESTING DQNAgent CLASS".center(60))
    print("="*60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Set up device for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Create configuration with smaller parameters for testing
    config = Config()
    config.T = 4  # Shorter time horizon
    config.num_episodes_per_epoch = 5  # Fewer episodes per epoch
    config.num_actions = 5  # Fewer actions for simplicity
    config.epsilon = 0.3  # Exploration rate
    config.batch_size = 4  # Smaller batch size
    config.num_epoch = 2  # Fewer epochs for testing
    config.num_batch_per_epoch = 10  # Fewer batches per epoch
    
    # Display test configuration
    print("\n1. TEST INITIALIZATION")
    print("\nTest configuration parameters:")
    config_data = [
        ["Time Horizon (T)", config.T],
        ["Episodes per Epoch", config.num_episodes_per_epoch],
        ["Number of Actions", config.num_actions],
        ["Exploration Rate (ε)", config.epsilon],
        ["Batch Size", config.batch_size],
        ["Test Epochs", config.num_epoch],
        ["Batches per Epoch", config.num_batch_per_epoch],
        ["Initial Wealth", config.initial_wealth],
        ["Learning Rate", config.alpha]
    ]
    print(tabulate(config_data, tablefmt="grid"))
    
    # Initialize Q-network and agent
    print("\nInitializing Q-network and DQNAgent...")
    qnn = Q().to(device)
    agent = DQNAgent(qnn, config)
    
    # Inspect agent attributes
    print("\nAgent initialized with attributes:")
    agent_attrs = [
        ["Q-network parameters", sum(p.numel() for p in agent.qnn.parameters())],
        ["Replay Buffer", "Empty" if not hasattr(agent.replay_buffer, "replay_buffer") or 
                          len(agent.replay_buffer.replay_buffer) == 0 else 
                          len(agent.replay_buffer.replay_buffer)],
        ["Optimizer", type(agent.optimizer).__name__],
        ["Loss Function", type(agent.loss_fn).__name__]
    ]
    print(tabulate(agent_attrs, tablefmt="grid"))
    
    # Test episode generation
    print("\n\n2. TESTING _get_episodes_from_mdp() METHOD")
    print("\nGenerating episodes with exploration (non-greedy)...")
    episodes = agent._get_episodes_from_mdp(num_episodes=3, greedy=False)
    
    # Display episode statistics
    episode_stats = []
    for i, episode in enumerate(episodes):
        terminal_wealth = episode["states"][-1].w
        actions = episode["actions"]
        rewards = episode["rewards"]
        episode_stats.append([
            f"Episode {i+1}",
            f"{len(actions)}",
            f"{actions}",
            f"{sum(rewards):.6f}",
            f"{terminal_wealth:.6f}"
        ])
    
    print("\nNon-greedy Episode Statistics:")
    print(tabulate(episode_stats, 
                   headers=["Episode", "Actions Count", "Actions Taken", 
                            "Total Reward", "Terminal Wealth"],
                   tablefmt="grid"))
    
    # Test greedy episode generation
    print("\nGenerating episodes with greedy policy...")
    greedy_episodes = agent._get_episodes_from_mdp(num_episodes=3, greedy=True)
    
    # Display greedy episode statistics
    greedy_stats = []
    for i, episode in enumerate(greedy_episodes):
        terminal_wealth = episode["states"][-1].w
        actions = episode["actions"]
        rewards = episode["rewards"]
        greedy_stats.append([
            f"Episode {i+1}",
            f"{len(actions)}",
            f"{actions}",
            f"{sum(rewards):.6f}",
            f"{terminal_wealth:.6f}"
        ])
    
    print("\nGreedy Episode Statistics:")
    print(tabulate(greedy_stats, 
                   headers=["Episode", "Actions Count", "Actions Taken", 
                            "Total Reward", "Terminal Wealth"],
                   tablefmt="grid"))
    
    # Test train method
    print("\n\n3. TESTING train() METHOD")
    print("\nPerforming a training epoch...")
    
    # Capture initial network state
    initial_params = deepcopy([p.clone().detach() for p in agent.qnn.parameters()])
    
    # Execute training
    agent.train()
    
    # Check if parameters were updated
    updated_params = [p.clone().detach() for p in agent.qnn.parameters()]
    params_changed = False
    for i, (init, updated) in enumerate(zip(initial_params, updated_params)):
        if not torch.allclose(init, updated):
            params_changed = True
            break
    
    print(f"\nParameters changed after training: {'Yes' if params_changed else 'No'}")
    
    # Display loss history
    if agent.loss_history:
        print("\nLoss History Statistics:")
        loss_stats = [
            ["Mean Loss", f"{np.mean(agent.loss_history):.6f}"],
            ["Min Loss", f"{np.min(agent.loss_history):.6f}"],
            ["Max Loss", f"{np.max(agent.loss_history):.6f}"]
        ]
        print(tabulate(loss_stats, tablefmt="grid"))
    else:
        print("\nNo loss history recorded during training.")
    
    # Test get_final_wealth_evaluation
    print("\n\n4. TESTING get_final_wealth_evaluation() METHOD")
    
    # Clear existing data
    agent.final_wealth = []
    
    # Run evaluations multiple times
    print("\nRunning wealth evaluations...")
    for _ in range(5):
        agent.get_final_wealth_evaluation()
    
    # Display wealth statistics
    if agent.final_wealth:
        print("\nWealth Evaluation Results:")
        wealth_stats = []
        for i, wealth in enumerate(agent.final_wealth):
            wealth_stats.append([f"Evaluation {i+1}", f"{wealth:.6f}"])
        
        wealth_stats.append(["Mean Wealth", f"{np.mean(agent.final_wealth):.6f}"])
        wealth_stats.append(["Min Wealth", f"{np.min(agent.final_wealth):.6f}"])
        wealth_stats.append(["Max Wealth", f"{np.max(agent.final_wealth):.6f}"])
        
        print(tabulate(wealth_stats, tablefmt="grid"))
    else:
        print("\nNo wealth evaluations recorded.")
    
    # Test multiple epoch training
    print("\n\n5. TESTING COMPLETE TRAINING PROCESS")
    
    # Reset agent for fresh training
    qnn_new = Q().to(device)
    agent_new = DQNAgent(qnn_new, config)
    
    # Run mini training process
    print("\nRunning complete training process with multiple epochs...")
    for epoch in tqdm(range(config.num_epoch)):
        agent_new.train()
        
        # Track epsilon decay
        if epoch > 0:
            print(f"Epoch {epoch+1}: Epsilon decayed to {agent_new.config.epsilon:.4f}")
    
    # Test plot_results
    print("\n\n6. TESTING plot_results() METHOD")
    print("\nGenerating performance visualization...")
    
    # Check if we have data to plot
    if agent_new.loss_history and agent_new.final_wealth:
        # Save the original plot function
        original_plot = plt.show
        
        # Replace with our test version that doesn't display
        plt.show = lambda: None
        
        # Run the plot function
        agent_new.plot_results()
        
        # Restore the original plot function
        plt.show = original_plot
        
        print("\nPlot generated successfully with:")
        print(f"- {len(agent_new.loss_history)} loss data points")
        print(f"- {len(agent_new.final_wealth)} wealth evaluation data points")
    else:
        print("\nInsufficient data for plotting results.")
    
    print("\n" + "="*60)
    print("DQNAgent TESTING COMPLETE".center(60))
    print("="*60 + "\n")

if __name__ == "__main__":
    test_dqn_agent()