import numpy as np
import random
from tabulate import tabulate
from qlearning import QTable, State

def test_qtable():
    print("\n" + "="*60)
    print("TESTING QTable CLASS".center(60))
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Initialize test parameters
    num_wealth = 10
    num_time = 5
    num_actions = 4
    
    print("\n1. INITIALIZATION TEST")
    print(f"\nCreating QTable with dimensions:")
    print(f"  - Time steps: {num_time}")
    print(f"  - Wealth levels: {num_wealth}")
    print(f"  - Actions: {num_actions}")
    
    # Create QTable instance
    qtable = QTable(num_wealth, num_time, num_actions)
    
    # Check shape of the table
    actual_shape = qtable.q.shape
    expected_shape = (num_time, num_wealth, num_actions)
    
    print(f"\nVerifying dimensions:")
    print(f"  - Expected shape: {expected_shape}")
    print(f"  - Actual shape: {actual_shape}")
    print(f"  - Correct dimensions: {'Yes' if actual_shape == expected_shape else 'No'}")
    
    # Display initialization stats
    init_stats = [
        ["Min value", f"{np.min(qtable.q):.6f}"],
        ["Max value", f"{np.max(qtable.q):.6f}"],
        ["Mean value", f"{np.mean(qtable.q):.6f}"],
        ["Standard deviation", f"{np.std(qtable.q):.6f}"]
    ]
    
    print("\nInitialization Statistics:")
    print(tabulate(init_stats, tablefmt="grid"))
    
    # Test the get method
    print("\n2. GET METHOD TEST")
    
    # Sample some random states and actions
    test_states = [
        State(0, 0),     # Beginning state with lowest wealth
        State(2, 5),     # Middle state with mid wealth
        State(4, 9)      # Terminal state with highest wealth
    ]
    
    test_actions = [0, 1, 3]  # First action, second action, and last action
    
    # Create a table for the results
    get_results = []
    
    for state in test_states:
        for action in test_actions:
            # Get the value from the table
            q_value = qtable.get(state, action)
            get_results.append([
                f"({state.t}, {state.w})",
                f"{action}",
                f"{q_value:.6f}"
            ])
    
    print("\nSampling Q-values with get() method:")
    print(tabulate(get_results, headers=["State (t,w)", "Action", "Q-Value"], tablefmt="grid"))
    
    # Test the update method
    print("\n3. UPDATE METHOD TEST")
    
    # Store original values
    original_values = {}
    update_tests = []
    
    for state in test_states:
        for action in test_actions:
            # Store the original value
            original_values[(state.t, state.w, action)] = qtable.get(state, action)
            
            # Generate a new value (just add 1.0 to make it clearly different)
            new_value = original_values[(state.t, state.w, action)] + 1.0
            
            # Update the Q-table
            qtable.update(state, action, new_value)
            
            # Get the updated value
            updated_value = qtable.get(state, action)
            
            # Add to results
            update_tests.append([
                f"({state.t}, {state.w})",
                f"{action}",
                f"{original_values[(state.t, state.w, action)]:.6f}",
                f"{new_value:.6f}",
                f"{updated_value:.6f}",
                "Yes" if abs(updated_value - new_value) < 1e-9 else "No"
            ])
    
    print("\nUpdating Q-values:")
    print(tabulate(update_tests, 
                  headers=["State (t,w)", "Action", "Original Value", 
                          "New Value", "Retrieved Value", "Update Successful"], 
                  tablefmt="grid"))
    
    # Test the effect on neighboring values (to verify updates are isolated)
    print("\n4. LOCALIZATION TEST")
    
    # Select a specific state and action to update
    test_state = State(2, 4)
    test_action = 2
    
    # Get the original value at the test state/action
    original_value = qtable.get(test_state, test_action)
    
    # Get values of neighbors before update
    neighbor_states = [
        State(1, 4),  # Same wealth, previous time
        State(3, 4),  # Same wealth, next time
        State(2, 3),  # Previous wealth, same time
        State(2, 5),  # Next wealth, same time
    ]
    neighbor_actions = [1, 3]  # Actions adjacent to the test action
    
    # Store neighbor values before update
    neighbors_before = {}
    for n_state in neighbor_states:
        for n_action in neighbor_actions:
            neighbors_before[(n_state.t, n_state.w, n_action)] = qtable.get(n_state, n_action)
    
    # Also check the value for the same state but different action
    same_state_diff_action_before = qtable.get(test_state, (test_action + 1) % num_actions)
    
    # Now apply a large update to the test state/action
    new_value = 100.0  # A value unlikely to naturally occur
    qtable.update(test_state, test_action, new_value)
    
    # Verify the update occurred
    updated_value = qtable.get(test_state, test_action)
    
    # Check if neighbors were affected
    isolation_tests = []
    isolation_tests.append([
        f"({test_state.t}, {test_state.w})",
        f"{test_action}",
        f"{original_value:.6f}",
        f"{new_value:.6f}",
        f"{updated_value:.6f}",
        "Yes" if abs(updated_value - new_value) < 1e-9 else "No"
    ])
    
    # Check neighbors
    all_preserved = True
    for n_state in neighbor_states:
        for n_action in neighbor_actions:
            before = neighbors_before[(n_state.t, n_state.w, n_action)]
            after = qtable.get(n_state, n_action)
            preserved = abs(before - after) < 1e-9
            all_preserved &= preserved
            
            isolation_tests.append([
                f"({n_state.t}, {n_state.w})",
                f"{n_action}",
                f"{before:.6f}",
                "N/A",
                f"{after:.6f}",
                "Yes" if preserved else "No"
            ])
    
    # Check same state but different action
    same_state_diff_action_after = qtable.get(test_state, (test_action + 1) % num_actions)
    preserved = abs(same_state_diff_action_before - same_state_diff_action_after) < 1e-9
    all_preserved &= preserved
    
    isolation_tests.append([
        f"({test_state.t}, {test_state.w})",
        f"{(test_action + 1) % num_actions}",
        f"{same_state_diff_action_before:.6f}",
        "N/A",
        f"{same_state_diff_action_after:.6f}",
        "Yes" if preserved else "No"
    ])
    
    print("\nTesting isolation of updates (neighbors should be unaffected):")
    print(tabulate(isolation_tests, 
                  headers=["State (t,w)", "Action", "Before Update", 
                          "New Value", "After Update", "Preserved"], 
                  tablefmt="grid"))
    
    print(f"\nAll neighbor values preserved: {'Yes' if all_preserved else 'No'}")
    
    # Test efficiency of access and update operations
    print("\n5. PERFORMANCE TEST")
    
    import time
    
    # Number of operations to test
    num_ops = 10000
    
    # Generate random states and actions
    random_states = [State(np.random.randint(0, num_time), 
                          np.random.randint(0, num_wealth)) 
                    for _ in range(num_ops)]
    random_actions = [np.random.randint(0, num_actions) for _ in range(num_ops)]
    random_values = [np.random.random() for _ in range(num_ops)]
    
    # Test get performance
    start_time = time.time()
    for i in range(num_ops):
        qtable.get(random_states[i], random_actions[i])
    get_time = time.time() - start_time
    
    # Test update performance
    start_time = time.time()
    for i in range(num_ops):
        qtable.update(random_states[i], random_actions[i], random_values[i])
    update_time = time.time() - start_time
    
    # Display performance results
    perf_results = [
        ["Get", f"{num_ops}", f"{get_time:.6f} sec", f"{get_time/num_ops*1e6:.2f} μs"],
        ["Update", f"{num_ops}", f"{update_time:.6f} sec", f"{update_time/num_ops*1e6:.2f} μs"]
    ]
    
    print("\nPerformance of access and update operations:")
    print(tabulate(perf_results, 
                  headers=["Operation", "Number of Calls", "Total Time", "Time per Call"], 
                  tablefmt="grid"))
    
    print("\n" + "="*60)
    print("QTable TESTING COMPLETE".center(60))
    print("="*60 + "\n")

if __name__ == "__main__": 
    test_qtable()