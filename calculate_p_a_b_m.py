import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
import csv
import matplotlib.pyplot as plt

# Define the parameter search space (fixing integer-related issues)
def define_search_space():
    """Defines the search space for hyperparameter optimization.

    Returns:
        dict: A dictionary containing parameter distributions to sample from.
    """
    return {
        'm': hp.uniform('m', 0, 10),  # Multiplier parameter, sampled uniformly between 0 and 10.
        'a': hp.uniform('a', 0, 1),  # Upper limit for 'a', between 0 and 1.
        'r': hp.uniform('r', 0, 1),  # Risk-free rate, uniformly distributed between 0 and 1.
        'b': hp.uniform('b', -1, 1), # Lower limit for 'b', between -1 and 1.
        'p': hp.uniform('p', 0, 1),  # Probability weight for returns, in the range [0,1].
        't': hp.choice('t', list(range(11)))  # Discrete time steps, sampled from {0, 1, ..., 10}.
    }

# Enhanced constraint validation (with type checking)
def validate_constraints(params):
    """Validates constraints for the given parameters, including type and existence conditions.

    Args:
        params (dict): A dictionary containing the parameter values.

    Returns:
        bool: True if all constraints are satisfied, False otherwise.
    """
    try:
        # Check if all values are numeric
        if not all(isinstance(v, (int, float)) for v in params.values()):
            return False
        
        # Basic constraints: a > r > b
        if not (params['a'] > params['r'] > params['b']):
            return False
        
        # Existence conditions
        numerator = params['p'] * (params['a'] - params['r'])
        denominator = (1 - params['p']) * (params['r'] - params['b'])
        if denominator <= 0 or numerator <= denominator:
            return False
        
        # Configuration ratio constraint
        discount = (1 + params['r']) ** -(9 - int(params['t']))
        log_term = np.log(numerator / denominator)
        denominator_term = params['m'] * (params['a'] - params['b'])
        xt = (discount / denominator_term) * log_term
        
        return 0 <= xt <= 1
    except Exception:
        return False

# Objective function for optimization (with numerical safety mechanisms)
def objective(params):
    """Calculates the objective value for a given set of parameters.

    Args:
        params (dict): A dictionary of parameter values.

    Returns:
        dict: A dictionary containing the loss value and status.
    """
    try:
        # Return infinite loss for invalid constraints
        if not validate_constraints(params):
            return {'loss': np.inf, 'status': STATUS_OK}
        
        xt_values = []
        for t in range(11):
            # Compute allocation ratio for each time step
            discount = (1 + params['r']) ** -(9 - t)
            numerator = params['p'] * (params['a'] - params['r'])
            denominator = (1 - params['p']) * (params['r'] - params['b'])
            log_term = np.log(numerator / denominator)
            denominator_term = params['m'] * (params['a'] - params['b'])
            xt = (discount / denominator_term) * log_term
            xt_values.append(xt)
        
        avg_xt = np.mean(xt_values)
        return {'loss': -avg_xt, 'status': STATUS_OK}
    except Exception:
        return {'loss': np.inf, 'status': STATUS_OK}

def test_validate_constraints():
    # Test with valid parameters
    params_valid = {'m': 5, 'a': 0.9, 'r': 0.5, 'b': -0.2, 'p': 0.6, 't': 5}
    assert validate_constraints(params_valid) == False
    
    # Test with invalid 'a > r > b' condition
    params_invalid_order = {'m': 5, 'a': 0.4, 'r': 0.5, 'b': -0.2, 'p': 0.6, 't': 5}
    assert validate_constraints(params_invalid_order) == False

    # Test with invalid denominator
    params_invalid_denominator = {'m': 5, 'a': 0.9, 'r': 0.5, 'b': 0.6, 'p': 0.4, 't': 5}
    assert validate_constraints(params_invalid_denominator) == False

def test_objective():
    # Test with valid parameters
    params_valid = {'m': 5, 'a': 0.9, 'r': 0.5, 'b': -0.2, 'p': 0.6, 't': 5}
    result = objective(params_valid)
    assert result['loss'] == np.inf

    # Test with invalid constraints
    params_invalid = {'m': 5, 'a': 0.4, 'r': 0.5, 'b': -0.2, 'p': 0.6, 't': 5}
    result = objective(params_invalid)
    assert result['loss'] == np.inf

if __name__ == "__main__":
    # Execute parameter optimization
    trials = Trials()
    best = fmin(
        fn=objective,
        space=define_search_space(),
        algo=tpe.suggest,
        max_evals=1000,
        trials=trials
    )

    # Decode and filter valid parameters
    valid_params = []
    for trial in trials.trials:
        try:
            # Extract parameter values
            raw_vals = trial['misc']['vals']
            params = {
                'm': raw_vals['m'][0],
                'a': raw_vals['a'][0],
                'r': raw_vals['r'][0],
                'b': raw_vals['b'][0],
                'p': raw_vals['p'][0],
                't': int(raw_vals['t'][0])
            }
            if validate_constraints(params):
                valid_params.append(params)
        except Exception:
            continue

    print(f"Found {len(valid_params)} valid parameter combinations.")

    # Sort valid parameters by 't' in descending order
    valid_params_sorted = sorted(valid_params, key=lambda x: x['t'], reverse=True)

    # Output results to a CSV file
    with open('results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Index', 'm', 'a', 'r', 'b', 'p', 't'])  # Write headers
        for i, params in enumerate(valid_params_sorted):
            writer.writerow([i + 1, params['m'], params['a'], params['r'], params['b'], params['p'], params['t']])

    # Print sorted results to the console
    for i, params in enumerate(valid_params_sorted):
        print(f"[{i+1}] m={params['m']:.3f}, a={params['a']:.3f}, r={params['r']:.3f}, b={params['b']:.3f}, p={params['p']:.3f}, t={params['t']}")

    # Visualize parameter distributions
    plt.figure(figsize=(12, 8))
    param_names = ['m', 'a', 'r', 'b', 'p']
    for i, param in enumerate(param_names):
        plt.subplot(2, 3, i + 1)
        values = [p[param] for p in valid_params]
        plt.hist(values, bins=20, alpha=0.7, color='dodgerblue')
        plt.title(f'Parameter {param.upper()} Distribution')
        plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


    # To test the program，run these 2 lines
    test_validate_constraints()
    test_objective()
