import numpy as np

def calculate_xt(m:float, a: float,r: float, b: float, p: float, t_range: tuple = (0,10)) -> list:
    """Calculate optimal asset allocation ratios over time.
    
    Implements dynamic asset allocation model with existence condition check.
    Ensures output values are clamped between 0 and 1.
    
    Args:
        m (float): Risk aviod parameter
        a (float): Expected return of risky asset (a > r > b)
        b (float): Minimum return of risky asset
        r (float): Risk-free rate
        p (float): Probability of positive return scenario
        t_range (tuple, optional): Time period range. Defaults to (0,10).
    
    Returns:
        list: Optimal allocation ratios for each time period
    
    Raises:
        ValueError: When existence condition p(a-r)/[(1-p)(r-b)] <= 1
    
    Notes:
        Existence condition derived from Kelly criterion optimization:
        Requires p(a-r)/[(1-p)(r-b)] > 1 to ensure positive allocation
    """
    T = 10
    xt_list = []
    for t in range(T):
        # Time decay factor calculation
        discount = (1 + r)**-(9 - t) #if t <=9 else (1 + r)**(t-9)
        
        # Validate model existence condition
        ratio = (p*(a - r)) / ((1-p)*(r - b))
        if ratio <= 1:
            raise ValueError("Condition not met")
        
        # Core allocation formula
        log_term = np.log(ratio)
        denominator = m * (a - b)
        xt = (discount / denominator) * log_term
        
        # Enforce allocation constraint
        xt = min(1, xt)  # Clamp maximum allocation
        xt_list.append(round(xt, 2))
    
    return xt_list

print(calculate_xt(m=6.030, a=0.876, r=0.790, b=-0.052, p=0.931))
print(calculate_xt(m=1.771, a=0.921, r=0.572, b=-0.696, p=0.941))
print(calculate_xt(m=6.002, a=0.899, r=0.766, b=-0.859, p=0.925))
print(calculate_xt(m=7.413, a=0.781, r=0.496, b=-0.633, p=0.981))
print(calculate_xt(m=1.176, a=1.000, r=0.886, b=-0.490, p=0.938))
print(calculate_xt(m=7.262, a=0.620, r=0.334, b=0.106, p=0.516))
