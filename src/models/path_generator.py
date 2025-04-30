import numpy as np
import pandas as pd
from src.models.pdf_cdf import sample_from_cdf

def generate_random_path(cdf, bin_edges, num_days=1000, drift=0.0):
    """
    Generate a random price path using the provided CDF.
    
    Args:
        cdf (numpy.ndarray): Cumulative distribution function
        bin_edges (numpy.ndarray): Bin edges corresponding to the CDF
        num_days (int): Number of days to simulate (default: 1000)
        drift (float): Daily drift parameter to add to returns (default: 0.0)
        
    Returns:
        numpy.ndarray: Array of daily returns
    """
    # Generate random returns using the CDF
    returns = sample_from_cdf(cdf, bin_edges, num_days)
    
    # Add drift if specified
    if drift != 0.0:
        returns = returns + drift
    
    return returns

def calculate_price_path(returns, initial_price=1.0):
    """
    Calculate price path from returns.
    
    Args:
        returns (numpy.ndarray): Array of daily returns
        initial_price (float): Initial price (default: 1.0)
        
    Returns:
        numpy.ndarray: Price path
    """
    # Calculate cumulative returns
    cum_returns = np.cumprod(1 + returns)
    
    # Calculate price path
    price_path = initial_price * cum_returns
    
    return price_path

def generate_multiple_paths(cdf, bin_edges, num_paths=100, num_days=1000, initial_price=1.0, drift=0.0):
    """
    Generate multiple random price paths.
    
    Args:
        cdf (numpy.ndarray): Cumulative distribution function
        bin_edges (numpy.ndarray): Bin edges corresponding to the CDF
        num_paths (int): Number of paths to generate (default: 100)
        num_days (int): Number of days in each path (default: 1000)
        initial_price (float): Initial price (default: 1.0)
        drift (float): Daily drift parameter to add to returns (default: 0.0)
        
    Returns:
        numpy.ndarray: Array of price paths, shape (num_paths, num_days+1)
            Each row is a separate path, with the first column being the initial price
    """
    try:
        # Check for empty inputs
        if len(cdf) == 0 or len(bin_edges) == 0:
            print("Warning: Empty CDF or bin_edges in generate_multiple_paths")
            return np.ones((num_paths, num_days + 1))  # Safe fallback
        
        # Check for NaN values in inputs
        if np.isnan(cdf).any() or np.isnan(bin_edges).any():
            print("Warning: NaN values in CDF or bin_edges")
            # Clean up inputs
            cdf = np.nan_to_num(cdf, nan=0.0)
            bin_edges = np.nan_to_num(bin_edges, nan=0.0)
        
        # Initialize array for paths
        paths = np.zeros((num_paths, num_days + 1))
        paths[:, 0] = initial_price
        
        # Generate random paths
        for i in range(num_paths):
            returns = sample_from_cdf(cdf, bin_edges, num_days)
            
            # Check for NaN values in returns
            returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=-0.99)
            
            # Add drift if specified
            if drift != 0.0:
                returns = returns + drift
            
            # Ensure reasonable return bounds to prevent extreme price paths
            # More conservative bounds for more realistic results
            returns = np.clip(returns, -0.3, 0.3)  # Limit to -30% to +30% daily returns
                
            price_path = calculate_price_path(returns, initial_price)
            
            # Check for NaN or infinite values in price path
            price_path = np.nan_to_num(price_path, nan=initial_price, posinf=initial_price*100, neginf=initial_price*0.01)
            
            # More aggressive checks to prevent unrealistic values
            # Ensure price path doesn't go too close to zero or too high
            # Use more realistic limits based on historical crypto performance
            price_path = np.maximum(price_path, initial_price * 1e-3)  # Minimum 0.1% of initial
            
            # For more realistic simulation, cap the maximum price 
            # For example, cap at 300x initial value which is still extremely high
            # but prevents unrealistic 1000x+ returns
            price_path = np.minimum(price_path, initial_price * 300)
            
            # Store path
            paths[i, 1:] = price_path
            
        # Final safety check for the entire paths array
        # Check for any remaining NaN or infinite values
        if np.isnan(paths).any() or np.isinf(paths).any():
            print("Warning: NaN or infinite values found in generated paths. Fixing...")
            paths = np.nan_to_num(paths, nan=initial_price, posinf=initial_price*100, neginf=initial_price*0.01)
        
        # One final check for extreme values in the entire array
        paths = np.maximum(paths, initial_price * 1e-3)  # Minimum 0.1% of initial
        paths = np.minimum(paths, initial_price * 300)   # Maximum 300x initial
        
        return paths
        
    except Exception as e:
        print(f"Error in generate_multiple_paths: {e}")
        # Return a safe fallback (all ones)
        return np.ones((num_paths, num_days + 1))

def calculate_drift_parameters(returns):
    """
    Calculate various drift parameters from historical returns.
    
    Args:
        returns (numpy.ndarray): Historical returns
        
    Returns:
        dict: Dictionary containing different drift calculations
    """
    log_returns = np.log(1 + returns)
    
    # Calculate different drift metrics
    arithmetic_drift = np.mean(returns)
    geometric_drift = np.mean(log_returns)
    adjusted_drift = geometric_drift - (np.var(log_returns) / 2)
    
    # Annual equivalents (assuming daily returns)
    annual_arithmetic = (1 + arithmetic_drift)**252 - 1
    annual_geometric = np.exp(geometric_drift * 252) - 1
    annual_adjusted = np.exp(adjusted_drift * 252) - 1
    
    return {
        "arithmetic": {
            "daily": arithmetic_drift,
            "annual": annual_arithmetic
        },
        "geometric": {
            "daily": geometric_drift,
            "annual": annual_geometric
        },
        "adjusted": {
            "daily": adjusted_drift,
            "annual": annual_adjusted
        }
    } 