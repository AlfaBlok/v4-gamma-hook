import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def calculate_returns(data, price_col='Close'):
    """
    Calculate daily returns from price data.
    
    Args:
        data (pandas.DataFrame): Historical price data
        price_col (str): Column name for price data (default: 'Close')
        
    Returns:
        numpy.ndarray: Array of daily returns
    """
    if data.empty:
        print("Warning: Empty data provided to calculate_returns")
        return np.array([])
    
    # Ensure price_col exists in the dataframe
    if price_col not in data.columns:
        print(f"Warning: Column '{price_col}' not found in data. Available columns: {data.columns.tolist()}")
        return np.array([])
    
    try:
        # Ensure prices are numeric
        prices = pd.to_numeric(data[price_col], errors='coerce')
        
        # Drop NaN values
        prices = prices.dropna()
        
        if len(prices) < 2:
            print("Warning: Not enough valid price data points to calculate returns")
            return np.array([])
        
        # Calculate daily returns (as percentage change) with fill_method=None to address the warning
        returns = prices.pct_change(fill_method=None).dropna().values
        
        # Remove extreme outliers (optional, adjust threshold as needed)
        # Calculate 99th percentile as threshold
        threshold = np.percentile(np.abs(returns), 99.9)
        returns = returns[np.abs(returns) < threshold]
        
        return returns
        
    except Exception as e:
        print(f"Error calculating returns: {e}")
        return np.array([])

def calculate_log_returns(data, price_col='Close'):
    """
    Calculate daily log returns from price data.
    
    Args:
        data (pandas.DataFrame): Historical price data
        price_col (str): Column name for price data (default: 'Close')
        
    Returns:
        numpy.ndarray: Array of daily log returns
    """
    if data.empty:
        print("Warning: Empty data provided to calculate_log_returns")
        return np.array([])
    
    # Ensure price_col exists in the dataframe
    if price_col not in data.columns:
        print(f"Warning: Column '{price_col}' not found in data. Available columns: {data.columns.tolist()}")
        return np.array([])
    
    try:
        # Ensure prices are numeric
        prices = pd.to_numeric(data[price_col], errors='coerce')
        
        # Drop NaN values
        prices = prices.dropna()
        
        if len(prices) < 2:
            print("Warning: Not enough valid price data points to calculate log returns")
            return np.array([])
        
        # Calculate log returns: ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
        log_returns = np.log(prices / prices.shift(1)).dropna().values
        
        # Remove extreme outliers (optional, adjust threshold as needed)
        threshold = np.percentile(np.abs(log_returns), 99.9)
        log_returns = log_returns[np.abs(log_returns) < threshold]
        
        return log_returns
        
    except Exception as e:
        print(f"Error calculating log returns: {e}")
        return np.array([])

def generate_pdf_cdf(returns, num_bins=100):
    """
    Generate probability density function (PDF) and cumulative distribution function (CDF)
    from returns data.
    
    Args:
        returns (numpy.ndarray): Array of returns
        num_bins (int): Number of bins for histogram (default: 100)
        
    Returns:
        tuple: (pdf, cdf, bin_edges)
            - pdf (numpy.ndarray): Probability density function values
            - cdf (numpy.ndarray): Cumulative distribution function values
            - bin_edges (numpy.ndarray): Bin edges used for histogram
    """
    if len(returns) == 0:
        # Return empty arrays if no data
        print("Warning: No returns data provided to generate PDF/CDF")
        return np.array([]), np.array([]), np.array([])
    
    try:
        # Create histogram
        hist, bin_edges = np.histogram(returns, bins=num_bins, density=True)
        
        # Calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate PDF (probability density function)
        pdf = hist
        
        # Calculate width of each bin for proper area calculation
        bin_widths = np.diff(bin_edges)
        
        # Calculate CDF (cumulative distribution function) with proper normalization
        # Multiply hist by bin_widths to get the area/probability of each bin
        area_per_bin = hist * bin_widths
        
        # Normalize areas to ensure total probability is 1.0
        if np.sum(area_per_bin) > 0:
            area_per_bin = area_per_bin / np.sum(area_per_bin)
        
        # Calculate CDF by cumulative sum of normalized areas
        cdf = np.cumsum(area_per_bin)
        
        # Ensure proper normalization: first value should be bin probability (not 0)
        # and last value should be 1.0
        if cdf[-1] != 1.0:
            cdf = cdf / cdf[-1]
        
        # Insert 0 at the beginning to represent CDF at negative infinity
        # This ensures that when we sample using searchsorted, we have proper bounds
        cdf = np.insert(cdf, 0, 0.0)
        
        # We need to adjust bin_edges to match the new CDF length
        # The bin_edges array is already one longer than hist or pdf,
        # so we don't need to modify it
        
        return pdf, cdf, bin_edges
        
    except Exception as e:
        print(f"Error generating PDF/CDF: {e}")
        # Return empty arrays in case of error
        return np.array([]), np.array([]), np.array([])

def calculate_log_return_metrics(log_returns):
    """
    Calculate key statistical metrics for log returns.
    
    Args:
        log_returns (numpy.ndarray): Array of log returns
        
    Returns:
        dict: Dictionary of key metrics
    """
    if len(log_returns) == 0:
        return {
            "daily_avg": 0,
            "annual_avg": 0,
            "volatility_daily": 0,
            "volatility_annual": 0,
            "sharpe_ratio": 0
        }
    
    # Calculate key metrics
    daily_avg = np.mean(log_returns)
    daily_var = np.var(log_returns)
    daily_std = np.std(log_returns)
    
    # Annualized metrics (assuming 252 trading days)
    annual_avg = daily_avg * 252
    annual_var = daily_var * 252
    annual_std = daily_std * np.sqrt(252)
    
    # Sharpe ratio (assuming risk-free rate = 0)
    sharpe_ratio = annual_avg / annual_std if annual_std > 0 else 0
    
    # Expected return after n days
    expected_return_formula = f"exp({daily_avg} * days)"
    expected_annual_return = np.exp(annual_avg)
    
    return {
        "daily_avg": daily_avg,
        "annual_avg": annual_avg,
        "volatility_daily": daily_std,
        "volatility_annual": annual_std,
        "sharpe_ratio": sharpe_ratio,
        "expected_return_formula": expected_return_formula,
        "expected_annual_return": expected_annual_return - 1  # Convert to percentage
    }

def sample_from_cdf(cdf, bin_edges, num_samples=1):
    """
    Generate samples from a CDF using inverse transform sampling.
    
    Args:
        cdf (numpy.ndarray): Cumulative distribution function values
        bin_edges (numpy.ndarray): Bin edges used for histogram
        num_samples (int): Number of samples to generate (default: 1)
        
    Returns:
        numpy.ndarray: Generated samples
    """
    if len(cdf) == 0 or len(bin_edges) == 0:
        print("Warning: Empty CDF or bin_edges provided to sample_from_cdf")
        return np.zeros(num_samples)
    
    # Check for NaN values in CDF and bin_edges
    if np.isnan(cdf).any() or np.isnan(bin_edges).any():
        print("Warning: NaN values in CDF or bin_edges detected in sample_from_cdf")
        cdf = np.nan_to_num(cdf, nan=0.0)
        bin_edges = np.nan_to_num(bin_edges, nan=0.0)
    
    # Ensure CDF is strictly monotonically increasing (important for searchsorted)
    # and properly bounded between 0 and 1
    if len(cdf) > 0:
        is_invalid = False
        
        # Check if not properly normalized
        if cdf[0] != 0 or cdf[-1] != 1 or np.any(np.diff(cdf) < 0):
            is_invalid = True
        
        # Handle case where CDF needs normalization
        if is_invalid:
            print("Warning: CDF not properly normalized, adjusting...")
            
            # First, ensure strict monotonicity using cumulative maximum
            # (this prevents decreasing values)
            sorted_cdf = np.maximum.accumulate(cdf)
            
            # Then normalize to [0,1] range
            if sorted_cdf[-1] > sorted_cdf[0]:
                normalized_cdf = (sorted_cdf - sorted_cdf[0]) / (sorted_cdf[-1] - sorted_cdf[0])
            else:
                # If all values are the same, create linear spacing
                normalized_cdf = np.linspace(0, 1, len(cdf))
            
            # Ensure first and last elements are exactly 0 and 1
            normalized_cdf[0] = 0.0
            normalized_cdf[-1] = 1.0
            
            cdf = normalized_cdf
    
    try:
        # Generate uniform random numbers between 0 and 1
        u = np.random.random(num_samples)
        
        # Inverse transform sampling - find where the random values fall in the CDF
        indices = np.searchsorted(cdf, u) - 1
        
        # Ensure indices are within valid range
        indices = np.clip(indices, 0, len(bin_edges) - 2)
        
        # Sample from within each bin using linear interpolation for more accuracy
        # First, get the lower and upper CDF values for each sampled point
        cdf_low = np.take(cdf, indices)
        cdf_high = np.take(cdf, indices + 1)
        
        # Prevent division by zero
        cdf_diff = np.maximum(cdf_high - cdf_low, 1e-10)
        
        # Calculate interpolation factor
        alpha = (u - cdf_low) / cdf_diff
        
        # Get bin edges for each sample
        bin_low = np.take(bin_edges, indices)
        bin_high = np.take(bin_edges, indices + 1)
        
        # Linear interpolation within the bin
        samples = bin_low + alpha * (bin_high - bin_low)
        
        # Check for NaN values in the samples
        if np.isnan(samples).any():
            print("Warning: NaN values found in samples from sample_from_cdf")
            samples = np.nan_to_num(samples, nan=0.0)
        
        return samples
        
    except Exception as e:
        print(f"Error sampling from CDF: {e}")
        return np.zeros(num_samples) 