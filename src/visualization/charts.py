import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

def plot_pdf_distributions(datasets, bin_edges, num_bins=100):
    """
    Plot the PDF distributions for multiple datasets.

    Args:
        # historical_returns (numpy.ndarray): Historical returns
        # random_returns (numpy.ndarray): Single simulation returns
        # validation_samples (numpy.ndarray): 10,000 samples for validation
        datasets (list): A list of tuples, where each tuple contains
                         (returns_array, name, color, dash_style).
                         `dash_style` can be None, 'dash', 'dot', 'dashdot'.
        bin_edges (numpy.ndarray): Bin edges for histogram
        num_bins (int): Number of bins for histograms (default: 100)

    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create figure
    fig = go.Figure()

    # Calculate bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find overall min/max for axis range calculation (using first dataset as reference)
    if datasets:
        reference_returns = datasets[0][0] # Use first dataset's returns for range
        q1, q99 = np.percentile(reference_returns, [1, 99])
        margin = (q99 - q1) * 0.5
        x_range = [q1 - margin, q99 + margin]
    else:
        x_range = None # Default range if no data

    # Plot PDF for each dataset
    for returns, name, color, dash_style in datasets:
        if returns is not None and len(returns) > 0:
            hist_data, _ = np.histogram(returns, bins=bin_edges, density=True)
            fig.add_trace(
                go.Scatter(x=bin_centers, y=hist_data, mode='lines', name=name,
                         line=dict(color=color, width=2, dash=dash_style))
            )

    # # Calculate histogram data for historical returns
    # hist_hist, _ = np.histogram(historical_returns, bins=num_bins, density=True)
    #
    # # Calculate histogram data for validation samples (10,000)
    # hist_val, _ = np.histogram(validation_samples, bins=bin_edges, density=True)
    #
    # # Calculate histogram data for single simulation
    # hist_rand, _ = np.histogram(random_returns, bins=bin_edges, density=True)
    #
    # # PDF plot
    # fig.add_trace(
    #     go.Scatter(x=bin_centers, y=hist_hist, mode='lines', name='Historical PDF',
    #              line=dict(color='blue', width=2))
    # )
    #
    # fig.add_trace(
    #     go.Scatter(x=bin_centers, y=hist_val, mode='lines', name='Validation Samples PDF (10,000)',
    #              line=dict(color='green', width=2))
    # )
    #
    # fig.add_trace(
    #     go.Scatter(x=bin_centers, y=hist_rand, mode='lines', name='Current Simulation PDF',
    #              line=dict(color='red', width=2))
    # )

    # Update layout
    fig.update_layout(
        height=400,
        title="Return Probability Density Functions (PDF)",
        xaxis_title="Return",
        yaxis_title="Probability Density",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_dark"
    )

    # Set range to focus on the main distribution
    if x_range:
        fig.update_xaxes(range=x_range)

    return fig

def plot_cdf_distributions(datasets, bin_edges, num_bins=100):
    """
    Plot the CDF distributions for multiple datasets.

    Args:
        # historical_returns (numpy.ndarray): Historical returns
        # random_returns (numpy.ndarray): Single simulation returns
        # validation_samples (numpy.ndarray): 10,000 samples for validation
        datasets (list): A list of tuples, where each tuple contains
                         (returns_array, name, color, dash_style).
                         `dash_style` can be None, 'dash', 'dot', 'dashdot'.
        bin_edges (numpy.ndarray): Bin edges for histogram
        num_bins (int): Number of bins for histograms (default: 100)

    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create figure
    fig = go.Figure()

    # Calculate bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0] # Assuming uniform bins

    # Find overall min/max for axis range calculation (using first dataset as reference)
    if datasets:
        reference_returns = datasets[0][0] # Use first dataset's returns for range
        q1, q99 = np.percentile(reference_returns, [1, 99])
        margin = (q99 - q1) * 0.5
        x_range = [q1 - margin, q99 + margin]
    else:
        x_range = None # Default range if no data

    # Plot CDF for each dataset
    for returns, name, color, dash_style in datasets:
        if returns is not None and len(returns) > 0:
            hist_data, _ = np.histogram(returns, bins=bin_edges, density=True)
            cdf_data = np.cumsum(hist_data) * bin_width
            # Normalize CDF to end at 1.0
            if cdf_data[-1] > 1e-6: # Avoid division by zero or near-zero
                 cdf_data = cdf_data / cdf_data[-1]
            else:
                 cdf_data = np.zeros_like(cdf_data) # Set to zero if sum is too small

            fig.add_trace(
                go.Scatter(x=bin_centers, y=cdf_data, mode='lines', name=name,
                         line=dict(color=color, width=2, dash=dash_style))
            )

    # # Calculate histogram data for historical returns
    # hist_hist, _ = np.histogram(historical_returns, bins=num_bins, density=True)
    # hist_cdf = np.cumsum(hist_hist) * (bin_edges[1] - bin_edges[0])
    #
    # # Calculate histogram data for validation samples (10,000)
    # hist_val, _ = np.histogram(validation_samples, bins=bin_edges, density=True)
    # val_cdf = np.cumsum(hist_val) * (bin_edges[1] - bin_edges[0])
    #
    # # Calculate histogram data for single simulation
    # hist_rand, _ = np.histogram(random_returns, bins=bin_edges, density=True)
    # rand_cdf = np.cumsum(hist_rand) * (bin_edges[1] - bin_edges[0])
    #
    # # Normalize CDFs to end at 1.0
    # if hist_cdf[-1] != 1.0:
    #     hist_cdf = hist_cdf / hist_cdf[-1]
    # if val_cdf[-1] != 1.0:
    #     val_cdf = val_cdf / val_cdf[-1]
    # if rand_cdf[-1] != 1.0:
    #     rand_cdf = rand_cdf / rand_cdf[-1]
    #
    # # CDF plot
    # fig.add_trace(
    #     go.Scatter(x=bin_centers, y=hist_cdf, mode='lines', name='Historical CDF',
    #              line=dict(color='blue', width=2))
    # )
    #
    # fig.add_trace(
    #     go.Scatter(x=bin_centers, y=val_cdf, mode='lines', name='Validation Samples CDF (10,000)',
    #              line=dict(color='green', width=2))
    # )
    #
    # fig.add_trace(
    #     go.Scatter(x=bin_centers, y=rand_cdf, mode='lines', name='Current Simulation CDF',
    #              line=dict(color='red', width=2))
    # )

    # Update layout
    fig.update_layout(
        height=400,
        title="Cumulative Distribution Functions (CDF)",
        xaxis_title="Return",
        yaxis_title="Cumulative Probability",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_dark"
    )

    # Set range to focus on the main distribution
    if x_range:
        fig.update_xaxes(range=x_range)

    return fig

def plot_log_returns_pdf(historical_log_returns, random_log_returns, validation_log_samples, bin_edges=None, num_bins=100):
    """
    Plot the PDF distributions of historical log returns vs. generated log returns.
    
    Args:
        historical_log_returns (numpy.ndarray): Historical log returns
        random_log_returns (numpy.ndarray): Single simulation log returns
        validation_log_samples (numpy.ndarray): Validation samples for log returns
        bin_edges (numpy.ndarray, optional): Bin edges for histogram
        num_bins (int): Number of bins for histograms (default: 100)
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Generate bin edges if not provided
    if bin_edges is None:
        min_val = min(np.min(historical_log_returns), np.min(random_log_returns), np.min(validation_log_samples))
        max_val = max(np.max(historical_log_returns), np.max(random_log_returns), np.max(validation_log_samples))
        margin = (max_val - min_val) * 0.1
        bin_edges = np.linspace(min_val - margin, max_val + margin, num_bins + 1)
    
    # Calculate bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate histogram data for historical log returns
    hist_hist, _ = np.histogram(historical_log_returns, bins=bin_edges, density=True)
    
    # Calculate histogram data for validation samples
    hist_val, _ = np.histogram(validation_log_samples, bins=bin_edges, density=True)
    
    # Calculate histogram data for single simulation
    hist_rand, _ = np.histogram(random_log_returns, bins=bin_edges, density=True)
    
    # PDF plot
    fig.add_trace(
        go.Scatter(x=bin_centers, y=hist_hist, mode='lines', name='Historical PDF',
                 line=dict(color='blue', width=2))
    )
    
    fig.add_trace(
        go.Scatter(x=bin_centers, y=hist_val, mode='lines', name='Validation Samples PDF',
                 line=dict(color='green', width=2))
    )
    
    fig.add_trace(
        go.Scatter(x=bin_centers, y=hist_rand, mode='lines', name='Current Simulation PDF',
                 line=dict(color='red', width=2))
    )
    
    # Add vertical line at the mean of historical log returns
    mean_log_return = np.mean(historical_log_returns)
    fig.add_vline(x=mean_log_return, line_dash="dash", line_color="white",
                 annotation_text=f"Mean: {mean_log_return:.6f}")
    
    # Update layout
    fig.update_layout(
        height=400,
        title="Log Return Probability Density Functions (PDF)",
        xaxis_title="Log Return (ln(1+r))",
        yaxis_title="Probability Density",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_dark"
    )
    
    # Set range to focus on the main distribution (exclude extreme outliers)
    q1, q99 = np.percentile(historical_log_returns, [1, 99])
    margin = (q99 - q1) * 0.5
    fig.update_xaxes(range=[q1-margin, q99+margin])
    
    return fig

def plot_log_returns_cdf(historical_log_returns, random_log_returns, validation_log_samples, bin_edges=None, num_bins=100):
    """
    Plot the CDF distributions of historical log returns vs. generated log returns.
    
    Args:
        historical_log_returns (numpy.ndarray): Historical log returns
        random_log_returns (numpy.ndarray): Single simulation log returns
        validation_log_samples (numpy.ndarray): Validation samples for log returns
        bin_edges (numpy.ndarray, optional): Bin edges for histogram
        num_bins (int): Number of bins for histograms (default: 100)
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Generate bin edges if not provided
    if bin_edges is None:
        min_val = min(np.min(historical_log_returns), np.min(random_log_returns), np.min(validation_log_samples))
        max_val = max(np.max(historical_log_returns), np.max(random_log_returns), np.max(validation_log_samples))
        margin = (max_val - min_val) * 0.1
        bin_edges = np.linspace(min_val - margin, max_val + margin, num_bins + 1)
    
    # Calculate bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate histogram data for historical log returns
    hist_hist, _ = np.histogram(historical_log_returns, bins=bin_edges, density=True)
    hist_cdf = np.cumsum(hist_hist) * (bin_edges[1] - bin_edges[0])
    
    # Calculate histogram data for validation samples
    hist_val, _ = np.histogram(validation_log_samples, bins=bin_edges, density=True)
    val_cdf = np.cumsum(hist_val) * (bin_edges[1] - bin_edges[0])
    
    # Calculate histogram data for single simulation
    hist_rand, _ = np.histogram(random_log_returns, bins=bin_edges, density=True)
    rand_cdf = np.cumsum(hist_rand) * (bin_edges[1] - bin_edges[0])
    
    # Normalize CDFs to end at 1.0
    if hist_cdf[-1] != 1.0:
        hist_cdf = hist_cdf / hist_cdf[-1]
    if val_cdf[-1] != 1.0:
        val_cdf = val_cdf / val_cdf[-1]
    if rand_cdf[-1] != 1.0:
        rand_cdf = rand_cdf / rand_cdf[-1]
    
    # CDF plot
    fig.add_trace(
        go.Scatter(x=bin_centers, y=hist_cdf, mode='lines', name='Historical CDF',
                 line=dict(color='blue', width=2))
    )
    
    fig.add_trace(
        go.Scatter(x=bin_centers, y=val_cdf, mode='lines', name='Validation Samples CDF',
                 line=dict(color='green', width=2))
    )
    
    fig.add_trace(
        go.Scatter(x=bin_centers, y=rand_cdf, mode='lines', name='Current Simulation CDF',
                 line=dict(color='red', width=2))
    )
    
    # Add vertical line at the mean of historical log returns
    mean_log_return = np.mean(historical_log_returns)
    fig.add_vline(x=mean_log_return, line_dash="dash", line_color="white",
                 annotation_text=f"Mean: {mean_log_return:.6f}")
    
    # Update layout
    fig.update_layout(
        height=400,
        title="Log Return Cumulative Distribution Functions (CDF)",
        xaxis_title="Log Return (ln(1+r))",
        yaxis_title="Cumulative Probability",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_dark"
    )
    
    # Set range to focus on the main distribution (exclude extreme outliers)
    q1, q99 = np.percentile(historical_log_returns, [1, 99])
    margin = (q99 - q1) * 0.5
    fig.update_xaxes(range=[q1-margin, q99+margin])
    
    return fig

def plot_distributions(historical_returns, random_returns, validation_samples, bin_edges, num_bins=100):
    """
    Legacy function - now just calls the separate PDF and CDF plots.
    Kept for backwards compatibility.
    """
    pdf_fig = plot_pdf_distributions(historical_returns, random_returns, validation_samples, bin_edges, num_bins)
    cdf_fig = plot_cdf_distributions(historical_returns, random_returns, validation_samples, bin_edges, num_bins)
    
    return pdf_fig, cdf_fig

def plot_price_paths(historical_path, random_path):
    """
    Plot historical and simulated price paths with cycle numbers.
    
    Args:
        historical_path (numpy.ndarray): Historical price path
        random_path (numpy.ndarray): Simulated price path
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Determine the maximum number of days to plot (based on the shorter of the two paths)
    max_days = min(len(historical_path), len(random_path))
    x_axis = list(range(max_days))
    
    # Add historical path
    fig.add_trace(
        go.Scatter(x=x_axis, y=historical_path[:max_days], mode='lines', name='Historical ETH Path',
                 line=dict(color='blue', width=2))
    )
    
    # Add random path
    fig.add_trace(
        go.Scatter(x=x_axis, y=random_path[:max_days], mode='lines', name='Simulated ETH Path',
                 line=dict(color='red', width=2))
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        xaxis_title="Cycle (Day)",
        yaxis_title="Normalized Price (Starting at 1.0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_dark"  # Using dark template to match screenshot
    )
    
    return fig

def plot_historical_timeline(eth_data, max_days=None):
    """
    Plot historical ETH price with actual dates.
    
    Args:
        eth_data (pandas.DataFrame): Historical ETH price data with DatetimeIndex
        max_days (int, optional): Maximum number of days to display
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Ensure data is valid
    if eth_data is None or eth_data.empty or 'Close' not in eth_data.columns:
        return fig  # Return empty figure if data is invalid
    
    # Convert to list for plotting to avoid any pandas-related issues
    dates = list(eth_data.index)
    prices = list(eth_data['Close'].values)
    
    # Limit to max_days if specified
    if max_days is not None and max_days > 0 and max_days < len(dates):
        dates = dates[-max_days:]
        prices = prices[-max_days:]
    
    # Force y-axis to start from 0 to better visualize the price
    y_min = 0
    y_max = max(prices) * 1.1  # Add 10% margin
    
    # Add historical price line
    fig.add_trace(
        go.Scatter(
            x=dates, 
            y=prices, 
            mode='lines', 
            name='Historical ETH Price',
            line=dict(color='blue', width=2)
        )
    )
    
    # Update layout
    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="ETH Price (USD)",
        title="Historical ETH Price Timeline",
        template="plotly_dark",  # Using dark template to match screenshot
        yaxis=dict(range=[y_min, y_max]),  # Force y-axis range
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add hover data for better interaction
    fig.update_traces(
        hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
    )
    
    return fig

def plot_monte_carlo_paths(paths, historical_path=None, max_display_paths=500):
    """
    Plot multiple simulated price paths from Monte Carlo simulation.
    
    Args:
        paths (numpy.ndarray): Array of price paths, shape (num_paths, num_days)
        historical_path (numpy.ndarray, optional): Historical price path to overlay
        max_display_paths (int): Maximum number of paths to display (for performance)
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Sample paths if there are too many
    num_paths = paths.shape[0]
    display_paths = min(num_paths, max_display_paths)
    
    if num_paths > display_paths:
        # Randomly sample paths to display
        indices = np.random.choice(num_paths, display_paths, replace=False)
        display_paths_data = paths[indices]
    else:
        display_paths_data = paths
    
    # X-axis values (days)
    num_days = paths.shape[1]
    x_axis = list(range(num_days))
    
    # Calculate percentiles for highlighting
    if num_paths >= 100:
        median_path = np.median(paths, axis=0)
        p5_path = np.percentile(paths, 5, axis=0)
        p95_path = np.percentile(paths, 95, axis=0)
    
    # Plot individual paths with transparency
    for i in range(display_paths):
        fig.add_trace(
            go.Scatter(
                x=x_axis, 
                y=display_paths_data[i], 
                mode='lines', 
                line=dict(color='rgba(100,100,255,0.05)'),
                showlegend=False,
                hoverinfo='skip'
            )
        )
    
    # Add percentile lines if we have enough paths
    if num_paths >= 100:
        # Add median path
        fig.add_trace(
            go.Scatter(
                x=x_axis, 
                y=median_path, 
                mode='lines', 
                name='Median Path',
                line=dict(color='white', width=2, dash='dash')
            )
        )
        
        # Add 5th percentile path
        fig.add_trace(
            go.Scatter(
                x=x_axis, 
                y=p5_path, 
                mode='lines', 
                name='5th Percentile',
                line=dict(color='red', width=2, dash='dot')
            )
        )
        
        # Add 95th percentile path
        fig.add_trace(
            go.Scatter(
                x=x_axis, 
                y=p95_path, 
                mode='lines', 
                name='95th Percentile',
                line=dict(color='green', width=2, dash='dot')
            )
        )
    
    # Add historical path if provided
    if historical_path is not None and len(historical_path) > 0:
        # Ensure historical path length matches simulation
        hist_len = min(len(historical_path), num_days)
        
        # Add historical path with a distinctive color
        fig.add_trace(
            go.Scatter(
                x=x_axis[:hist_len], 
                y=historical_path[:hist_len], 
                mode='lines', 
                name='Historical ETH Path',
                line=dict(color='orange', width=3)
            )
        )
    
    # Update layout
    fig.update_layout(
        height=500,
        title=f"Monte Carlo Simulation ({num_paths:,} paths, {display_paths:,} shown)",
        xaxis_title="Cycle (Day)",
        yaxis_title="Normalized Price (Starting at 1.0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_dark"
    )
    
    return fig

def plot_monte_carlo_returns_distribution(final_returns, historical_return=None, num_bins=100):
    """
    Plot the distribution of final returns from Monte Carlo simulation.
    
    Args:
        final_returns (numpy.ndarray): Array of final returns for each path
        historical_return (float): Actual historical return for comparison
        num_bins (int): Number of bins for histogram
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Calculate histogram
    hist, bin_edges = np.histogram(final_returns, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Add histogram as line
    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=hist,
            mode='lines',
            name='Monte Carlo Returns PDF',
            line=dict(color='blue', width=2)
        )
    )
    
    # Calculate statistics
    mean_return = np.mean(final_returns)
    median_return = np.median(final_returns)
    std_return = np.std(final_returns)
    p5_return = np.percentile(final_returns, 5)
    p95_return = np.percentile(final_returns, 95)
    
    # Add mean line
    fig.add_trace(
        go.Scatter(
            x=[mean_return, mean_return],
            y=[0, max(hist) * 1.1],
            mode='lines',
            name=f'Mean Return: {mean_return:.2f}x',
            line=dict(color='green', width=2, dash='dash')
        )
    )
    
    # Add historical return line if provided
    if historical_return is not None:
        fig.add_trace(
            go.Scatter(
                x=[historical_return, historical_return],
                y=[0, max(hist) * 1.1],
                mode='lines',
                name=f'Historical Return: {historical_return:.2f}x',
                line=dict(color='orange', width=3)
            )
        )
    
    # Add annotations for statistics
    annotations = [
        f"Mean: {mean_return:.2f}x",
        f"Median: {median_return:.2f}x",
        f"Std Dev: {std_return:.2f}",
        f"5th Percentile: {p5_return:.2f}x",
        f"95th Percentile: {p95_return:.2f}x"
    ]
    
    # Update layout
    fig.update_layout(
        height=500,
        title="Distribution of Monte Carlo Final Returns",
        xaxis_title="Final Return Multiple (x)",
        yaxis_title="Probability Density",
        annotations=[
            dict(
                x=0.01,
                y=0.99,
                xref="paper",
                yref="paper",
                text="<br>".join(annotations),
                showarrow=False,
                align="left",
                bgcolor="rgba(50, 50, 50, 0.8)",
                bordercolor="white",
                borderwidth=1,
                font=dict(size=12)
            )
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_dark"
    )
    
    return fig

def plot_lp_payoff_curve(price_ratios, payoffs, fee_percentages, use_dynamic_fee=False, use_log_scale=False):
    """
    Plot the LP payoff curve comparing various fee percentages against ETH price movements.
    
    Args:
        price_ratios (np.ndarray): Array of price ratios to plot on x-axis
        payoffs (list): List of arrays, each containing LP position values for a fee percentage
        fee_percentages (list): List of fee percentages corresponding to each payoff array
        use_dynamic_fee (bool): Whether dynamic fee (equal to IL) is being used
        use_log_scale (bool): Whether to use logarithmic scale for the x-axis
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Import here to avoid circular imports
    import plotly.graph_objects as go
    import numpy as np
    
    # Make sure inputs are numpy arrays
    price_ratios = np.array(price_ratios)
    
    # Calculate the theoretical LP value without fees
    # Value normalized to initial investment = sqrt(price_ratio)
    # Ensure it's never negative (sqrt handles this)
    theoretical_lp = np.sqrt(price_ratios)
    theoretical_lp = np.maximum(theoretical_lp, 0.0) # Redundant but safe
    
    # Create figure
    fig = go.Figure()
    
    # Colors for different curves - Simplified as we only plot one payoff curve now
    # colors = ['green', 'red', 'purple', 'orange', 'pink', 'cyan']
    lp_color = 'red' # LP Value is always red now
    
    # Add Buy & Hold ETH curve
    fig.add_trace(
        go.Scatter(
            x=price_ratios, 
            y=price_ratios, 
            mode='lines', 
            name='Buy and Hold ETH',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add theoretical LP value (without fees)
    fig.add_trace(
        go.Scatter(
            x=price_ratios, 
            y=theoretical_lp, 
            mode='lines', 
            name='Theoretical LP Value (No Fees)',
            line=dict(color='red', width=2, dash='dot') # Changed color to red
        )
    )
    
    # Calculate impermanent loss (buy_and_hold - LP)
    # This is a positive value that forms a U-shaped parabola, minimized at 0 when price_ratio = 1
    il_values = price_ratios - theoretical_lp
    il_values = np.maximum(il_values, 0.0)  # Ensure non-negative
    
    # Add impermanent loss curve
    fig.add_trace(
        go.Scatter(
            x=price_ratios, 
            y=il_values, 
            mode='lines', 
            name='Impermanent Loss',
            line=dict(color='magenta', width=2, dash='dashdot') # Changed color to magenta
        )
    )
    
    # Add LP payoff curves for each fee percentage
    for i, (payoff, fee) in enumerate(zip(payoffs, fee_percentages)):
        # color = colors[i % len(colors)] if i > 0 or len(payoffs) == 1 else 'green'  # Use green for first fee (common case)
        color = lp_color # Always use red for the LP value curve
        if use_dynamic_fee:
            curve_name = f'LP Value (Dynamic Fee = IL)'
        else:
            curve_name = f'LP Value ({fee*100:.1f}% fee)'
            
        fig.add_trace(
            go.Scatter(
                x=price_ratios, 
                y=payoff, 
                mode='lines', 
                name=curve_name,
                line=dict(color=color, width=2)
            )
        )
    
    # Add a horizontal line at y=1.0 (break-even)
    fig.add_shape(
        type="line",
        x0=price_ratios[0],
        y0=1.0,
        x1=price_ratios[-1],
        y1=1.0,
        line=dict(color="white", width=1, dash="dash"),
    )
    
    # Add a vertical line at x=1.0 (no price change)
    fig.add_shape(
        type="line",
        x0=1.0,
        y0=fig.layout.yaxis.range[0] if fig.layout.yaxis.range else 0,
        x1=1.0,
        y1=fig.layout.yaxis.range[1] if fig.layout.yaxis.range else 2,
        line=dict(color="white", width=1, dash="dash"),
    )
    
    # Add annotations to explain the formulas
    annotations = [
        dict(
            x=price_ratios[-1] * 0.9,
            y=price_ratios[-1] * 0.9,
            xref="x",
            yref="y",
            text="Buy & Hold: P₁/P₀",
            showarrow=True,
            arrowhead=3,
            ax=-40,
            ay=-40,
            font=dict(size=12)
        ),
        dict(
            x=price_ratios[-1] * 0.8,
            y=theoretical_lp[-1] * 0.8,
            xref="x",
            yref="y",
            text="LP Value (norm): √(P₁/P₀)",
            showarrow=True,
            arrowhead=3,
            ax=-40,
            ay=40,
            font=dict(size=12)
        )
    ]
    
    # Add additional annotation for dynamic fee model
    if use_dynamic_fee:
        annotations.append(
            dict(
                x=price_ratios[-1] * 0.7,
                y=payoffs[0][-1] * 0.9,
                xref="x",
                yref="y",
                text="Dynamic Fee = IL × Fee Dial",
                showarrow=True,
                arrowhead=3,
                ax=-40,
                ay=0,
                font=dict(size=12, color=color)
            )
        )
        
        # Add an annotation to explain how the dynamic fee works
        annotations.append(
            dict(
                x=1.5,
                y=1.5,
                xref="x",
                yref="y",
                text="IL = Buy & Hold - LP (without fees)\nFee = IL × Fee Dial setting",
                showarrow=False,
                font=dict(size=12, color="white")
            )
        )
    
    # Update layout
    fig.update_layout(
        title="Uniswap LP Payoff vs ETH Price Change",
        xaxis_title="ETH Price Ratio (Current/Initial)",
        yaxis_title="Payoff Multiple (Relative to Initial Investment)",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        template="plotly_dark",
        annotations=annotations
    )
    
    # Set range to focus on the main area of interest
    if use_log_scale:
        fig.update_xaxes(type='log', range=[np.log10(price_ratios[0]), np.log10(3)])
    else:
        fig.update_xaxes(range=[0, 3])
    fig.update_yaxes(range=[0, 3.5])
    
    return fig

def plot_price_paths_with_lp(historical_path, random_path, historical_lp_path, random_lp_path, fee_percentage=0.003, use_dynamic_fee=False):
    """
    Plot historical and simulated price paths with their corresponding LP paths.
    
    Args:
        historical_path (numpy.ndarray): Historical price path
        random_path (numpy.ndarray): Simulated price path
        historical_lp_path (numpy.ndarray): Historical LP path
        random_lp_path (numpy.ndarray): Simulated LP path
        fee_percentage (float): Fee percentage used for LP calculation
        use_dynamic_fee (bool): Whether dynamic fee (equal to IL) is being used
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Determine the maximum number of days to plot (based on the shortest path)
    max_days = min(len(historical_path), len(random_path), len(historical_lp_path), len(random_lp_path))
    x_axis = list(range(max_days))
    
    # Add historical price path
    fig.add_trace(
        go.Scatter(
            x=x_axis, 
            y=historical_path[:max_days], 
            mode='lines', 
            name='Historical ETH',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add historical LP path (same color as historical ETH but dashed)
    if use_dynamic_fee:
        hist_lp_name = f'Historical LP (Dynamic Fee)'
    else:
        hist_lp_name = f'Historical LP ({fee_percentage*100:.1f}% fee)'
        
    fig.add_trace(
        go.Scatter(
            x=x_axis, 
            y=historical_lp_path[:max_days], 
            mode='lines', 
            name=hist_lp_name,
            line=dict(color='blue', width=2, dash='dash')
        )
    )
    
    # Add simulated price path
    fig.add_trace(
        go.Scatter(
            x=x_axis, 
            y=random_path[:max_days], 
            mode='lines', 
            name='Simulated ETH',
            line=dict(color='red', width=2)
        )
    )
    
    # Add simulated LP path (same color as simulated ETH but dashed)
    fig.add_trace(
        go.Scatter(
            x=x_axis, 
            y=random_lp_path[:max_days], 
            mode='lines', 
            name=f'Simulated LP ({fee_percentage*100:.1f}% fee)',
            line=dict(color='red', width=2, dash='dash')
        )
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        title="Price Paths: ETH vs LP Position",
        xaxis_title="Cycle (Day)",
        yaxis_title="Normalized Value (Starting at 1.0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_dark"
    )
    
    return fig

def plot_monte_carlo_paths_with_lp(price_paths, lp_paths, historical_price_path=None, historical_lp_path=None, 
                                  max_display_paths=200, fee_percentage=0.003, use_dynamic_fee=False):
    """
    Plot Monte Carlo simulated price paths and corresponding LP paths.
    
    Args:
        price_paths (numpy.ndarray): Array of price paths (shape: [num_paths, num_days])
        lp_paths (numpy.ndarray): Array of LP paths (shape: [num_paths, num_days])
        historical_price_path (numpy.ndarray, optional): Historical price path to overlay
        historical_lp_path (numpy.ndarray, optional): Historical LP path to overlay
        max_display_paths (int): Maximum number of paths to display (defaults to 200)
        fee_percentage (float): Fee percentage used for LP calculation
        use_dynamic_fee (bool): Whether dynamic fee (equal to IL) is being used
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Sample paths if there are too many
    num_paths = price_paths.shape[0]
    display_paths = min(num_paths, max_display_paths)
    
    if num_paths > display_paths:
        # Randomly sample paths to display
        indices = np.random.choice(num_paths, display_paths, replace=False)
        display_price_paths = price_paths[indices]
        display_lp_paths = lp_paths[indices]
    else:
        display_price_paths = price_paths
        display_lp_paths = lp_paths
    
    # X-axis values (days)
    num_days = price_paths.shape[1]
    x_axis = list(range(num_days))
    
    # Plot individual price paths with transparency
    for i in range(display_paths):
        fig.add_trace(
            go.Scatter(
                x=x_axis, 
                y=display_price_paths[i], 
                mode='lines', 
                line=dict(color='rgba(100,100,255,0.05)'),
                showlegend=False,
                hoverinfo='skip'
            )
        )
    
    # Plot individual LP paths with transparency
    for i in range(display_paths):
        fig.add_trace(
            go.Scatter(
                x=x_axis, 
                y=display_lp_paths[i], 
                mode='lines', 
                line=dict(color='rgba(255,100,100,0.05)'),
                showlegend=False,
                hoverinfo='skip'
            )
        )
    
    # Calculate percentiles for highlighting
    if num_paths >= 100:
        # For price paths
        median_price_path = np.median(price_paths, axis=0)
        p5_price_path = np.percentile(price_paths, 5, axis=0)
        p95_price_path = np.percentile(price_paths, 95, axis=0)
        
        # For LP paths
        median_lp_path = np.median(lp_paths, axis=0)
        p5_lp_path = np.percentile(lp_paths, 5, axis=0)
        p95_lp_path = np.percentile(lp_paths, 95, axis=0)
        
        # Add 5th percentile path for ETH
        fig.add_trace(
            go.Scatter(
                x=x_axis, 
                y=p5_price_path, 
                mode='lines', 
                name='5th Percentile ETH',
                line=dict(color='blue', width=1, dash='dot')
            )
        )
        
        # Add 95th percentile path for ETH
        fig.add_trace(
            go.Scatter(
                x=x_axis, 
                y=p95_price_path, 
                mode='lines', 
                name='95th Percentile ETH',
                line=dict(color='blue', width=1, dash='dot')
            )
        )
        
        # Add filled area between 5th and 95th percentile for ETH
        fig.add_trace(
            go.Scatter(
                x=x_axis+x_axis[::-1],
                y=list(p5_price_path)+list(p95_price_path[::-1]),
                fill='toself',
                fillcolor='rgba(100,100,255,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=False
            )
        )
        
        # Add 5th percentile path for LP
        fig.add_trace(
            go.Scatter(
                x=x_axis, 
                y=p5_lp_path, 
                mode='lines', 
                name='5th Percentile LP',
                line=dict(color='red', width=1, dash='dot')
            )
        )
        
        # Add 95th percentile path for LP
        fig.add_trace(
            go.Scatter(
                x=x_axis, 
                y=p95_lp_path, 
                mode='lines', 
                name='95th Percentile LP',
                line=dict(color='red', width=1, dash='dot')
            )
        )
        
        # Add filled area between 5th and 95th percentile for LP
        fig.add_trace(
            go.Scatter(
                x=x_axis+x_axis[::-1],
                y=list(p5_lp_path)+list(p95_lp_path[::-1]),
                fill='toself',
                fillcolor='rgba(255,100,100,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=False
            )
        )
        
        # Add median path for price
        fig.add_trace(
            go.Scatter(
                x=x_axis, 
                y=median_price_path, 
                mode='lines', 
                name='Median ETH Path',
                line=dict(color='blue', width=2, dash='dash')
            )
        )
        
        # Add median path for LP
        fig.add_trace(
            go.Scatter(
                x=x_axis, 
                y=median_lp_path, 
                mode='lines', 
                name=f'Median LP Path ({fee_percentage*100:.1f}% fee)',
                line=dict(color='red', width=2, dash='dash')
            )
        )
    
    # Add historical paths if provided
    if historical_price_path is not None and len(historical_price_path) > 0:
        # Ensure historical path length matches simulation
        hist_len = min(len(historical_price_path), num_days)
        
        # Add historical price path
        fig.add_trace(
            go.Scatter(
                x=x_axis[:hist_len], 
                y=historical_price_path[:hist_len], 
                mode='lines', 
                name='Historical ETH Path',
                line=dict(color='green', width=3)
            )
        )
    
    if historical_lp_path is not None and len(historical_lp_path) > 0:
        # Ensure historical LP path length matches simulation
        hist_len = min(len(historical_lp_path), num_days)
        
        # Add historical LP path
        fig.add_trace(
            go.Scatter(
                x=x_axis[:hist_len], 
                y=historical_lp_path[:hist_len], 
                mode='lines', 
                name=f'Historical LP Path ({fee_percentage*100:.1f}% fee)',
                line=dict(color='orange', width=3)
            )
        )
    
    # Labels and titles based on fee model
    if use_dynamic_fee:
        fee_label = "Dynamic Fee (Equal to IL)"
    else:
        fee_label = f"{fee_percentage*100:.1f}% Fee"
        
    title = f"Monte Carlo Simulation: ETH vs LP ({fee_label})"

    # Update layout
    fig.update_layout(
        height=600,
        title=title,
        xaxis_title="Cycle (Day)",
        yaxis_title="Normalized Value (Starting at 1.0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_dark"
    )
    
    return fig

def plot_monte_carlo_returns_distribution_with_lp(eth_final_returns, lp_final_returns, 
                                               historical_eth_return=None, historical_lp_return=None, 
                                               num_bins=100, fee_percentage=0.003, use_dynamic_fee=False):
    """
    Plot histogram distribution of Monte Carlo final returns for ETH and LP.
    
    Args:
        eth_final_returns (numpy.ndarray): Array of final ETH returns from Monte Carlo
        lp_final_returns (numpy.ndarray): Array of final LP returns from Monte Carlo
        historical_eth_return (float, optional): Historical ETH return to overlay
        historical_lp_return (float, optional): Historical LP return to overlay
        num_bins (int): Number of histogram bins
        fee_percentage (float): Fee percentage used for LP calculation
        use_dynamic_fee (bool): Whether dynamic fee (equal to IL) is being used
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Filter out extreme outliers for better visualization
    # Calculate reasonable upper bounds for display (99th percentile × 1.5)
    eth_upper_bound = np.percentile(eth_final_returns, 99) * 1.5
    lp_upper_bound = np.percentile(lp_final_returns, 99) * 1.5
    
    # Filter returns for display purposes only
    eth_filtered = eth_final_returns[eth_final_returns <= eth_upper_bound]
    lp_filtered = lp_final_returns[lp_final_returns <= lp_upper_bound]
    
    # Print info about filtered values
    if len(eth_filtered) < len(eth_final_returns):
        print(f"Filtered {len(eth_final_returns) - len(eth_filtered)} extreme ETH returns > {eth_upper_bound:.2f}x for better visualization")
    
    if len(lp_filtered) < len(lp_final_returns):
        print(f"Filtered {len(lp_final_returns) - len(lp_filtered)} extreme LP returns > {lp_upper_bound:.2f}x for better visualization")
    
    # Calculate histograms with filtered data for better visualization
    eth_hist, eth_bin_edges = np.histogram(eth_filtered, bins=num_bins, density=True)
    lp_hist, lp_bin_edges = np.histogram(lp_filtered, bins=num_bins, density=True)
    
    eth_bin_centers = (eth_bin_edges[:-1] + eth_bin_edges[1:]) / 2
    lp_bin_centers = (lp_bin_edges[:-1] + lp_bin_edges[1:]) / 2
    
    # Add ETH returns histogram as line
    fig.add_trace(
        go.Scatter(
            x=eth_bin_centers,
            y=eth_hist,
            mode='lines',
            name='ETH Returns PDF',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add LP returns histogram as line
    fig.add_trace(
        go.Scatter(
            x=lp_bin_centers,
            y=lp_hist,
            mode='lines',
            name=f'LP Returns PDF ({fee_percentage*100:.1f}% fee)',
            line=dict(color='red', width=2)
        )
    )
    
    # Calculate statistics using all data (not just filtered data)
    eth_mean = np.mean(eth_final_returns)
    eth_median = np.median(eth_final_returns)
    eth_std = np.std(eth_final_returns)
    eth_p5 = np.percentile(eth_final_returns, 5)
    eth_p95 = np.percentile(eth_final_returns, 95)
    
    # Calculate statistics for LP returns
    lp_mean = np.mean(lp_final_returns)
    lp_median = np.median(lp_final_returns)
    lp_std = np.std(lp_final_returns)
    lp_p5 = np.percentile(lp_final_returns, 5)
    lp_p95 = np.percentile(lp_final_returns, 95)
    
    # Find visible range for mean lines
    visible_mean_eth = min(eth_mean, eth_upper_bound) if eth_mean <= eth_upper_bound else None
    visible_mean_lp = min(lp_mean, lp_upper_bound) if lp_mean <= lp_upper_bound else None
    
    # Add mean lines (only if visible in the current range)
    if visible_mean_eth is not None:
        fig.add_trace(
            go.Scatter(
                x=[visible_mean_eth, visible_mean_eth],
                y=[0, max(eth_hist) * 1.1],
                mode='lines',
                name=f'Mean ETH Return: {eth_mean:.2f}x',
                line=dict(color='blue', width=2, dash='dash')
            )
        )
    
    if visible_mean_lp is not None:
        fig.add_trace(
            go.Scatter(
                x=[visible_mean_lp, visible_mean_lp],
                y=[0, max(lp_hist) * 1.1],
                mode='lines',
                name=f'Mean LP Return: {lp_mean:.2f}x',
                line=dict(color='red', width=2, dash='dash')
            )
        )
    
    # Add historical return lines if provided (and if visible in current range)
    if historical_eth_return is not None and historical_eth_return <= eth_upper_bound:
        fig.add_trace(
            go.Scatter(
                x=[historical_eth_return, historical_eth_return],
                y=[0, max(eth_hist) * 1.1],
                mode='lines',
                name=f'Historical ETH Return: {historical_eth_return:.2f}x',
                line=dict(color='green', width=3)
            )
        )
    
    if historical_lp_return is not None and historical_lp_return <= lp_upper_bound:
        fig.add_trace(
            go.Scatter(
                x=[historical_lp_return, historical_lp_return],
                y=[0, max(lp_hist) * 1.1],
                mode='lines',
                name=f'Historical LP Return: {historical_lp_return:.2f}x',
                line=dict(color='orange', width=3)
            )
        )
    
    # Add annotations for statistics
    eth_annotations = [
        f"ETH Mean: {eth_mean:.2f}x",
        f"ETH Median: {eth_median:.2f}x",
        f"ETH Std Dev: {eth_std:.2f}",
        f"ETH 5th Percentile: {eth_p5:.2f}x",
        f"ETH 95th Percentile: {eth_p95:.2f}x"
    ]
    
    lp_annotations = [
        f"LP Mean: {lp_mean:.2f}x",
        f"LP Median: {lp_median:.2f}x",
        f"LP Std Dev: {lp_std:.2f}",
        f"LP 5th Percentile: {lp_p5:.2f}x",
        f"LP 95th Percentile: {lp_p95:.2f}x"
    ]
    
    # Labels and titles based on fee model
    if use_dynamic_fee:
        fee_label = "Dynamic Fee (Equal to IL)"
        lp_name = f"LP Position (Dynamic Fee)"
    else:
        fee_label = f"{fee_percentage*100:.1f}% Fee"
        lp_name = f"LP Position ({fee_percentage*100:.1f}% Fee)"
        
    title = f"Distribution of Returns: ETH vs LP ({fee_label})"

    # Update layout - Move stat boxes to the right side
    fig.update_layout(
        height=600,
        title=title,
        xaxis_title="Final Return Multiple (x)",
        yaxis_title="Probability Density",
        annotations=[
            dict(
                x=0.99,  # Moved from 0.01 to 0.99 (right side)
                y=0.99,
                xref="paper",
                yref="paper",
                text="<br>".join(eth_annotations),
                showarrow=False,
                align="right",  # Changed from 'left' to 'right'
                bgcolor="rgba(0, 0, 255, 0.1)",
                bordercolor="blue",
                borderwidth=1,
                font=dict(size=12)
            ),
            dict(
                x=0.99,  # Moved from 0.01 to 0.99 (right side)
                y=0.73,
                xref="paper",
                yref="paper",
                text="<br>".join(lp_annotations),
                showarrow=False,
                align="right",  # Changed from 'left' to 'right'
                bgcolor="rgba(255, 0, 0, 0.1)",
                bordercolor="red",
                borderwidth=1,
                font=dict(size=12)
            )
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_dark"
    )
    
    # Add a note about filtering for better visualization
    if len(eth_filtered) < len(eth_final_returns) or len(lp_filtered) < len(lp_final_returns):
        fig.add_annotation(
            x=0.5,
            y=1.05,
            xref="paper",
            yref="paper",
            text=f"Note: Chart excludes extreme outliers > {eth_upper_bound:.0f}x for better visualization. Stats reflect all data.",
            showarrow=False,
            align="center",
            font=dict(size=10, color="gray")
        )
    
    return fig

def plot_fee_distribution(fee_paths, use_dynamic_fee=False, fee_percentage=0.003, fee_dial_setting=0.01, num_bins=50):
    """
    Plot the distribution of fees charged in the simulations.
    
    Args:
        fee_paths (np.ndarray): 2D array of fee paths from Monte Carlo simulation
        use_dynamic_fee (bool): Whether dynamic fee model was used
        fee_percentage (float): Fee percentage used (for flat fee model)
        fee_dial_setting (float): Fee dial setting used (for dynamic fee model)
        num_bins (int): Number of bins for the histogram
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Extract fee values, removing zeros (no fees on day 0)
    fee_values = fee_paths.flatten()
    fee_values = fee_values[fee_values > 0]
    
    # Convert fees to percentages for display (multiply by 100)
    fee_values_pct = fee_values * 100
    
    # Calculate relevant statistics (as percentages)
    mean_fee_pct = np.mean(fee_values_pct)
    median_fee_pct = np.median(fee_values_pct)
    max_fee_pct = np.max(fee_values_pct)
    
    # Create histogram of fee values using histnorm to normalize the counts
    fig = go.Figure()
    
    # Set title and layout based on fee model
    if use_dynamic_fee:
        title = f"Distribution of Dynamic Fees (Fee Dial: {fee_dial_setting*100:.2f}%)"
        x_axis_title = "Fee Value (%)"
        hist_color = "rgba(55, 128, 191, 0.7)"
    else:
        title = f"Flat Fee Model ({fee_percentage*100:.2f}%)"
        x_axis_title = "Fee Value (%)"
        hist_color = "rgba(219, 64, 82, 0.7)"
    
    # Create histogram with normalized counts
    fig.add_trace(go.Histogram(
        x=fee_values_pct,
        nbinsx=num_bins,
        marker_color=hist_color,
        name="Fee Distribution",
        histnorm='percent'  # Normalize to percentage
    ))
    
    # Add vertical lines for statistics
    fig.add_vline(x=mean_fee_pct, line_dash="dash", line_color="green", 
                 annotation_text=f"Mean: {mean_fee_pct:.4f}%", annotation_position="top right")
    fig.add_vline(x=median_fee_pct, line_dash="dash", line_color="blue", 
                 annotation_text=f"Median: {median_fee_pct:.4f}%", annotation_position="top left")
    
    # If using flat fee, add annotation explaining the single spike
    if not use_dynamic_fee:
        fig.add_annotation(
            x=median_fee_pct,
            y=0.9,
            xref="x",
            yref="paper",
            text="Flat fee model shows single value",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title="Frequency (%)",
        bargap=0.1,
        template="plotly_dark",
        margin=dict(l=10, r=10, b=10, t=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
    )
    
    # Add explanatory text as annotation
    if use_dynamic_fee:
        fig.add_annotation(
            x=0.5,
            y=1.05,
            xref="paper",
            yref="paper",
            text="Dynamic fees vary based on the amount of impermanent loss at each step",
            showarrow=False,
            font=dict(size=14),
            align="center",
            bgcolor="rgba(50, 50, 50, 0.6)",
            bordercolor="rgba(100, 100, 100, 0.8)",
            borderwidth=1,
            borderpad=4
        )
    
    return fig 