#%% BASIC IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
sys.path.append("../../")
import plotly.express as px

from src.models.lp_model import UniswapV2Pool

path = "/Users/jordi/Documents/GitHub/no-gamma-hook-v4/data/swap_fee_data.csv" 

def load_data(path):
    return pd.read_csv(path)

def save_data(data, path):
    data.to_csv(path, index=False)
# %% UNFOLD RESERVE COLUMN

df = load_data(path)

df[['reserve_amount_0', 'reserve_amount_1']] = df['reserveAmounts'].str.split(',', expand=True)

# Remove brackets and quotes
df['reserve_amount_arb'] = (df['reserve_amount_0'].str.replace(r"[\[\]']", '', regex=True))
df['reserve_amount_usdc'] = (df['reserve_amount_1'].str.replace(r"[\[\]']", '', regex=True))

df['reserve_amount_arb'] = df['reserve_amount_arb'].astype(float)
df['reserve_amount_usdc'] = df['reserve_amount_usdc'].astype(float)

df.head()

# %% WE EXTEND SWAP DATA TO INCLUDE SYMBOL AND AMOUNT INFO AND CRUCIALLY amount_vs_reserve
import numpy as np
# Ensure amount columns are numeric
df['amountIn'] = pd.to_numeric(df['amountIn'])
df['amountOut'] = pd.to_numeric(df['amountOut'])

# Create USD_amount and ARB_amount columns
df['USDC_amount'] = np.where(df['tokenIn_symbol'] == 'USDC',
                              df['amountIn'],
                              -df['amountOut'])

df['ARB_amount'] = np.where(df['tokenIn_symbol'] == 'ARB',
                              df['amountIn'],
                              -df['amountOut'])

df['arb_amount_vs_reserve'] = df['ARB_amount'] / df['reserve_amount_arb']
df['usdc_amount_vs_reserve'] = df['USDC_amount'] / df['reserve_amount_usdc']

df[['tokenIn_symbol', 'amountIn', 'amountOut', 'USDC_amount', 'ARB_amount', 'reserve_amount_usdc', 'reserve_amount_arb', 'arb_amount_vs_reserve', 'usdc_amount_vs_reserve']].head()

df.head()

# %% WE PLOT HISTOGRAMS OF ARB AND USDC AMOUNT VS RESERVE

# Create subplots
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('Histogram of ARB Amount vs Reserve',
                                    'Histogram of USDC Amount vs Reserve'))

# Add ARB histogram
fig.add_trace(
    go.Histogram(x=df['arb_amount_vs_reserve'], nbinsx=1000, name='ARB'),
    row=1, col=1
)

# Add USDC histogram
fig.add_trace(
    go.Histogram(x=df['usdc_amount_vs_reserve'], nbinsx=1000, name='USDC'),
    row=1, col=2
)

# Update layout
fig.update_layout(title_text='Histograms of Amount vs Reserve Ratios',
                  bargap=0.1)

fig.update_xaxes(title_text="ARB Amount / Reserve ARB", row=1, col=1)
fig.update_xaxes(title_text="USDC Amount / Reserve USDC", row=1, col=2)
fig.update_yaxes(title_text="Frequency", row=1, col=1)
fig.update_yaxes(title_text="Frequency", row=1, col=2)

fig.show()

# %% WE GET PROBABILITY DENSITIES of sum 1
# Calculate density using np.histogram
arb_pdf = np.histogram(df['arb_amount_vs_reserve'], bins=1000, density=True)
usdc_pdf = np.histogram(df['usdc_amount_vs_reserve'], bins=1000, density=True)

arb_pdf_bins = arb_pdf[1]
usdc_pdf_bins = usdc_pdf[1]

arb_pdf_bin_midpoints = (arb_pdf_bins[:-1] + arb_pdf_bins[1:]) / 2
usdc_pdf_bin_midpoints = (usdc_pdf_bins[:-1] + usdc_pdf_bins[1:]) / 2

arb_pdf_density = arb_pdf[0]
usdc_pdf_density = usdc_pdf[0]



# %% PLOT NORMALIZED HISTOGRAMS (PDFs)
fig = go.Figure()

# ARB PDF
fig.add_trace(go.Scatter(
    x=arb_pdf_bin_midpoints, # Use left bin edges for x
    y=arb_pdf_density,      # Use the calculated density directly
    mode='lines', 
    name='ARB PDF'
))

# USDC PDF
fig.add_trace(go.Scatter(
    x=usdc_pdf_bin_midpoints, # Use left bin edges for x
    y=usdc_pdf_density,      # Use the calculated density directly
    mode='lines', 
    name='USDC PDF'
))

fig.update_layout(
    title='Probability Density Functions (PDF)',
    xaxis_title='Relative Amount vs Reserve',
    yaxis_title='Density',
    legend_title='Token'
)

fig.show()

# %% WE CREATE THE CDF FROM THE PDF

# Calculate ARB CDF (Cumulative sum of Density * Bin Width)
arb_bin_widths = np.diff(arb_pdf_bins) # Width of each bin
# CDF value at the right edge of bin i is sum(density[j]*width[j] for j<=i)
arb_cdf_vals = np.cumsum(arb_pdf_density * arb_bin_widths)
# Add 0 at the beginning for the CDF value at the first edge
arb_cdf_vals = np.insert(arb_cdf_vals, 0, 0) 
arb_cdf_x_values = arb_pdf_bins # CDF x-values are the bin edges

# Calculate USDC CDF
usdc_bin_widths = np.diff(usdc_pdf_bins)
usdc_cdf_vals = np.cumsum(usdc_pdf_density * usdc_bin_widths)
usdc_cdf_vals = np.insert(usdc_cdf_vals, 0, 0)
usdc_cdf_x_values = usdc_pdf_bins


# Plot ARB CDF 
fig = go.Figure()
# Plot CDF values against the bin edges
fig.add_trace(go.Scatter(x=arb_cdf_x_values, y=arb_cdf_vals, mode='lines', name='ARB CDF'))
fig.add_trace(go.Scatter(x=usdc_cdf_x_values, y=usdc_cdf_vals, mode='lines', name='USDC CDF'))
fig.update_layout(title='ARB Cumulative Distribution Function (CDF)',
                  xaxis_title='Relative ARB Amount vs Reserve',
                  yaxis_title='Cumulative Probability')
fig.show()

# %% GENERATE ARB TRADES FROM CDF

def generate_arb_trades(num_trades):
    """Generates synthetic trades using inverse transform sampling with linear interpolation."""
    u = np.random.rand(num_trades)
    
    # Find indices where u would be inserted into the CDF values
    # We search sorted on arb_cdf_vals, which correspond to arb_cdf_x_values (bin edges)
    indices = np.searchsorted(arb_cdf_vals, u, side='right')

    # Handle edge cases: ensure indices are within valid range [1, len(cdf_vals))
    # so that indices-1 is always valid.
    indices = np.clip(indices, 1, len(arb_cdf_vals) - 1)

    # Get CDF values and bin edges for interpolation
    cdf_right = arb_cdf_vals[indices]
    cdf_left = arb_cdf_vals[indices - 1] 
    bin_edge_right = arb_cdf_x_values[indices] # Bin edge corresponding to cdf_right
    bin_edge_left = arb_cdf_x_values[indices - 1] # Bin edge corresponding to cdf_left
    
    # Calculate interpolation fraction (how far is u between cdf_left and cdf_right)
    cdf_diff = cdf_right - cdf_left
    # Avoid division by zero if CDF is flat (highly concentrated data)
    fraction = np.where(cdf_diff > 1e-12, (u - cdf_left) / cdf_diff, 0) 

    # Linear interpolation between the bin edges
    samples = bin_edge_left + fraction * (bin_edge_right - bin_edge_left)
    
    return samples


num_synthetic_trades = 1000
synthetic_arb_trades = generate_arb_trades(num_synthetic_trades)

# Display first 10 samples
print("First 10 synthetic relative ARB trades:")
print(synthetic_arb_trades[:10])

# %% COMPARE HISTORICAL AND SYNTHETIC ARB PDF

# Generate 10,000 synthetic trades for comparison
num_synthetic_trades_comp = 10000
synthetic_arb_trades_comp = generate_arb_trades(num_synthetic_trades_comp)

# Calculate the PDF of the synthetic trades using the same bins as the historical data
# Use the density values directly
synthetic_pdf_arb_vals, synthetic_pdf_arb_bins = np.histogram(synthetic_arb_trades_comp, bins=arb_pdf_bins, density=True)

# Plot both PDFs for comparison
fig = go.Figure()

# Historical PDF
fig.add_trace(go.Scatter(
    x=arb_pdf_bin_midpoints, # Use bin midpoints for plotting PDF
    y=arb_pdf_density, 
    mode='lines', 
    name='Historical ARB PDF'
))

# Synthetic PDF
# Calculate midpoints for synthetic plot
synthetic_pdf_arb_bin_midpoints = (synthetic_pdf_arb_bins[:-1] + synthetic_pdf_arb_bins[1:]) / 2
fig.add_trace(go.Scatter(
    x=synthetic_pdf_arb_bin_midpoints, 
    y=synthetic_pdf_arb_vals, 
    mode='lines', 
    name='Synthetic ARB PDF',
    line=dict(dash='dash')
))

fig.update_layout(
    title='Comparison of Historical vs Synthetic ARB Trade Distribution (PDF)',
    xaxis_title='Relative ARB Amount vs Reserve',
    yaxis_title='Probability Density',
    legend_title='Data Source'
)

fig.show()

# %% COMPARE HISTORICAL AND SYNTHETIC ARB CDF

# Calculate the synthetic CDF from the synthetic PDF
synthetic_arb_bin_widths = np.diff(synthetic_pdf_arb_bins)
synthetic_arb_cdf_vals = np.cumsum(synthetic_pdf_arb_vals * synthetic_arb_bin_widths)
# Add 0 at the beginning for the first edge
synthetic_arb_cdf_vals = np.insert(synthetic_arb_cdf_vals, 0, 0)

# Plot both CDFs for comparison
fig = go.Figure()

# Historical CDF
fig.add_trace(go.Scatter(
    x=arb_cdf_x_values, # Use bin edges 
    y=arb_cdf_vals, 
    mode='lines', 
    name='Historical ARB CDF'
))

# Synthetic CDF
fig.add_trace(go.Scatter(
    x=synthetic_pdf_arb_bins, # Use bin edges from synthetic histogram 
    y=synthetic_arb_cdf_vals, 
    mode='lines', 
    name='Synthetic ARB CDF',
    line=dict(dash='dash')
))

fig.update_layout(
    title='Comparison of Historical vs Synthetic ARB Trade Cumulative Distribution (CDF)',
    xaxis_title='Relative ARB Amount vs Reserve',
    yaxis_title='Cumulative Probability',
    legend_title='Data Source'
)

fig.show()

# %% ========= USDC Analysis =========

# Plot USDC CDF
fig = go.Figure()
fig.add_trace(go.Scatter(x=usdc_cdf_x_values, 
                         y=usdc_cdf_vals, 
                         mode='lines', 
                         name='USDC CDF'))
fig.update_layout(title='USDC Cumulative Distribution Function (CDF)',
                  xaxis_title='Relative USDC Amount vs Reserve',
                  yaxis_title='Cumulative Probability')
fig.show()

# %% GENERATE USDC TRADES FROM CDF

# Function to generate synthetic USDC trades
def generate_usdc_trades(num_trades):
    """Generates synthetic trades using inverse transform sampling with linear interpolation."""
    u = np.random.rand(num_trades)
    indices = np.searchsorted(usdc_cdf_vals, u, side='right')

    # Handle edge cases
    indices = np.clip(indices, 1, len(usdc_cdf_vals) - 1)

    # Interpolation logic (similar to ARB)
    cdf_right = usdc_cdf_vals[indices]
    cdf_left = usdc_cdf_vals[indices - 1]
    bin_edge_right = usdc_cdf_x_values[indices]
    bin_edge_left = usdc_cdf_x_values[indices - 1]
    
    cdf_diff = cdf_right - cdf_left
    fraction = np.where(cdf_diff > 1e-12, (u - cdf_left) / cdf_diff, 0)
    samples = bin_edge_left + fraction * (bin_edge_right - bin_edge_left)

    return samples

# %% GENERATE & VIEW SYNTHETIC USDC TRADES
# Generate 10,000 synthetic USDC trades
num_synthetic_trades_usdc = 10000
synthetic_usdc_trades = generate_usdc_trades(num_synthetic_trades_usdc)

# Display first 10 samples
print("First 10 synthetic relative USDC trades:")
print(synthetic_usdc_trades[:10])

# %% COMPARE HISTORICAL AND SYNTHETIC USDC PDF

# Calculate the PDF of the synthetic USDC trades
synthetic_usdc_pdf_vals, synthetic_usdc_pdf_bins = np.histogram(synthetic_usdc_trades, bins=usdc_pdf_bins, density=True)

# Plot both PDFs for comparison
fig = go.Figure()

# Historical PDF
fig.add_trace(go.Scatter(
    x=usdc_pdf_bin_midpoints, 
    y=usdc_pdf_density, 
    mode='lines', 
    name='Historical USDC PDF'
))

# Synthetic PDF
synthetic_usdc_pdf_bin_midpoints = (synthetic_usdc_pdf_bins[:-1] + synthetic_usdc_pdf_bins[1:]) / 2
fig.add_trace(go.Scatter(
    x=synthetic_usdc_pdf_bin_midpoints, 
    y=synthetic_usdc_pdf_vals, 
    mode='lines', 
    name='Synthetic USDC PDF',
    line=dict(dash='dash')
))

fig.update_layout(
    title='Comparison of Historical vs Synthetic USDC Trade Distribution (PDF)',
    xaxis_title='Relative USDC Amount vs Reserve',
    yaxis_title='Probability Density',
    legend_title='Data Source'
)

fig.show()

# %% COMPARE HISTORICAL AND SYNTHETIC USDC CDF

# Calculate the synthetic CDF from the PDF
synthetic_usdc_bin_widths = np.diff(synthetic_usdc_pdf_bins)
synthetic_usdc_cdf_vals = np.cumsum(synthetic_usdc_pdf_vals * synthetic_usdc_bin_widths)
synthetic_usdc_cdf_vals = np.insert(synthetic_usdc_cdf_vals, 0, 0)

# Plot both CDFs for comparison
fig = go.Figure()

# Historical CDF
fig.add_trace(go.Scatter(
    x=usdc_cdf_x_values, 
    y=usdc_cdf_vals, 
    mode='lines', 
    name='Historical USDC CDF'
))

# Synthetic CDF
fig.add_trace(go.Scatter(
    x=synthetic_usdc_pdf_bins, # Use bin edges 
    y=synthetic_usdc_cdf_vals, 
    mode='lines', 
    name='Synthetic USDC CDF',
    line=dict(dash='dash')
))

fig.update_layout(
    title='Comparison of Historical vs Synthetic USDC Trade Cumulative Distribution (CDF)',
    xaxis_title='Relative USDC Amount vs Reserve',
    yaxis_title='Cumulative Probability',
    legend_title='Data Source'
)

fig.show()


# %% ===== TRADE SEQUENCE SIMULATION ======

# Function to create a sequence of trades with running reserves using UniswapV2Pool class
def simulate_trade_sequence_class(pool, arb_rel_amounts, usdc_rel_amounts, initial_reserve_x, initial_reserve_y):
    """
    Simulates a sequence of trades using synthetic samples and the UniswapV2Pool class,
    updating reserves after each trade.

    Args:
        arb_rel_amounts (np.ndarray): Numpy array of synthetic relative ARB amounts.
        usdc_rel_amounts (np.ndarray): Numpy array of synthetic relative USDC amounts.
        initial_reserve_x (float): Initial reserve of token X (USDC).
        initial_reserve_y (float): Initial reserve of token Y (ARB).
        fee_tier (float): Fee percentage (default 0.003 for 0.3%).

    Returns:
        pd.DataFrame: DataFrame with trade inputs and outputs.
    """
    num_trades = len(arb_rel_amounts)
    if len(usdc_rel_amounts) != num_trades:
        raise ValueError("Sample arrays must have the same length")


    results = []

    for i in range(num_trades):
        # Store state before the swap
        reserve_x_before = pool.x
        reserve_y_before = pool.y

        # Randomly choose which token is coming in (50/50 chance)
        # Assuming Token X is USDC (index 0) and Token Y is ARB (index 1)
        token_in_is_USDC = np.random.random() > 0.5

        # Get the appropriate sample and calculate absolute amount_in
        if token_in_is_USDC:
            # Token X (USDC) is coming in
            relative_amount = usdc_rel_amounts[i] # Use numpy array indexing
            amount_in = abs(relative_amount) * pool.x
        else:
            # Token Y (ARB) is coming in
            relative_amount = arb_rel_amounts[i] # Use numpy array indexing
            amount_in = abs(relative_amount) * pool.y

        # Ensure amount_in is positive (direction is handled by token_in_is_x)
        amount_in = abs(amount_in)

        # Execute the swap using the pool object's method
        try:
            amount_out = pool.swap(amount_in, token_in_is_USDC)
        except ValueError as e:
            print(f"Warning: Swap {i} failed: {e}. Skipping trade.")
            # Optionally add a row with NaNs or skip appending
            continue

        # Get pool state *after* the swap
        reserve_x_after, reserve_y_after, fees_x_total, fees_y_total = pool.get_balances()

        # Calculate fee for *this* trade
        fee_x_trade = amount_in * fee_tier if token_in_is_USDC else 0
        fee_y_trade = amount_in * fee_tier if not token_in_is_USDC else 0
        
        
        # Store the inputs and outputs for this trade
        trade_data = {
            'trade_number': i,
            'reserve_x_before': reserve_x_before,
            'reserve_y_before': reserve_y_before,
            'token_in_is_USDC': token_in_is_USDC, # True if USDC in
            'amount_in': amount_in,
            'relative_amount': relative_amount, # The sample used
            'reserve_x_after': reserve_x_after,
            'reserve_y_after': reserve_y_after,
            'amount_out': amount_out,
            'fee_x_trade': fee_x_trade, # Fee for this specific trade
            'fee_y_trade': fee_y_trade, # Fee for this specific trade
            'fee_tier': fee_tier
        }
        results.append(trade_data)


    trade_simulation = pd.DataFrame(results)

    # WE DERIVE THE PRICE FROM THE RESERVES AND PLOT BOTH RESERVES AND PRICE

    # we derive the price from the reserves (Price = Y/X = ARB/USDC)
    trade_simulation['price'] = trade_simulation['reserve_x_after'] / trade_simulation['reserve_y_after']

    # Calculate Pool Value (in USDC)
    # Value = reserve_x + reserve_y * (USDC/ARB price)
    # Since price = ARB/USDC, USDC/ARB = 1/price
    # Need to handle potential division by zero if price is zero
    trade_simulation['pool_value_usdc'] = trade_simulation['reserve_x_after'] + \
                                            trade_simulation['reserve_y_after'] * trade_simulation['price'].replace(0, np.nan)

    # Calculate HODL Value (in USDC)
    # Value = initial_reserve_x + initial_reserve_y * (USDC/ARB price at step i)
    hodl_value_usdc = initial_reserve_x + initial_reserve_y * trade_simulation['price'].replace(0, np.nan)
    trade_simulation['hodl_value_usdc'] = hodl_value_usdc


    return trade_simulation 


# Generate synthetic trade samples *before* calling the simulation function
num_trades_to_simulate = 15000 # Or use the desired number
num_plot_trades = 7000 # Keep this smaller for faster initial testing/plotting if needed - also only positives showing
fee_tier = 0.005

initial_reserve_x = 100  # 1 million USDC
initial_reserve_y = 100   # 1M ARB (as per user change)

pool = UniswapV2Pool(initial_reserve_x, initial_reserve_y, fee_percentage=fee_tier)

print("Pool fee percentage:", pool.fee_percentage)

arb_samples = generate_arb_trades(num_trades_to_simulate)
arb_samples = arb_samples[arb_samples > 0]
usdc_samples = generate_usdc_trades(num_trades_to_simulate)
usdc_samples = usdc_samples[usdc_samples > 0]

arb_samples_run = arb_samples[:num_plot_trades]
usdc_samples_run = usdc_samples[:num_plot_trades]

trade_simulation = simulate_trade_sequence_class(
    pool,
    arb_samples_run,
    usdc_samples_run,
    initial_reserve_x=initial_reserve_x,
    initial_reserve_y=initial_reserve_y
)


def get_3_plots(trade_simulation):
    # Create subplots: 1 row, 3 columns
    fig_combined = make_subplots(rows=1, cols=3,
                            horizontal_spacing=0.05, # Adjust spacing
                            subplot_titles=(
                                'Reserve Changes',
                                'Derived Price (ARB/USDC)',
                                'Portfolio Value (USDC)'
                            ))

    # Subplot 1: Reserves
    # Plot USDC Reserve (X)
    fig_combined.add_trace(go.Scatter(
        x=trade_simulation['trade_number'],
        y=trade_simulation['reserve_x_after'],
        mode='lines',
        name='USDC Reserve (X)',
        legendgroup='reserves' # Group legends
    ), row=1, col=1)

    # Plot ARB Reserve (Y)
    fig_combined.add_trace(go.Scatter(
        x=trade_simulation['trade_number'],
        y=trade_simulation['reserve_y_after'],
        mode='lines',
        name='ARB Reserve (Y)',
        legendgroup='reserves'
    ), row=1, col=1)

    # Subplot 2: Derived Price
    fig_combined.add_trace(go.Scatter(
        x=trade_simulation['trade_number'],
        y=trade_simulation['price'],
        mode='lines',
        name='Derived Price (ARB/USDC)', # Updated name for clarity
        legendgroup='price'
    ), row=1, col=2)

    # Subplot 3: Portfolio Values
    # Plot Pool Value
    fig_combined.add_trace(go.Scatter(
        x=trade_simulation['trade_number'],
        y=trade_simulation['pool_value_usdc'],
        mode='lines',
        name='LP Value (USDC)',
        legendgroup='value'
    ), row=1, col=3)

    # Plot HODL Value
    fig_combined.add_trace(go.Scatter(
        x=trade_simulation['trade_number'],
        y=trade_simulation['hodl_value_usdc'],
        mode='lines',
        name='HODL Value (USDC)',
        legendgroup='value',
        line=dict(dash='dash') # Dashed line for HODL
    ), row=1, col=3)


    # Update layout for the combined figure
    fig_combined.update_layout(
        title_text='Simulation Results: Reserves, Price, and Value Comparison',
        height=500, # Adjust height
        width=1200, # Adjust width for side-by-side (3 plots)
        legend=dict(
            orientation="h", # Horizontal legend
            yanchor="bottom",
            y=-0.2, # Position below plots
            xanchor="center",
            x=0.5
        )
    )

    # Update y-axis titles for each subplot
    fig_combined.update_yaxes(title_text='Reserve Amount', row=1, col=1)
    fig_combined.update_yaxes(title_text='Derived Price (ARB/USDC)', row=1, col=2)
    fig_combined.update_yaxes(title_text='Portfolio Value (USDC)', row=1, col=3)

    # Update x-axis titles for all subplots
    fig_combined.update_xaxes(title_text='Trade Number', row=1, col=1)
    fig_combined.update_xaxes(title_text='Trade Number', row=1, col=2)
    fig_combined.update_xaxes(title_text='Trade Number', row=1, col=3)

    return fig_combined

fig_combined = get_3_plots(trade_simulation)
fig_combined.show()

#%% PAYOFF PLOT
def payoff_plot(trade_simulation):

    fig = go.Figure()

    starting_value = initial_reserve_x + initial_reserve_y
    x_range = trade_simulation['price']
    y_range_bh = trade_simulation['hodl_value_usdc']/starting_value
    y_range_lp = trade_simulation['pool_value_usdc']/starting_value


    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_range_bh,
        mode='markers',
        name='HODL Value'
    ))

    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_range_lp,
        mode='markers',
        name='LP Value'
    ))

    fig.update_layout(
        title='HODL Value and LP Value Over Price',
        xaxis_title='Price (ARB/USDC)', 
        yaxis_title='Value (USDC)'
    )

    return fig

fig = payoff_plot(trade_simulation)
fig.show()


# %% ===== SWAP RANGE SWEEP =====
def sweep_swap_amount_range(pool_instance, initial_x, initial_y, relative_amounts, fee_tier):
    """
    Simulates single swaps for a range of order sizes expressed relative to the reserve balance.
    This function helps test the behaviour of a single swap for a wide range of potential incoming order sizes.

    Args:
        pool_instance (UniswapV2Pool): An initialized pool object.
        initial_x (float): Initial reserve of token X (USDC).
        initial_y (float): Initial reserve of token Y (ARB).
        relative_amounts (np.ndarray): Array of relative trade sizes (e.g., 0.01 for 1%).
        fee_tier (float): The fee percentage used for the pool.

    Returns:
        pd.DataFrame: DataFrame containing simulation results.
    """
    results = []
    pool_instance.fee_percentage = fee_tier # Ensure pool has the correct fee

    for rel_amount in relative_amounts:
        pool_instance.reset() # Start fresh for each relative amount
        pool_instance.fee_percentage = fee_tier # Ensure pool has the correct fee

        if rel_amount > 0: # Positive means USDC is coming in, ARB is going out
            amount_in = rel_amount * initial_y
        else: # Negative means ARB is coming in, USDC is going out
            amount_in = rel_amount * initial_x

        if amount_in <= 1e-9 and amount_in >= 0: # Skip negligible or zero swaps
            continue

        try:
            is_usdc_in = True if amount_in > 0 else False
            # amount_in = abs(amount_in)
            amount_out = pool_instance.swap(abs(amount_in), is_usdc_in)
            new_x, new_y, fee_x_total, fee_y_total = pool_instance.get_balances()

            # Calculate metrics AFTER the swap
            if new_x > 1e-12:
                new_price = new_x / new_y # Price = ARB / USDC
            else:
                new_price = np.nan # Avoid division by zero
            
            lp_value_usdc = new_x + new_y * new_price 
            bh_value_usdc = initial_x + initial_y * new_price 

            if np.isnan(lp_value_usdc) or np.isnan(bh_value_usdc):
                 print(f"Warning: NaN value calculated for rel_amount={rel_amount}, new_price={new_price}")
                 continue # Skip if calculation failed

            if rel_amount < 0:
                amount_in = amount_in*new_price
                swap_price_usdc = amount_out/amount_in

            if rel_amount >= 0:
                swap_price_usdc = amount_in/amount_out

            results.append({
                'relative_amount_in': rel_amount,
                'amount_in': amount_in,
                'amount_out': amount_out,
                'token_in_is_USDC': is_usdc_in,
                'new_x': new_x,
                'new_y': new_y,
                'new_price': new_price, # ARB/USDC
                'lp_value': lp_value_usdc,
                'bh_value': bh_value_usdc,
                'swap_price': new_price,
            })

        except ValueError as e:
            print(f"Swap failed for relative amount {rel_amount}: {e}")
            continue

    return pd.DataFrame(results)


def plot_single_swap_payoff(df_single_swap, fee_tier=None):
    """
    Plots the LP vs HODL value against the resulting price from single swap simulations.
    """
    fig = go.Figure()

    # Ensure price is not NaN for plotting
    plot_df = df_single_swap.dropna(subset=['new_price'])

    if fee_tier != 'Hook' and fee_tier is not None:
        series_name = f'V2 Pool (Fee: {fee_tier*100:.2f}%)'
    elif fee_tier == 'Hook':
        series_name = 'V4 Pool (V2 + Gamma Hook)'

    if fee_tier == 0.0:
        series_name = 'V2 Pool (0% Fee)'

    


    # Plot LP Value first (blue markers)
    fig.add_trace(go.Scatter(
        x=plot_df['new_price'],
        y=plot_df['lp_value'],
        mode='markers',
        name=series_name,
        marker=dict(color='blue')
    ))

    # Plot HODL Value second (orange dashed markers) to ensure it is on top
    fig.add_trace(go.Scatter(
        x=plot_df['new_price'],
        y=plot_df['bh_value'],
        mode='markers', # Keep markers
        name='HODL Value',
        marker=dict(color='orange'), # Use a marker that looks like a dash segment
        line=dict(dash='dash') # This might not render correctly with mode='markers'. Let's try marker symbol.
    ))

    # Add hover text details - Applying to both traces
    fig.update_traces(hovertemplate=
        "<b>Price (ARB/USDC):</b> %{x:.4f}<br>" +
        "<b>Value (USDC):</b> %{y:,.2f}<br>" +
        "Relative Amount In: %{customdata[0]:.2%}<extra></extra>",
        customdata=plot_df[['relative_amount_in', 'token_in_is_USDC']].applymap(lambda x: 'USDC' if x is True else ('ARB' if x is False else x))
    )

    initial_price = initial_reserve_x / initial_reserve_y if initial_reserve_y > 1e-12 else np.nan
    initial_value = initial_reserve_x + initial_reserve_y * initial_price if not np.isnan(initial_price) and abs(initial_price) > 1e-12 else initial_reserve_x

    if fee_tier != 'Hook' and fee_tier is not None:
        fee_tier_str = f'{fee_tier*100:.2f}%'
    elif fee_tier == 'Hook':
        fee_tier_str = 'Dynamic - Gamma Hook'

    fig.update_layout(
        title=f'Single Swap Payoff (Fee: {fee_tier_str}) - Initial Price: {initial_price:.4f}' if not df_single_swap.empty else 'Single Swap Payoff', # Get fee from df
        xaxis_title='Resulting Price (ARB/USDC)',
        yaxis_title='Portfolio Value (USDC)',
        legend_title='Portfolio Type',
        hovermode='closest'
    )

    # Add a vertical line for the initial price
    if not np.isnan(initial_price):
        fig.add_vline(x=initial_price, line_width=1, line_dash="dash", line_color="grey",
                      annotation_text="Initial Price", annotation_position="top right")

    # Add a horizontal line for the initial value
    if not np.isnan(initial_value):
         fig.add_hline(y=initial_value, line_width=1, line_dash="dash", line_color="grey",
                       annotation_text="Initial Value", annotation_position="bottom right")

    return fig

#%% V2 SWEEP 
# --- Simulation Parameters ---
single_swap_fee_tier = 0.3 # Example: 0.3% fee    
initial_x_single = float(100) # 1 million USDC
initial_y_single = float(100)   # 1 million ARB
num_points = 100 # Number of relative amounts to test
max_relative_amount = 0.5 # Test swaps up to 50% of reserve
gamma_factor = 0.5
# Generate a range of relative amounts (log scale might be better for small values)
# relative_amounts_in = np.linspace(0.001, max_relative_amount, num_points)
relative_amounts_in = np.linspace(-max_relative_amount+0.000000001, max_relative_amount, num_points) # 0.01% to max%


single_swap_pool_0_fee = UniswapV2Pool(initial_x_single, initial_y_single, fee_percentage=0.0)
print(f"Starting single swap simulation with Fee: {single_swap_fee_tier*100:.2f}%")
v2_swap_sweep_results_0_fee_df = sweep_swap_amount_range(
    pool_instance=single_swap_pool_0_fee,
    initial_x=initial_x_single,
    initial_y=initial_y_single,
    relative_amounts=relative_amounts_in,
    fee_tier=0.0
)


single_swap_pool_flat_fee = UniswapV2Pool(initial_x_single, initial_y_single, fee_percentage=single_swap_fee_tier)
print(f"Starting single swap simulation with Fee: {single_swap_fee_tier*100:.2f}%")
v2_swap_sweep_results_flat_fee_df = sweep_swap_amount_range(
    pool_instance=single_swap_pool_flat_fee,
    initial_x=initial_x_single,
    initial_y=initial_y_single,
    relative_amounts=relative_amounts_in,
    fee_tier=single_swap_fee_tier
)

print(f"Generated {len(v2_swap_sweep_results_flat_fee_df)} single swap results.")
print(v2_swap_sweep_results_flat_fee_df.head())

# --- Plot Results ---
v2_swap_sweep_plot = plot_single_swap_payoff(v2_swap_sweep_results_flat_fee_df, fee_tier=single_swap_fee_tier)
v2_swap_sweep_plot.show()

# This is an interesting plot, because it incorporates the size of the order into the price calculation, and therefore flat fee is applied differently depending on the size, resulting in non liner positive shift to the curve.
# Since Impermanent Loss is non linear, the flat fee eventually eventually is fully overtaken by the IL, and the curve is shifted down.


# %% ==== GAMMA HOOK) =====
class v4_gamma_hook_pool:
    """
    Represents a Uniswap V2 plus a v4 gamma protecting hook.
    """
    def __init__(self, initial_x, initial_y, fee_percentage=0.0, gamma_factor=0.0):
        v2_pool = UniswapV2Pool(initial_x, initial_y, fee_percentage)
        self.v2_pool = v2_pool
        self.gamma_factor = gamma_factor


    def run_gamma_hook(self, amount_in, amount_out, previous_x, previous_y, new_price, token_in_is_0):
        if token_in_is_0:
            print('--------------------------------')
            sr = amount_in/previous_x
            dynamic_fee_pct = sr / (1+sr)
            print(f"Dynamic fee pct: {dynamic_fee_pct}")
            amount_fee = amount_out * (dynamic_fee_pct + self.gamma_factor)
            print('Original amount out: ', amount_out)
            print('Fee qty (y (ARB)): ', amount_fee)
            amount_out_net = amount_out - amount_fee
            print('New Amount out net of fees (y (ARB)): ', amount_out_net)
            self.v2_pool.fee_x += amount_fee/2*new_price
            self.v2_pool.fee_y += amount_fee/2
            print('--------------------------------')
            print('My previous x: ', previous_x)
            print('My previous y: ', previous_y)

            self.v2_pool.x += amount_fee/2*new_price
            self.v2_pool.y += amount_fee/2


            print('My new x after fee: ', self.v2_pool.x)
            print('My new y after fee: ', self.v2_pool.y)
            print('Fee state x: ', self.v2_pool.fee_x)
            print('Fee state y: ', self.v2_pool.fee_y)

            print('Took total fee: ', amount_fee)
            print('Amount out net: ', amount_out_net)
            print('Amount out: ', amount_out)
            print('Amount in: ', amount_in)
            print('X fee: ', amount_fee/2)
            print('Y fee: ', amount_fee/2/new_price)
            print('--------------------------------')
        else:
            sr = amount_in/previous_y
            dynamic_fee_pct = sr / (1+sr)
            print(f"Dynamic fee pct: {dynamic_fee_pct}")
            amount_fee = amount_out * (dynamic_fee_pct + self.gamma_factor)
            print('Original amount out: ', amount_out)
            print('Fee qty (x (USDC)): ', amount_fee)
            amount_out_net = amount_out - amount_fee
            print('New Amount out net of fees (x (USDC)): ', amount_out_net)

            # DISTRIBUTING AND ADJUSTING FEE STATE

            self.v2_pool.fee_x += amount_fee/2
            self.v2_pool.fee_y += amount_fee/2/new_price
            print('--------------------------------')
            print('My previous x: ', previous_x)
            print('My previous y: ', previous_y)
            self.v2_pool.x += amount_fee/2
            self.v2_pool.y += amount_fee/2/new_price



            print('My new x after fee: ', self.v2_pool.x)
            print('My new y after fee: ', self.v2_pool.y)
            print('Fee state x: ', self.v2_pool.fee_x)
            print('Fee state y: ', self.v2_pool.fee_y)

            print('Took total fee: ', amount_fee)
            print('Amount out net: ', amount_out_net)
            print('Amount out: ', amount_out)
            print('Amount in: ', amount_in)
            print('X fee: ', amount_fee/2)
            print('Y fee: ', amount_fee/2/new_price)
            print('--------------------------------')
        return amount_out_net



    def swap(self, amount_in, token_in_is_0):
        """
        Performs a swap on the pool, booking fees proportionally first.
        """
        print('--------------------------------')
        print('Swapping...')
        previous_x = self.v2_pool.x
        previous_y = self.v2_pool.y
        previous_price = previous_x/previous_y
        amount_out = self.v2_pool.swap(amount_in, token_in_is_0)
        new_x, new_y, _, _ = self.v2_pool.get_balances()
        new_price = new_x/new_y
        print('Price before, after: ', previous_price, new_price)
        print('Amount in, out: ', amount_in, amount_out)
        print('X before, after: ', previous_x, new_x)
        print('Y before, after: ', previous_y, new_y)
        print('--------------------------------')
        print('Swapped...')
        amount_out_net = self.run_gamma_hook(amount_in, amount_out, previous_x, previous_y, new_price, token_in_is_0)
        print('🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵')
        print('Prevous amount_out: ', amount_out)
        print('New amount_out: ', amount_out_net)
        print('🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵🔵')

        return amount_out_net

    def get_balances(self):
        """Returns current balances (x, y, fees_x, fees_y)."""
        return self.v2_pool.get_balances()
        
    def reset(self):
        """Resets the pool to the initial state."""
        self.v2_pool.reset()
        self.v2_pool.fee_x = 0
        self.v2_pool.fee_y = 0
        

relative_amounts_in = np.linspace(-max_relative_amount+0.0001, max_relative_amount, num_points) # 0.01% to max%


gamma_pool = v4_gamma_hook_pool(initial_x_single, initial_y_single, gamma_factor)

print(f"Starting single swap simulation with Dynamic Fee")
# --- Run Simulation ---
swap_sweep_results_v4hook = sweep_swap_amount_range(
    pool_instance=gamma_pool,
    initial_x=initial_x_single,
    initial_y=initial_y_single,
    relative_amounts=relative_amounts_in,
    fee_tier=0.0
)

print(f"Generated {len(swap_sweep_results_v4hook)} single swap results.")
print(swap_sweep_results_v4hook.head())

# --- Plot Results ---
hook_swap_sweep_plot = plot_single_swap_payoff(swap_sweep_results_v4hook, fee_tier='Hook')
hook_swap_sweep_plot.show()
# %% FULL RANGE SWEEP 


initial_x_single = float(500) # 1 million USDC
initial_y_single = float(5)   # 1 million ARB
num_points = 200 # Number of relative amounts to test
max_relative_amount = 0.99 # Test swaps up to 50% of reserve
v2_pool_fee_tier = 0.03
hook_fee_tier = 0.0
gamma_factor = 0.5

relative_amounts_in = np.linspace(-max_relative_amount+0.000000001, max_relative_amount, num_points) # 0.01% to max%

v2_pool_0_fee = UniswapV2Pool(initial_x_single, initial_y_single, fee_percentage=0.0)
v2_swap_sweep_results_0_fee_df = sweep_swap_amount_range(
    pool_instance=v2_pool_0_fee,
    initial_x=initial_x_single,
    initial_y=initial_y_single,
    relative_amounts=relative_amounts_in,
    fee_tier=0.0
)

v2_pool = UniswapV2Pool(initial_x_single, initial_y_single, fee_percentage=single_swap_fee_tier)
v2_swap_sweep_results_flat_fee_df = sweep_swap_amount_range(
    pool_instance=v2_pool,
    initial_x=initial_x_single,
    initial_y=initial_y_single,
    relative_amounts=relative_amounts_in,
    fee_tier=v2_pool_fee_tier
)

v4_pool = v4_gamma_hook_pool(initial_x_single, initial_y_single, gamma_factor)
v4_gamma_hook_swap_sweep_results = sweep_swap_amount_range(
    pool_instance=v4_pool,
    initial_x=initial_x_single,
    initial_y=initial_y_single,
    relative_amounts=relative_amounts_in,
    fee_tier=hook_fee_tier
)


fig = go.Figure()


fig.add_trace(go.Scatter(
    x=v2_swap_sweep_results_0_fee_df['new_price'],
    y=v2_swap_sweep_results_0_fee_df['lp_value'],
    mode='markers',
    name='V2 Pool (fee: 0.0%)',
    marker=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x=v2_swap_sweep_results_flat_fee_df['new_price'],
    y=v2_swap_sweep_results_flat_fee_df['lp_value'],
    mode='markers',
    name='V2 Pool (fee: 0.3%)',
    marker=dict(color='red')
))

fig.add_trace(go.Scatter(
    x=v4_gamma_hook_swap_sweep_results['new_price'],
    y=v4_gamma_hook_swap_sweep_results['lp_value'],
    mode='markers',
    name='V4 Gamma Hook (fee: dynamic)',
    marker=dict(color='mediumseagreen')
))

fig.add_trace(go.Scatter(
    x=v2_swap_sweep_results_0_fee_df['new_price'],
    y=v2_swap_sweep_results_0_fee_df['bh_value'],
    mode='markers',
    name='Buy & Hold',
    marker=dict(color='black')
))

price_range = max( v4_gamma_hook_swap_sweep_results['new_price'].max(), v2_swap_sweep_results_0_fee_df['new_price'].max(), v2_swap_sweep_results_flat_fee_df['new_price'].max())

fig.update_layout(
    title='V2 Pool vs V4 Pool Gamma Hook',
    xaxis_title='Price (ARB/USDC)',
    yaxis_title='Portfolio Value (USDC)',
    legend_title='Profile',
    xaxis_range=[0, price_range]
)

fig.show()


# %% PLOT FINAL RESERVE BALANCES (X vs Y)

def plot_reserve_balances(df_0fee_df_v2, df_flatfee_df_v2, df_v4hook):
    """
    Plots the final x and y reserve balances for V2 and V4 hook simulations.
    """
    fig = go.Figure()

    # Plot V2 Results
    fig.add_trace(go.Scatter(
        x=df_0fee_df_v2['new_x'],
        y=df_0fee_df_v2['new_y'],
        mode='markers',
        name=f'V2 Pool - Fee 0%',
        marker=dict(color='coral', size=6)
    ))

    # Plot V4 Hook Results
    fig.add_trace(go.Scatter(
        x=df_flatfee_df_v2['new_x'],
        y=df_flatfee_df_v2['new_y'],
        mode='markers',
        name='V2 Pool - Flat Fee',
        marker=dict(color='mediumseagreen', size=6)
    ))

    fig.add_trace(go.Scatter(
        x=df_v4hook['new_x'],
        y=df_v4hook['new_y'],
        mode='markers',
        name='V4 Gamma Hook (fee: dynamic)',
        marker=dict(color='blue', size=6)
    ))

    # Add hover text details
    fig.update_traces(hovertemplate=
        "<b>Final X (USDC):</b> %{x:,.2f}<br>" +
        "<b>Final Y (ARB):</b> %{y:,.2f}<br>" +
        "Relative Amount In: %{customdata[0]:.2%}<extra></extra>",
        customdata=df_0fee_df_v2[['relative_amount_in']] # Assuming both dfs have same rel amounts
    )

    # Determine plot range dynamically
    min_x = min(df_0fee_df_v2['new_x'].min(), df_flatfee_df_v2['new_x'].min(), df_v4hook['new_x'].min()) * 0.98
    max_x = max(df_0fee_df_v2['new_x'].max(), df_flatfee_df_v2['new_x'].max(), df_v4hook['new_x'].max()) * 1.02
    min_y = min(df_0fee_df_v2['new_y'].min(), df_flatfee_df_v2['new_y'].min(), df_v4hook['new_y'].min()) * 0.98
    max_y = max(df_0fee_df_v2['new_y'].max(), df_flatfee_df_v2['new_y'].max(), df_v4hook['new_y'].max()) * 1.02

    fig.update_layout(
        title='Final Pool Reserve Balances (X vs Y) after Single Swaps',
        xaxis_title='Final X Reserve (USDC)',
        yaxis_title='Final Y Reserve (ARB)',
        legend_title='Pool Type',
        hovermode='closest',
        xaxis_range=[min_x, max_x],
        yaxis_range=[min_y, max_y],
        width=700,
        height=700
    )

    # Optionally add the constant product curve k = x*y for the initial state
    # initial_k = initial_x_single * initial_y_single
    # x_curve = np.linspace(min_x, max_x, 200)
    # y_curve = initial_k / x_curve
    # fig.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', name='Initial k Curve', line=dict(dash='dot', color='grey')))

    return fig

# --- Generate and Show Plot ---
# Ensure the results dataframes exist from the previous cell
fig_balances = plot_reserve_balances(v2_swap_sweep_results_0_fee_df, v2_swap_sweep_results_flat_fee_df, v4_gamma_hook_swap_sweep_results)
fig_balances.show()





# %% PLOT NEW PRICE AS FUNCTION OF RELATIVE AMOUNT

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=v2_swap_sweep_results_0_fee_df['relative_amount_in'],
    y=v2_swap_sweep_results_0_fee_df['new_price'],
    mode='markers',
    name='V2 Pool - Fee 0%',
    marker=dict(color='blue', size=6)
))

fig.add_trace(go.Scatter(
    x=v2_swap_sweep_results_flat_fee_df['relative_amount_in'],
    y=v2_swap_sweep_results_flat_fee_df['new_price'],
    mode='markers',
    name='V2 Pool - Flat Fee',
    marker=dict(color='red', size=6)
))

fig.add_trace(go.Scatter(
    x=v4_gamma_hook_swap_sweep_results['relative_amount_in'],
    y=v4_gamma_hook_swap_sweep_results['new_price'],
    mode='markers',
    name='V4 Gamma Hook - Dynamic Fee',
    marker=dict(color='mediumseagreen', size=6)
))

fig.update_layout(
    title='Price vs. Relative Amount In',
    xaxis_title='Relative Amount In',
    yaxis_title='Price (ARB/USDC)',
    legend_title='Pool Type'
)

fig.show()



#%% PLOT RELATIVE AMOUNT VS SWAP PRICE
import plotly.express as px

v2_swap_sweep_results_0_fee_df['pool_type'] = 'V2 Pool - 0%'
v2_swap_sweep_results_flat_fee_df['pool_type'] = 'V2 Pool - Flat Fee'
v4_gamma_hook_swap_sweep_results['pool_type'] = 'V4 Gamma Hook - Dynamic'

joint_df = pd.concat([v2_swap_sweep_results_0_fee_df, v2_swap_sweep_results_flat_fee_df, v4_gamma_hook_swap_sweep_results])

# Use plotly.express for direct color mapping






# %% PLOT SWAP PRICE VS RELATIVE AMOUNT
fig = px.scatter(
    joint_df,
    x='relative_amount_in',
    y='swap_price',
    color='pool_type',
    title='Swap Price vs. Relative Amount In'
)

fig.show()
# %%

