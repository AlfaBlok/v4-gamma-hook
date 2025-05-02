import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from src.models.v2_pool import UniswapV2Pool
from src.models.v4_gamma_pool import v4_gamma_hook_pool
import os # Make sure os is imported if not already

def sweep_swap_amount_range(pool_instance, parameters):
    """
    Simulates single swaps for a range of order sizes expressed relative to the reserve balance.
    This function helps test the behaviour of a single swap for a wide range of potential incoming order sizes.

    Args:
        pool_instance (UniswapV2Pool): An initialized pool object.
        parameters (dict): A dictionary containing the following parameters:
            - initial_x (float): Initial reserve of token X (USDC).
            - initial_y (float): Initial reserve of token Y (ARB).
            - relative_amounts (np.ndarray): Array of relative trade sizes (e.g., 0.01 for 1%).
            - fee_pct (float): The fee percentage used for the pool.

    Returns:
        pd.DataFrame: DataFrame containing simulation results.
    """
    results = []
    max_relative_amount = parameters['max_relative_amount']
    num_points = parameters['num_points']
    fee_pct = parameters['fee_pct']
    initial_x = parameters['initial_x']
    initial_y = parameters['initial_y']
    # fee_percentage = parameters['fee_pct']

    pool_instance.fee_percentage = fee_pct # Ensure pool has the correct fee
    relative_amounts = np.linspace(-max_relative_amount+0.00001, max_relative_amount, num_points) # 0.01% to max%

    initial_price_usdc = initial_x / initial_y

    for rel_amount in relative_amounts:
        pool_instance.reset() # Start fresh for each relative amount
        # pool_instance.fee_percentage = fee_pct # Ensure pool has the correct fee

        if rel_amount > 0: # Positive means USDC is coming in, ARB is going out
            amount_in = rel_amount * initial_x  
            order_size_usdc = amount_in
            is_usdc_in = True
        else: # Negative means ARB is coming in, USDC is going out
            amount_in = -rel_amount * initial_y
            order_size_usdc = -amount_in*initial_price_usdc
            is_usdc_in = False

        if amount_in <= 1e-9 and amount_in >= 0: # Skip negligible or zero swaps
            continue

        try:
            # is_usdc_in = True if amount_in > 0 else False
            # amount_in = abs(amount_in)
            previous_price = pool_instance.get_price()
            amount_out = pool_instance.swap(abs(amount_in), is_usdc_in)
            new_x, new_y, fee_x_total, fee_y_total = pool_instance.get_balances()

            # Calculate metrics AFTER the swap
            if new_x > 1e-12:
                new_price = new_x / new_y # Price = ARB / USDC
            else:
                new_price = np.nan # Avoid division by zero
            
            lp_value_usdc = new_x + new_y * new_price 
            hodl_value_usdc = initial_x + initial_y * new_price



            if np.isnan(lp_value_usdc):
                 print(f"Warning: NaN value calculated for rel_amount={rel_amount}, new_price={new_price}")
                 continue # Skip if calculation failed

            if rel_amount < 0:
                # amount_in = amount_in*initial_price_usdc
                swap_price_usdc = amount_out/amount_in

            if rel_amount >= 0:
                swap_price_usdc = amount_in/amount_out

            impermanent_loss_usdc = lp_value_usdc - (initial_x + initial_y * new_price)
            if is_usdc_in:
                IL_O = abs(impermanent_loss_usdc) / amount_in
                impermanent_loss_pct_hodl = abs(impermanent_loss_usdc) / hodl_value_usdc
            else:
                IL_O = abs(impermanent_loss_usdc) / amount_out
                impermanent_loss_pct_hodl = impermanent_loss_usdc / hodl_value_usdc

            x_fee = pool_instance.last_trade_fee_x
            y_fee = pool_instance.last_trade_fee_y

            total_fee_usdc = x_fee + y_fee * previous_price
            total_fee_pct_order_size =total_fee_usdc / order_size_usdc

            results.append({
                'relative_amount_in': rel_amount,
                'amount_in': amount_in,
                'amount_out': amount_out,
                'token_in_is_USDC': is_usdc_in,
                'new_x': new_x,
                'new_y': new_y,
                'new_price': new_price, # ARB/USDC
                'lp_value': lp_value_usdc,
                'swap_price': swap_price_usdc,
                'pool_name': pool_instance.name,
                'order_size_usdc': order_size_usdc,
                'impermanent_loss_usdc': impermanent_loss_usdc,
                'impermanent_loss_pct_hodl': impermanent_loss_pct_hodl,
                'IL_O': IL_O,
                'last_trade_fee_x': pool_instance.last_trade_fee_x,
                'last_trade_fee_y': pool_instance.last_trade_fee_y,
                'total_fee_usdc': total_fee_usdc,
                'total_fee_pct': total_fee_pct_order_size
            })

        except ValueError as e:
            print(f"Swap failed for relative amount {rel_amount}: {e}")
            continue


    results_df = pd.DataFrame(results)

    # Ensure results_df is sorted if necessary for meaningful diff
    # results_df = results_df.sort_values(by='relative_amount_in').reset_index(drop=True) 
    # -> It seems relative_amounts is already sorted, so sorting might be redundant but safe

    # Calculate Delta and Gamma on the DataFrame
    if not results_df.empty and 'new_price' in results_df.columns and 'lp_value' in results_df.columns:
        # Calculate differences
        d_lp_value = results_df['lp_value'].diff()
        d_new_price = results_df['new_price'].diff()

        # Delta = d(lp_value) / d(new_price)
        # Avoid division by zero: replace 0 in d_new_price with NaN before division
        delta = d_lp_value / d_new_price.replace(0, np.nan)

        # Gamma = d(Delta) / d(new_price)
        d_delta = delta.diff()
        gamma = d_delta / d_new_price.replace(0, np.nan)

        results_df['delta'] = delta
        results_df['gamma'] = gamma
    else:
        # Handle empty DataFrame or missing column case
        results_df['delta'] = np.nan
        results_df['gamma'] = np.nan

    return results_df

def get_symbol(pool_name):
    if 'v2_pool_f=0.0' in pool_name:
        return 'circle' # Filled circle
    elif 'v2_pool_f' in pool_name: # Catch other V2 fees
        return 'square' # Filled square
    elif 'v4_gamma_pool_g=0.0' in pool_name:
        return 'diamond' # Filled diamond
    elif 'v4_gamma_pool' in pool_name: # Catch other V4 gamma
        return 'cross' # Filled cross
    elif 'buy_and_hold' in pool_name:
        return 'x' # Filled x
    else:
        return 'circle' # Default filled circle

def get_xy_fig(joint_df, x_col, y_col, title, xaxis_title=None, yaxis_title=None):

    pool_names = joint_df['pool_name'].unique()
    colors = px.colors.qualitative.Plotly # Use Plotly Express's default color sequence
    symbol_map = {name: get_symbol(name) for name in pool_names}
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(pool_names)}

    fig = go.Figure()

    for pool_name in joint_df['pool_name'].unique():
        pool_df = joint_df[joint_df['pool_name'] == pool_name]

        if pool_name == 'buy_and_hold':
            fig.add_trace(go.Scatter(
                x=pool_df[x_col],
                y=pool_df[y_col],
                mode='lines',
                name=pool_name,
                line=dict( # Use line dictionary for line styling
                    color='black', 
                    width=2,
                    dash='dash'
                )
            ))
        else:


            fig.add_trace(go.Scatter(
                x=pool_df[x_col],
                y=pool_df[y_col],
                mode='markers',
                name=pool_name,
                marker=dict(
                    color=color_map[pool_name], # Outline color
                    symbol=symbol_map[pool_name],
                    size=7 # Make symbols a bit larger
                )
        ))

    if xaxis_title is None:
        xaxis_title = x_col
    if yaxis_title is None:
        yaxis_title = y_col

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend_title=''
    )

    return fig




def get_3_plots(trade_simulation, title):
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
        title_text=title,
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

def get_arb_usdc_historical_data():
    # Construct absolute path relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    path = os.path.join(project_root, "data", "swap_fee_data.csv")
    print(f"Reading data from: {path}") # Optional: print path for verification

    df = pd.read_csv(path)

    df[['reserve_amount_0', 'reserve_amount_1']] = df['reserveAmounts'].str.split(',', expand=True)

    # Remove brackets and quotes
    df['reserve_amount_arb'] = (df['reserve_amount_0'].str.replace(r"[\[\]']", '', regex=True))
    df['reserve_amount_usdc'] = (df['reserve_amount_1'].str.replace(r"[\[\]']", '', regex=True))

    df['reserve_amount_arb'] = df['reserve_amount_arb'].astype(float)
    df['reserve_amount_usdc'] = df['reserve_amount_usdc'].astype(float)

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

    return df


def get_usdc_distribution():
    df = get_arb_usdc_historical_data()

    usdc_pdf = np.histogram(df['usdc_amount_vs_reserve'], bins=1000, density=True)
    usdc_pdf_bins = usdc_pdf[1]
    usdc_pdf_bin_midpoints = (usdc_pdf_bins[:-1] + usdc_pdf_bins[1:]) / 2

    usdc_pdf_density = usdc_pdf[0]

    # Calculate USDC CDF
    usdc_bin_widths = np.diff(usdc_pdf_bins)
    usdc_cdf_vals = np.cumsum(usdc_pdf_density * usdc_bin_widths)
    usdc_cdf_vals = np.insert(usdc_cdf_vals, 0, 0)
    usdc_cdf_x_values = usdc_pdf_bins

    return usdc_cdf_vals, usdc_cdf_x_values



def get_arb_distribution():
    df = get_arb_usdc_historical_data()
    # We generate the PDF from the historical data
    arb_pdf = np.histogram(df['arb_amount_vs_reserve'], bins=1000, density=True)
    arb_pdf_bins = arb_pdf[1]
    arb_pdf_bin_midpoints = (arb_pdf_bins[:-1] + arb_pdf_bins[1:]) / 2
    arb_pdf_density = arb_pdf[0]

    # Calculate ARB CDF (Cumulative sum of Density * Bin Width)
    arb_bin_widths = np.diff(arb_pdf_bins) # Width of each bin
    # CDF value at the right edge of bin i is sum(density[j]*width[j] for j<=i)
    arb_cdf_vals = np.cumsum(arb_pdf_density * arb_bin_widths)
    # Add 0 at the beginning for the CDF value at the first edge
    arb_cdf_vals = np.insert(arb_cdf_vals, 0, 0) 
    arb_cdf_x_values = arb_pdf_bins # CDF x-values are the bin edges

    return   arb_cdf_vals, arb_cdf_x_values


def generate_arb_trades(num_trades):
    """Generates synthetic trades using inverse transform sampling and linear interpolation from historical swap data for ARB:USDC pool in Arbitrum L2 as example"""

    arb_cdf_vals, arb_cdf_x_values = get_arb_distribution()

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


# Function to generate synthetic USDC trades
def generate_usdc_trades(num_trades):
    """Generates synthetic trades using inverse transform sampling with linear interpolation."""
    usdc_cdf_vals, usdc_cdf_x_values = get_usdc_distribution()

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



# Function to create a sequence of trades with running reserves using UniswapV2Pool class
def simulate_trade_sequence(pool, arb_rel_amounts, usdc_rel_amounts, parameters):
    """
    Simulates a sequence of trades using synthetic samples and the UniswapV2Pool class,
    updating reserves after each trade.

    Args:
        arb_rel_amounts (np.ndarray): Numpy array of synthetic relative ARB amounts.
        usdc_rel_amounts (np.ndarray): Numpy array of synthetic relative USDC amounts.
        parameters (dict): A dictionary containing the following parameters:
            - initial_x (float): Initial reserve of token X (USDC).
            - initial_y (float): Initial reserve of token Y (ARB).
            - fee_pct (float): Fee percentage (default 0.003 for 0.3%).

    Returns:
        pd.DataFrame: DataFrame with trade inputs and outputs.
    """

    initial_x = parameters['initial_x']
    initial_y = parameters['initial_y']
    fee_pct = parameters['fee_pct']

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
        fee_x_trade = pool.last_trade_fee_x
        fee_y_trade = pool.last_trade_fee_y

        pool_name = pool.name
        new_price = reserve_x_after / reserve_y_after
        hodl_value_usdc = pool.initial_x + pool.initial_y * new_price
        pool_value_usdc = reserve_x_after + reserve_y_after * new_price
        
        impermanent_loss = pool_value_usdc-hodl_value_usdc 
        impermanent_loss_pct = impermanent_loss / hodl_value_usdc

        fee_pct = pool.last_fee_pct
        
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
            'fee_pct': fee_pct,
            'pool_name': pool_name,
            'price': new_price,
            'hodl_value_usdc': hodl_value_usdc,
            'pool_value_usdc': pool_value_usdc,
            'impermanent_loss': impermanent_loss,
            'impermanent_loss_pct': impermanent_loss_pct
        }
        results.append(trade_data)


    trade_simulation = pd.DataFrame(results)



    return trade_simulation 

def get_bh_df(pool_instance, df):
    initial_x = pool_instance.initial_x
    initial_y = pool_instance.initial_y



    buy_and_hold_df = pd.DataFrame({
        'relative_amount_in': df['relative_amount_in'],
        'new_price': df['new_price'],
        'lp_value': initial_x + initial_y * df['new_price'],
        'pool_name': 'buy_and_hold'
    })

    d_lp_value = buy_and_hold_df['lp_value'].diff()
    d_new_price = buy_and_hold_df['new_price'].diff()

        # Delta = d(lp_value) / d(new_price)
        # Avoid division by zero: replace 0 in d_new_price with NaN before division
    delta = d_lp_value / d_new_price.replace(0, np.nan)

        # Gamma = d(Delta) / d(new_price)
    d_delta = delta.diff()
    gamma = d_delta / d_new_price.replace(0, np.nan)

    buy_and_hold_df['delta'] = delta
    buy_and_hold_df['gamma'] = gamma


    return buy_and_hold_df


def run_simulation(parameters):
    arb_samples = (generate_arb_trades(int(parameters['num_trades_to_simulate']*2.5)))
    arb_samples = arb_samples[arb_samples > 0] # I don't know how many this will be, but it's double the number of trades
    usdc_samples = (generate_usdc_trades(int(parameters['num_trades_to_simulate']*2.5)))
    usdc_samples = usdc_samples[usdc_samples > 0]

    num_plot_trades = int(parameters['num_trades_to_simulate']) # Keep this smaller for faster initial testing/plotting if needed - also only positives showing
    arb_samples_run = arb_samples[:num_plot_trades]
    usdc_samples_run = usdc_samples[:num_plot_trades]

    pool = UniswapV2Pool(parameters, is_zero_fee=True)
    trade_simulation_0_fee = simulate_trade_sequence(
        pool,
        arb_samples_run,
        usdc_samples_run,
        parameters
    )

    pool = UniswapV2Pool(parameters, is_zero_fee=False)
    trade_simulation_f_fee = simulate_trade_sequence(
        pool,
        arb_samples_run,
        usdc_samples_run,
        parameters
    )

    pool = v4_gamma_hook_pool(parameters, is_zero_gee=True)
    trade_simulation_v4_0_gee = simulate_trade_sequence(
        pool,
        arb_samples_run,
        usdc_samples_run,
        parameters
    )

    pool = v4_gamma_hook_pool(parameters, is_zero_gee=False)
    trade_simulation_v4_g_gee = simulate_trade_sequence(
        pool,
        arb_samples_run,
        usdc_samples_run,
        parameters
    )

    joint_df = pd.concat([
        trade_simulation_0_fee, 
        trade_simulation_f_fee, 
        trade_simulation_v4_0_gee, 
        trade_simulation_v4_g_gee, 
        ])
    
    return joint_df

