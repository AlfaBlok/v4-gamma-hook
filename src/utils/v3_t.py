#%%
import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Project root added to sys.path: {project_root}")

from src.utils.clean import sweep_swap_amount_range, get_xy_fig
# from src.models.v3_pool import UniswapV3Pool

class UniswapV3Pool:
    def __init__(self, params):
        token0_initial = params['token0']
        token1_initial = params['token1']
        price_low = params['price_low']
        price_high = params['price_high']
        current_price = params['current_price']
        fee = params['fee']
        self.sqrt_P = current_price ** 0.5
        self.sqrt_P_lower = price_low ** 0.5
        self.sqrt_P_upper = price_high ** 0.5
        self.fee = fee

        # Compute liquidity based on provided amounts and range
        # Handle potential division by zero or invalid sqrt prices
        delta_0 = (1 / self.sqrt_P - 1 / self.sqrt_P_upper) if self.sqrt_P > 1e-12 and self.sqrt_P_upper > 1e-12 else float('inf')
        delta_1 = (self.sqrt_P - self.sqrt_P_lower)

        L0 = token0_initial / delta_0 if delta_0 > 1e-12 else float('inf')
        L1 = token1_initial / delta_1 if delta_1 > 1e-12 else float('inf')
        self.L = min(L0, L1)

        # Initialize reserves based on the *calculated* liquidity L
        self.token0 = self.L * delta_0 if delta_0 != float('inf') else 0
        self.token1 = self.L * delta_1 if delta_1 != float('inf') else 0

        # Store the actual initial amounts used by the pool for B&H comparison
        self.initial_effective_token0 = self.token0
        self.initial_effective_token1 = self.token1

    def get_value_at_price(self, price):
        """Calculates the theoretical value of the position at a given price."""
        sqrt_P = price ** 0.5
        # Clamp sqrt_P within the position's range
        clamped_sqrt_P = max(self.sqrt_P_lower, min(sqrt_P, self.sqrt_P_upper))

        # Calculate theoretical reserves at the clamped price
        theoretical_token0 = self.L * (1 / clamped_sqrt_P - 1 / self.sqrt_P_upper) if clamped_sqrt_P > 1e-12 else 0
        theoretical_token1 = self.L * (clamped_sqrt_P - self.sqrt_P_lower)

        # Value is calculated using the *input* price (not clamped), reflecting market value
        value = theoretical_token0 * price + theoretical_token1
        return value

    def price(self):
        # Returns the price of token1 in terms of token0 (token1/token0)
        return self.sqrt_P ** 2

    def swap(self, amount_in, token_in_is_0):
        """
        Performs a swap on the pool, respecting concentrated liquidity range.

        Args:
            amount_in (float): Amount of the input token.
            token_in_is_0 (bool): True if input is token0 (ETH), False if input is token1 (USDC).

        Returns:
            float: Amount of the output token received.
        """
        amount_in_eff = amount_in * (1 - self.fee)
        if amount_in_eff <= 1e-12 or self.L <= 1e-12: # Avoid issues with tiny amounts or liquidity
            return 0

        initial_sqrt_P = self.sqrt_P
        final_sqrt_P = initial_sqrt_P # Initialize final price

        if token_in_is_0: # Selling ETH (token0) for USDC (token1) -> Price Increases
            # Calculate the theoretical target sqrt_P if liquidity were infinite
            # delta_X = L * (1/sqrtP - 1/sqrtP_next) => 1/sqrtP_next = 1/sqrtP - delta_X / L
            try:
                term = 1 / initial_sqrt_P - amount_in_eff / self.L
                if term <= 1e-12: # Consumes all token1 liquidity in range
                    target_sqrt_P = self.sqrt_P_upper
                else:
                    target_sqrt_P = 1 / term
            except (OverflowError, ZeroDivisionError):
                 target_sqrt_P = self.sqrt_P_upper # Treat overflow/zero division as consuming all token1

            # Clamp the final price to the position's range
            final_sqrt_P = min(target_sqrt_P, self.sqrt_P_upper)
            # Ensure price doesn't decrease when adding token0
            final_sqrt_P = max(final_sqrt_P, initial_sqrt_P)

            # Calculate amount out based on the actual price movement within the range
            token1_out = self.L * (final_sqrt_P - initial_sqrt_P)

            # Update state: Price and Reserves
            self.sqrt_P = final_sqrt_P
            # Recalculate reserves based on the NEW price and constant L
            # Protect against division by zero if sqrt_P becomes zero (shouldn't happen here)
            self.token0 = self.L * (1 / self.sqrt_P - 1 / self.sqrt_P_upper) if self.sqrt_P > 1e-12 else 0
            self.token1 = self.L * (self.sqrt_P - self.sqrt_P_lower)

            return max(0, token1_out)

        else: # Selling USDC (token1) for ETH (token0) -> Price Decreases
            # Calculate the theoretical target sqrt_P if liquidity were infinite
            # delta_Y = L * (sqrtP_next - sqrtP) => sqrtP_next = sqrtP + delta_Y / L
            # NOTE: Formula uses delta_Y = token1_in. Our input is token1. Price *decreases*.
            # sqrtP_next = initial_sqrt_P - amount_in_eff / self.L (Mistake in formula derivation often seen)
            # Correct: amount_token1_in = L * (sqrtP_initial - sqrtP_final) -> sqrtP_final = sqrtP_initial - amount_token1_in / L

            target_sqrt_P = initial_sqrt_P - amount_in_eff / self.L

            # Clamp the final price to the position's range
            final_sqrt_P = max(target_sqrt_P, self.sqrt_P_lower)
             # Ensure price doesn't increase when adding token1
            final_sqrt_P = min(final_sqrt_P, initial_sqrt_P)

            # Calculate amount out based on the actual price movement within the range
            # delta_X = L * (1/sqrtP_final - 1/sqrtP_initial)
            # Protect against division by zero if final_sqrt_P hits lower bound exactly (or is zero)
            if final_sqrt_P <= 1e-12:
                 # If price goes to zero (or below lower bound), calculate max token0 out
                 token0_out = self.L * (1 / self.sqrt_P_lower - 1 / initial_sqrt_P) if initial_sqrt_P > 1e-12 else 0
                 final_sqrt_P = self.sqrt_P_lower # Clamp price to lower bound
            else:
                 token0_out = self.L * (1 / final_sqrt_P - 1 / initial_sqrt_P) if initial_sqrt_P > 1e-12 else 0

            # Update state: Price and Reserves
            self.sqrt_P = final_sqrt_P
             # Recalculate reserves based on the NEW price and constant L
            self.token0 = self.L * (1 / self.sqrt_P - 1 / self.sqrt_P_upper) if self.sqrt_P > 1e-12 else 0
            self.token1 = self.L * (self.sqrt_P - self.sqrt_P_lower)

            return max(0, token0_out)

    def state(self):
        return {
            "price (token1 per token0)": self.price(),
            "sqrt_price": self.sqrt_P,
            "token0": self.token0,
            "token1": self.token1,
            "liquidity (internal)": self.L
        }

# Store initial parameters for buy & hold calculation
initial_params = {
    'token0': 500, # ETH
    'token1': 500, # USDC
    'price_low': 0.5, # Price range lower bound (USDC per ETH)
    'price_high': 1.5, # Price range upper bound (USDC per ETH)
    'current_price': 1, # Initial price: token1 / token0 (e.g., 10000 USDC / 100 ETH = 100)
    'fee': 0.0
}

# Initial amounts for buy & hold calculation
initial_token0 = initial_params['token0'] # Initial ETH
initial_token1 = initial_params['token1'] # Initial USDC
initial_price_usdc_per_eth = initial_params['current_price']

# Define swap *value* range in terms of USDC
swap_value_range_usdc = np.arange(0, 50000, 50) # Swapping value from 0 up to 5000 USDC

# Calculate corresponding token amounts for the swap ranges
# Swap token0 in (selling ETH): amount of ETH to sell
swap_amount_range_t0_in = swap_value_range_usdc / initial_price_usdc_per_eth
# Swap token1 in (selling USDC): amount of USDC to sell
swap_amount_range_t1_in = swap_value_range_usdc

results = []
# Simulate swapping token0 in (selling ETH, buying USDC)
for amount_eth_in in swap_amount_range_t0_in:
    if amount_eth_in == 0: continue # Skip zero swap
    pool = UniswapV3Pool(initial_params) # Reset pool for each swap simulation
    usdc_out = pool.swap(amount_in=amount_eth_in, token_in_is_0=True)
    new_price_usdc_per_eth = pool.price() # Price = token1/token0

    # Value in terms of token1 (USDC)
    lp_value_usdc = pool.token0 * new_price_usdc_per_eth + pool.token1
    bh_value_usdc = pool.initial_effective_token0 * new_price_usdc_per_eth + pool.initial_effective_token1
    # Theoretical LP value at the new price
    theoretical_lp_value_usdc = pool.get_value_at_price(new_price_usdc_per_eth)

    # Calculate the USDC value of the ETH sold, using the initial price for the x-axis
    swap_value_usdc = -amount_eth_in * initial_price_usdc_per_eth # Negative value for selling ETH

    results.append(
        {
            "swap_value_usdc": swap_value_usdc,
            "eth_out": 0, # Selling ETH
            "usdc_out": usdc_out, # Buying USDC
            "sqrt_price": pool.sqrt_P,
            "eth_reserve": pool.token0,
            "usdc_reserve": pool.token1,
            'pool_name': 'v3_pool',
            'price_usdc_per_eth': new_price_usdc_per_eth,
            'lp_value_usdc': lp_value_usdc,
            'bh_value_usdc': bh_value_usdc,
            'theoretical_lp_value_usdc': theoretical_lp_value_usdc,
        }
    )

# Simulate swapping token1 in (selling USDC, buying ETH)
for amount_t1_in in swap_amount_range_t1_in:
    pool = UniswapV3Pool(initial_params) # Reset pool
    eth_out = pool.swap(amount_in=amount_t1_in, token_in_is_0=False)
    new_price_usdc_per_eth = pool.price() # Price = token1/token0

    # Value in terms of token1 (USDC)
    lp_value_usdc = pool.token0 * new_price_usdc_per_eth + pool.token1
    bh_value_usdc = pool.initial_effective_token0 * new_price_usdc_per_eth + pool.initial_effective_token1
    # Theoretical LP value at the new price
    theoretical_lp_value_usdc = pool.get_value_at_price(new_price_usdc_per_eth)

    # The swap value is simply the amount of USDC sold
    swap_value_usdc = amount_t1_in # Positive value for selling USDC

    results.append(
        {
            "swap_value_usdc": swap_value_usdc,
            "eth_out": eth_out, # Buying ETH
            "usdc_out": 0, # Selling USDC
            "sqrt_price": pool.sqrt_P,
            "eth_reserve": pool.token0,
            "usdc_reserve": pool.token1,
            'pool_name': 'v3_pool',
            'price_usdc_per_eth': new_price_usdc_per_eth,
            'lp_value_usdc': lp_value_usdc,
            'bh_value_usdc': bh_value_usdc,
            'theoretical_lp_value_usdc': theoretical_lp_value_usdc,
        }
    )

df = pd.DataFrame(results).sort_values(by="swap_value_usdc").reset_index(drop=True)

# --- Plotting ---
# Plot amount out vs swap value
fig_usdc_out = get_xy_fig(df, "swap_value_usdc", "usdc_out", "USDC Out vs. Swap Value (Negative = Sell ETH)")
fig_eth_out = get_xy_fig(df, "swap_value_usdc", "eth_out", "ETH Out vs. Swap Value (Positive = Sell USDC)")
fig_usdc_out.show()
fig_eth_out.show()


# Plot price vs swap value
fig_price = get_xy_fig(df, "swap_value_usdc", "price_usdc_per_eth", "Price (USDC per ETH) vs. Swap Value (USDC)")
fig_price.show()

# Plot reserves curve
fig_reserves = get_xy_fig(df, "eth_reserve", "usdc_reserve", "ETH vs. USDC Reserves")
fig_reserves.show()

# Plot LP value vs price
fig_lp_value = get_xy_fig(df, "price_usdc_per_eth", "lp_value_usdc", "LP Value (USDC) vs. Price (USDC per ETH)")
fig_lp_value.show()

# Plot LP value vs B&H value
fig_lp_vs_bh = go.Figure()
fig_lp_vs_bh.add_trace(go.Scatter(x=df["price_usdc_per_eth"], y=df["lp_value_usdc"], mode='lines', name='LP Value (Simulated)'))
fig_lp_vs_bh.add_trace(go.Scatter(x=df["price_usdc_per_eth"], y=df["bh_value_usdc"], mode='lines', name='Buy & Hold Value (Effective Start)'))
fig_lp_vs_bh.add_trace(go.Scatter(x=df["price_usdc_per_eth"], y=df["theoretical_lp_value_usdc"], mode='lines', name='LP Value (Theoretical)', line=dict(dash='dash')))
fig_lp_vs_bh.update_layout(
    title="LP Value vs Buy&Hold Value vs Price (USDC per ETH)",
    xaxis_title="Price (USDC per ETH)",
    yaxis_title="Value (USDC)"
)
fig_lp_vs_bh.show()

# %%
df.head(30)
# %%
def lp_value_token1(P, L, P_min, P_max):
    """
    Compute the value of a Uniswap V3 LP position in terms of token1.

    Parameters:
        P      : float - current price (token1 per token0)
        L      : float - liquidity
        P_min  : float - lower bound of price range
        P_max  : float - upper bound of price range

    Returns:
        float - value of the position in token1
    """
    sqrtP = P ** 0.5
    sqrtP_min = P_min ** 0.5
    sqrtP_max = P_max ** 0.5

    if P <= P_min:
        return L * (1 / sqrtP_min - 1 / sqrtP_max) * P
    elif P < P_max:
        return L * ((1 / sqrtP - 1 / sqrtP_max) * P + (sqrtP - sqrtP_min))
    else:
        return L * (sqrtP_max - sqrtP_min)
    
# %%

# Define your price bounds (use the ones from your pool setup)
P_min = initial_params['price_low']
P_max = initial_params['price_high']
sqrt_P_min = np.sqrt(P_min)
sqrt_P_max = np.sqrt(P_max)

# Calculate liquidity L
def calc_liquidity(row):
    sqrt_P = row['sqrt_price']
    token0 = row['eth_reserve']
    token1 = row['usdc_reserve']
    try:
        L0 = token0 / (1 / sqrt_P - 1 / sqrt_P_max)
        L1 = token1 / (sqrt_P - sqrt_P_min)
        return min(L0, L1)
    except ZeroDivisionError:
        return np.nan

df['L'] = df.apply(calc_liquidity, axis=1)

# Recalculate token0/token1 from L, and value in each
def calc_lp_values(row):
    sqrt_P = row['sqrt_price']
    P = row['price_usdc_per_eth']
    L = row['L']
    
    if sqrt_P <= sqrt_P_min:
        token0 = L * (1 / sqrt_P_min - 1 / sqrt_P_max)
        token1 = 0
    elif sqrt_P < sqrt_P_max:
        token0 = L * (1 / sqrt_P - 1 / sqrt_P_max)
        token1 = L * (sqrt_P - sqrt_P_min)
    else:
        token0 = 0
        token1 = L * (sqrt_P_max - sqrt_P_min)
    
    if P != 0:
        value_token1 = token1 + token0 * P
        value_token0 = token0 + token1 / P
    else:
        value_token1 = token1
        value_token0 = token0
    return pd.Series([value_token0, value_token1])

df[['test_lp_value_token0', 'test_lp_value_token1']] = df.apply(calc_lp_values, axis=1)
# %%
df['test_lp_value_usdc'] = df['test_lp_value_token1'] # This column already holds the value in USDC

# %%
