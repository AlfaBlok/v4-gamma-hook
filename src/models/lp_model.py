import numpy as np
import pandas as pd

def calculate_lp_payoff(price_ratio, fee_percentage=0.003, initial_price=1.0, volume_tvl_ratio=1.0, use_dynamic_fee=False, fee_dial_setting=0.01, show_fee=False):
    """
    Calculate the payoff of a Uniswap LP position based on the price change.
    This version aligns with V_lp_norm = sqrt(price_ratio) before fees.
    
    Args:
        price_ratio (float or np.ndarray): Current price divided by initial price (P_current / P_initial)
        fee_percentage (float): Fee percentage (e.g., 0.003 for 0.3%)
        initial_price (float): Initial price when LP position was created (default: 1.0)
        volume_tvl_ratio (float): Ratio of daily trading volume to TVL (default: 1.0)
        use_dynamic_fee (bool): If True, fee will be equal to the impermanent loss (default: False)
        fee_dial_setting (float): Scaling factor for the dynamic fee (default: 0.01 for 1%)
        
    Returns:
        float or np.ndarray: LP position value relative to initial investment (minimum 0)
    """
    if isinstance(price_ratio, (list, tuple)):
        price_ratio = np.array(price_ratio)
    
    # Ensure price ratio is positive
    price_ratio = np.maximum(price_ratio, 1e-10)
    
    # Calculate base LP value using the square root formula: V_lp / V_0 = sqrt(P1/P0)
    # lp_value_without_fees = 2*np.sqrt(price_ratio)-1 # Old formula
    lp_value_without_fees = np.sqrt(price_ratio)
    
    # Ensure LP value without fees is never negative (sqrt ensures this for positive price_ratio)
    lp_value_without_fees = np.maximum(lp_value_without_fees, 0.0)
    
    # Calculate fee accumulation
    if show_fee and use_dynamic_fee:
        # Dynamic fee equal to impermanent loss (defined as buy_and_hold - LP)
        il = impermanent_loss(price_ratio)
        fee_accumulation =  il + fee_dial_setting # No need to scale, as IL is already an absolute value
    elif show_fee:
        # Original flat fee model
        # This is a simplified model - in reality, fees depend on actual trading volume
        fee_accumulation = fee_percentage * volume_tvl_ratio
    else:
        fee_accumulation = 0
    
    # Total LP value with fees
    lp_value = lp_value_without_fees + fee_accumulation
    
    return lp_value

def calculate_theoretical_lp_value(price_ratio):
    """
    Calculate the theoretical value of a Uniswap LP position without fees.
    This shows the pure constant product formula result: 2*sqrt(price_ratio)-1
    
    The formula is bounded to never return negative values, as a real LP position
    cannot be worth less than 0.
    
    Args:
        price_ratio (float or np.ndarray): Current price divided by initial price (P_current / P_initial)
        
    Returns:
        float or np.ndarray: LP position value relative to initial investment (minimum 0)
    """
    if isinstance(price_ratio, (list, tuple)):
        price_ratio = np.array(price_ratio)
    
    # Ensure price ratio is positive
    price_ratio = np.maximum(price_ratio, 1e-10)
    
    # Calculate LP value using the constant product formula
    # LP value = 2*sqrt(price_ratio)-1
    lp_value = 2*np.sqrt(price_ratio)-1
    
    # Ensure LP value is never negative - at worst, an LP position is worth 0
    lp_value = np.maximum(lp_value, 0.0)
    
    return lp_value

def calculate_lp_path(price_path, fee_percentage=0.003, volume_tvl_ratio=1.0, use_dynamic_fee=False, fee_dial_setting=0.01):
    """
    Calculate the LP position value path based on a price path.
    
    Args:
        price_path (np.ndarray): Array of prices over time (normalized to start at 1.0)
        fee_percentage (float): Fee percentage (e.g., 0.003 for 0.3%)
        volume_tvl_ratio (float): Ratio of daily trading volume to TVL (default: 1.0)
        use_dynamic_fee (bool): If True, fee will be equal to the impermanent loss (default: False)
        fee_dial_setting (float): Scaling factor for the dynamic fee (default: 0.01 for 1%)
        
    Returns:
        tuple: (lp_path, fee_path) where:
            - lp_path (np.ndarray): LP position value path (normalized to start at 1.0)
            - fee_path (np.ndarray): Daily fee values collected for each day
    """
    if len(price_path) == 0:
        return np.array([]), np.array([])
    
    # Initialize LP path with the same shape as price_path
    lp_path = np.ones_like(price_path)
    lp_returns_path = np.ones_like(price_path)
    
    # Calculate LP value for each day relative to the starting point
    for i in range(1, len(price_path)):
        price_ratio = price_path[i] / price_path[0]
        # Guard against extreme price ratios to prevent numerical issues
        price_ratio = np.clip(price_ratio, 1e-10, 1e10)
        lp_returns_path[i] = calculate_lp_payoff(price_ratio, fee_percentage=0.0, use_dynamic_fee=False, fee_dial_setting=0)

    # we do pct change to get the daily returns
    lp_returns_path = pd.Series(lp_returns_path).pct_change()
    
    # Replace NaN values (first value will be NaN after pct_change)
    lp_returns_path = lp_returns_path.fillna(0)
    
    # Add fee returns based on fee model
    fee_path = np.zeros_like(price_path)
    if use_dynamic_fee:
        # For dynamic fee, calculate step-by-step impermanent loss for each transition
        for i in range(1, len(price_path)):
            # Calculate price ratio between consecutive steps
            step_price_ratio = price_path[i] / price_path[i-1]
            step_price_ratio = np.clip(step_price_ratio, 1e-10, 1e10)
            
            # Calculate LP value (no fees) for this step based on sqrt formula
            # lp_value_step = 2 * np.sqrt(step_price_ratio) - 1 # Old
            lp_value_step_norm = np.sqrt(step_price_ratio)
            lp_value_step_norm = np.maximum(lp_value_step_norm, 0.0)
            
            # Calculate impermanent loss: V_lp / V_bh - 1 
            # Here V_bh_norm = (x0*P1 + y0) / (x0*P0 + y0). Assume x0*P0=y0=V0/2.
            # V_bh_norm = ( (V0/(2*P0))*P1 + V0/2 ) / V0 = P1/(2*P0) + 1/2 = (P1/P0 + 1)/2 = (step_price_ratio + 1)/2
            v_bh_step_norm = (step_price_ratio + 1.0) / 2.0
            step_il_relative = (lp_value_step_norm / v_bh_step_norm - 1.0) if v_bh_step_norm > 1e-9 else 0.0
            # IL is typically negative or zero, fee should compensate (be positive)
            # We want fee = abs(IL) = V_bh_norm - V_lp_norm for V_lp <= V_bh
            fee_component = max(0, v_bh_step_norm - lp_value_step_norm) 
            
            # Scale the fee component by the fee dial setting
            fee_path[i] = fee_component * fee_dial_setting # Assign daily fee
        
        # Add the fee path to the lp returns path
        # lp_returns_path = lp_returns_path + fee_path + fee_dial_setting # Old dynamic fee logic needs adjustment
        lp_returns_path = lp_returns_path + fee_path # Add calculated daily dynamic fee return
    else:
        # Standard flat fee
        flat_fee = fee_percentage * volume_tvl_ratio
        fee_path = np.zeros_like(price_path)
        fee_path[1:] = flat_fee  # First day has no fee
        lp_returns_path = lp_returns_path + flat_fee
    
    # Ensure returns are finite (no NaN or infinity values)
    lp_returns_path = np.array(lp_returns_path)
    lp_returns_path = np.nan_to_num(lp_returns_path, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure returns aren't too extreme to prevent numerical issues
    # LP positions can't lose more than 100% of their value in a single step
    lp_returns_path = np.maximum(lp_returns_path, -0.99)
    
    # we add 1 to the returns and then cumprod to get the lp path
    lp_path = np.cumprod(1 + lp_returns_path)
    
    # Final safety check - ensure all values are positive and finite
    lp_path = np.maximum(lp_path, 0.0)
    lp_path = np.nan_to_num(lp_path, nan=1.0, posinf=0.0, neginf=0.0)
    
    return lp_path, fee_path

def calculate_lp_returns(price_returns, fee_percentage=0.003, volume_tvl_ratio=1.0, use_dynamic_fee=False):
    """
    Calculate daily returns for LP positions based on underlying price returns.
    
    Args:
        price_returns (np.ndarray): Array of daily price returns
        fee_percentage (float): Fee percentage (e.g., 0.003 for 0.3%)
        volume_tvl_ratio (float): Ratio of daily trading volume to TVL (default: 1.0)
        use_dynamic_fee (bool): If True, fee will be equal to the impermanent loss (default: False)
        
    Returns:
        np.ndarray: Array of daily LP returns
    """
    if len(price_returns) == 0:
        return np.array([])
    
    # Handle any potential NaN values in price returns
    price_returns = np.array(price_returns)
    price_returns = np.nan_to_num(price_returns, nan=0.0, posinf=0.0, neginf=-0.99)
    
    # Convert price returns to price path starting at 1.0
    price_path = np.cumprod(1 + np.array(price_returns))
    
    # Safety check for extreme values
    price_path = np.maximum(price_path, 1e-10)
    price_path = np.minimum(price_path, 1e10)
    
    # Prepend 1.0 to represent the initial price
    full_price_path = np.concatenate(([1.0], price_path))
    
    # Calculate LP path
    lp_path, fee_path = calculate_lp_path(full_price_path, fee_percentage, volume_tvl_ratio, use_dynamic_fee)
    
    # Calculate daily returns from the LP path
    # Avoid division by zero
    lp_returns = np.zeros_like(lp_path[:-1])
    safe_denominator = np.maximum(lp_path[:-1], 1e-10)
    lp_returns = np.diff(lp_path) / safe_denominator
    
    # Final safety check
    lp_returns = np.nan_to_num(lp_returns, nan=0.0, posinf=0.0, neginf=-0.99)
    
    return lp_returns

def plot_lp_payoff_curve(price_range=None, fee_percentages=(0.001, 0.003, 0.01), use_dynamic_fee=False):
    """
    Generate data points for plotting the LP payoff curve.
    
    Args:
        price_range (np.ndarray, optional): Range of price ratios to calculate payoff for
        fee_percentages (tuple): Tuple of fee percentages to calculate curves for
        use_dynamic_fee (bool): If True, fee will be equal to the impermanent loss (default: False)
        
    Returns:
        tuple: (price_ratios, payoffs) where:
            - price_ratios is an array of price ratios
            - payoffs is a list of arrays of payoffs, one array per fee percentage
    """
    if price_range is None:
        # Default price range from -100% to +200% (price ratios from 0 to 3)
        price_range = np.linspace(0.01, 3.0, 300)
    
    # Calculate payoff for each fee percentage
    payoffs = []
    for fee in fee_percentages:
        payoff = calculate_lp_payoff(price_range, fee_percentage=fee, use_dynamic_fee=use_dynamic_fee)
        payoffs.append(payoff)
    
    return price_range, payoffs

def impermanent_loss(price_ratio):
    """
    Calculate impermanent loss as (buy_and_hold - LP_position), which is a positive value.
    This is the definition that creates a U-shaped parabola bottoming at 0 when price_ratio = 1.
    
    Args:
        price_ratio (float or np.ndarray): Current price divided by initial price
        
    Returns:
        float or np.ndarray: Impermanent loss as a positive value
    """
    # Ensure price ratio is positive
    price_ratio = np.maximum(price_ratio, 1e-10)
    
    # Calculate LP value without fees using the constant product formula
    # Ensure it's never negative
    lp_value_norm = np.sqrt(price_ratio)
    lp_value_norm = np.maximum(lp_value_norm, 0.0)
    
    # Calculate Buy & Hold value (normalized)
    # Assumes balanced initial investment (x0*p0 = y0 = V0/2)
    # V_bh / V0 = (x0*p1 + y0) / (x0*p0 + y0) = ( (V0/(2p0))*p1 + V0/2 ) / V0 = p1/(2p0) + 1/2 = (price_ratio + 1) / 2
    bh_value_norm = (price_ratio + 1.0) / 2.0
    
    # Calculate relative IL
    if bh_value_norm < 1e-9:
        # Avoid division by zero if B&H value is somehow zero
        il = 0.0
    else:
        il = (lp_value_norm / bh_value_norm) - 1.0
    
    # Ensure IL is not positive (it represents a loss or break-even compared to B&H)
    il = min(il, 0.0)
    
    return il

def simulate_v2_trade(x, y, amount_in, token_in_is_0, fee_percentage):
    """
    Simulates a single Uniswap V2 trade (x * y = k) without reinvesting fees.

    Args:
        x (float): Initial amount of token0 in the pool.
        y (float): Initial amount of token1 in the pool.
        amount_in (float): Amount of the input token being traded.
        token_in_is_0 (bool): True if token0 is the input token, False if token1 is.
        fee_percentage (float): Swap fee percentage (e.g., 0.003 for 0.3%).

    Returns:
        tuple: (new_x, new_y, fee0_collected, fee1_collected, amount_out)
            - new_x: Final amount of token0 in the pool.
            - new_y: Final amount of token1 in the pool.
            - fee0_collected: Amount of fee collected in token0.
            - fee1_collected: Amount of fee collected in token1.
            - amount_out: Amount of the output token received by the trader.
    """
    if x <= 0 or y <= 0 or amount_in <= 0:
        # Invalid state or input
        return x, y, 0, 0, 0

    k = x * y # Constant product
    fee_collected_0 = 0
    fee_collected_1 = 0

    if token_in_is_0:
        # Trading token0 (dx) for token1 (dy)
        dx = amount_in
        fee = dx * fee_percentage
        dx_after_fee = dx * (1 - fee_percentage)
        
        # Calculate new y based on constant k
        # (x + dx_after_fee) * new_y = k
        new_x = x + dx
        new_y = k / (x + dx_after_fee) 
        amount_out = y - new_y # dy (amount of token1 out)
        fee_collected_0 = fee

    else:
        # Trading token1 (dy) for token0 (dx)
        dy = amount_in
        fee = dy * fee_percentage
        dy_after_fee = dy * (1 - fee_percentage)

        # Calculate new x based on constant k
        # new_x * (y + dy_after_fee) = k
        new_y = y + dy
        new_x = k / (y + dy_after_fee)
        amount_out = x - new_x # dx (amount of token0 out)
        fee_collected_1 = fee
        
    # Ensure non-negative balances and amount_out
    new_x = max(new_x, 0)
    new_y = max(new_y, 0)
    amount_out = max(amount_out, 0)

    return new_x, new_y, fee_collected_0, fee_collected_1, amount_out







class UniswapV2Pool:
    """
    Represents a Uniswap V2 pool with x*y=k dynamics.
    Handles swaps and tracks balances and fees (fees kept separate).
    """
    def __init__(self, initial_x, initial_y, fee_percentage):
        """
        Initialize the pool.

        Args:
            initial_x (float): Initial amount of token0.
            initial_y (float): Initial amount of token1.
            fee_percentage (float): Swap fee (e.g., 0.003).
        """
        if initial_x <= 0 or initial_y <= 0:
            raise ValueError("Initial balances must be positive.")
        if not 0 <= fee_percentage < 1:
             raise ValueError("Fee percentage must be between 0 and 1.")

        self.initial_x = float(initial_x)
        self.initial_y = float(initial_y)
        self.x = float(initial_x)
        self.y = float(initial_y)
        self.fee_percentage = float(fee_percentage)
        self.k = self.x * self.y
        self.fees_x = 0.0 # Fees collected in token0
        self.fees_y = 0.0 # Fees collected in token1

    def get_price(self):
        """Returns the current price (y/x)."""
        if self.x <= 1e-12:
            return np.nan # Or raise error / return infinity
        return self.y / self.x

    def reset(self):
         """Resets the pool to its initial state."""
         self.x = self.initial_x
         self.y = self.initial_y
         self.k = self.initial_x * self.initial_y
         self.fees_x = 0.0
         self.fees_y = 0.0
         print(f"Pool reset to x={self.x}, y={self.y}")

    def swap(self, amount_in, token_in_is_0):
        """
        Performs a swap on the pool, booking fees proportionally first.

        Args:
            amount_in (float): Amount of the input token.
            token_in_is_0 (bool): True if input is token0, False if input is token1.

        Returns:
            float: Amount of the output token received.

        Raises:
            ValueError: If amount_in is non-positive.
        """
        if amount_in <= 0:
             raise ValueError("Swap amount must be positive.")

        # 1. Calculate fee
        fee = amount_in * self.fee_percentage

        # 2. Store pre-fee state
        x_before_fee = self.x
        y_before_fee = self.y

        # 3. Book fee by adding value proportionally to reserves (maintains price)
        if token_in_is_0:
            # Fee 'fee' is in token0
            self.fees_x += fee # Track total collected fee
            if x_before_fee > 1e-12:
                price = x_before_fee / y_before_fee
                increase_x = fee / 2.0 # THIS ASSUMES A 50/50 FEE DISTRIBUTION ASSUMING PERFECT INSTANT ARBITRAGE OF THE FEE IMBALANCE
                increase_y = increase_x / price
                self.x += increase_x
                self.y += increase_y
            else:
                # If x is near zero, price is ill-defined/infinite.
                # Add fee value primarily to the token0 reserve.
                self.x += fee / 2.0 # Keep y unchanged as price is infinite

        else: # token_in_is_1
            # Fee 'fee' is in token1
            self.fees_y += fee # Track total collected fee
            if y_before_fee > 1e-12:
                price = x_before_fee / y_before_fee
                increase_y = fee / 2.0
                increase_x = increase_y * price
                self.x += increase_x
                self.y += increase_y
            else:
                # If y is near zero, price is zero, inv_price infinite.
                # Add fee value primarily to the token1 reserve.
                self.y += fee / 2.0 # Keep x unchanged

        # 4. Recalculate k after fee booking
        self.k = self.x * self.y

        # 5. Perform swap using full amount_in on the fee-adjusted pool
        x_after_fee_booking = self.x
        y_after_fee_booking = self.y
        k_after_fee_booking = self.k # Use the updated k

        if token_in_is_0:
            # Swap token0 (dx) for token1 (dy)
            dx = amount_in
            # (x_after_fee_booking + dx) * final_y = k_after_fee_booking
            final_y = k_after_fee_booking / (x_after_fee_booking + dx)
            amount_out = y_after_fee_booking - final_y

            # Update pool state post-swap
            self.x = x_after_fee_booking + dx
            self.y = final_y

        else: # token_in_is_1
            # Swap token1 (dy) for token0 (dx)
            dy = amount_in
            # final_x * (y_after_fee_booking + dy) = k_after_fee_booking
            final_x = k_after_fee_booking / (y_after_fee_booking + dy)
            amount_out = x_after_fee_booking - final_x

            # Update pool state post-swap
            self.y = y_after_fee_booking + dy
            self.x = final_x

        # 6. Ensure non-negative results
        amount_out = max(0, amount_out)
        self.x = max(0, self.x)
        self.y = max(0, self.y)

        # Note: self.k is implicitly updated by the final self.x, self.y assignments.
        # We could recalculate self.k = self.x * self.y here for consistency,
        # but it's not strictly needed until the *next* swap.

        return amount_out

    def get_balances(self):
         """Returns current balances (x, y, fees_x, fees_y)."""
         return self.x, self.y, self.fees_x, self.fees_y 