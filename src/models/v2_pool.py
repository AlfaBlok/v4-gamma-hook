class UniswapV2Pool:
    """
    Represents a Uniswap V2 pool with x*y=k dynamics.
    Handles swaps and tracks balances and fees (fees kept separate).
    """
    def __init__(self, parameters, is_zero_fee=True):
        """
        Initialize the pool.

        Args:
            initial_x (float): Initial amount of token0.
            initial_y (float): Initial amount of token1.
            fee_percentage (float): Swap fee (e.g., 0.003).
        """
        initial_x = parameters['initial_x']
        initial_y = parameters['initial_y']
        fee_pct = parameters['fee_pct'] if not is_zero_fee else 0.0

        if initial_x <= 0 or initial_y <= 0:
            raise ValueError("Initial balances must be positive.")
        if not 0 <= fee_pct < 1:
             raise ValueError("Fee percentage must be between 0 and 1.")


        self.initial_x = float(initial_x)
        self.initial_y = float(initial_y)
        self.x = float(initial_x)
        self.y = float(initial_y)
        self.fee_pct = float(fee_pct)
        self.k = self.x * self.y
        self.fees_x = 0.0 # Fees collected in token0
        self.fees_y = 0.0 # Fees collected in token1
        self.name = f'v2_pool_f={fee_pct}'
        self.last_fee_pct = self.fee_pct

    def get_price(self):
        """Returns the current price (y/x)."""
        if self.x <= 1e-12:
            return np.nan # Or raise error / return infinity
        return self.x / self.y

    def reset(self):
         """Resets the pool to its initial state."""
         self.x = self.initial_x
         self.y = self.initial_y
         self.k = self.initial_x * self.initial_y
         self.fees_x = 0.0
         self.fees_y = 0.0
         self.last_trade_fee_x = 0.0
         self.last_trade_fee_y = 0.0
        #  print(f"Pool reset to x={self.x}, y={self.y}")

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
        invariant = self.x * self.y

        if token_in_is_0:
 
            fee = amount_in * self.fee_pct 
            new_x_pool = self.x + amount_in
            new_y_pool = invariant / (new_x_pool - fee)
            amount_out = self.y - new_y_pool # Amount of token1 out
            # Update pool balances
            self.x = new_x_pool
            self.y = new_y_pool
            self.fees_x += fee # for tracking purposes only, pool doesn't have a fee balance
            self.fees_y += 0 # for tracking purposes only, pool doesn't have a fee balance
            self.last_trade_fee_x = fee # for tracking purposes only, pool doesn't have a fee balance
            self.last_trade_fee_y = 0 # for tracking purposes only, pool doesn't have a fee balance

        else: # token_in_is_1
            fee = amount_in * self.fee_pct 
            new_y_pool = self.y + amount_in
            new_x_pool = invariant / (new_y_pool - fee)
            amount_out = self.x - new_x_pool # Amount of token0 out

            # Update pool balances
            self.x = new_x_pool
            self.y = new_y_pool
            self.fees_x += 0
            self.fees_y += fee
            self.last_trade_fee_x = 0
            self.last_trade_fee_y = fee

        return amount_out

    def get_balances(self):
         """Returns current balances (x, y, fees_x, fees_y)."""
         return self.x, self.y, self.fees_x, self.fees_y 