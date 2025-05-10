class UniswapV3Pool:
    def __init__(self,params):
        token0 = params['token0']
        token1 = params['token1']
        price_low = params['price_low']
        price_high = params['price_high']
        current_price = params['current_price']
        fee = params['fee']
        self.sqrt_P = current_price ** 0.5
        self.sqrt_P_lower = price_low ** 0.5
        self.sqrt_P_upper = price_high ** 0.5
        self.fee = fee

        # Compute liquidity
        L0 = token0 / (1 / self.sqrt_P - 1 / self.sqrt_P_upper)
        L1 = token1 / (self.sqrt_P - self.sqrt_P_lower)
        self.L = min(L0, L1)

        # Initialize reserves
        self.token0 = self.L * (1 / self.sqrt_P - 1 / self.sqrt_P_upper)
        self.token1 = self.L * (self.sqrt_P - self.sqrt_P_lower)

    def price(self):
        return self.sqrt_P ** 2  # token1 per token0
    
    

    def swap(self, amount_in, token_in_is_0):
        """
        Performs a swap on the pool.

        Args:
            amount_in (float): Amount of the input token.
            token_in_is_0 (bool): True if input is token0, False if input is token1.

        Returns:
            float: Amount of the output token received.
        """
        if token_in_is_0:
            # Logic from swap_token0_in
            amount_in_after_fee = amount_in * (1 - self.fee)
            # Avoid division by zero if liquidity or denominator is zero
            denominator = self.L * (1 / self.sqrt_P + 1 / self.sqrt_P_upper) if self.L > 0 and self.sqrt_P > 0 and self.sqrt_P_upper > 0 else 0
            if denominator == 0:
                return 0 # Cannot perform swap

            delta_sqrtP = amount_in_after_fee / denominator
            new_sqrt_P = min(self.sqrt_P + delta_sqrtP, self.sqrt_P_upper)
            token1_out = self.L * (new_sqrt_P - self.sqrt_P)

            self.sqrt_P = new_sqrt_P
            self.token0 += amount_in
            self.token1 -= token1_out
            return token1_out
        else:
            # Logic from swap_token1_in
            amount_in_after_fee = amount_in * (1 - self.fee)
             # Avoid division by zero if liquidity or denominator is zero
            denominator = self.L * (self.sqrt_P + self.sqrt_P_lower) if self.L > 0 else 0
            if denominator == 0:
                 return 0 # Cannot perform swap

            delta_sqrtP = amount_in_after_fee / denominator
            new_sqrt_P = max(self.sqrt_P - delta_sqrtP, self.sqrt_P_lower)
            # Ensure new_sqrt_P is positive before taking reciprocal
            if new_sqrt_P <= 0:
                 return 0 # Cannot perform swap
            token0_out = self.L * (1 / self.sqrt_P - 1 / new_sqrt_P) if self.sqrt_P > 0 else 0

            self.sqrt_P = new_sqrt_P
            self.token1 += amount_in
            self.token0 -= token0_out
            return token0_out

    def state(self):
        return {
            "price (token1 per token0)": self.price(),
            "sqrt_price": self.sqrt_P,
            "token0": self.token0,
            "token1": self.token1,
            "liquidity (internal)": self.L
        }