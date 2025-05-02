from src.models.v2_pool import UniswapV2Pool

class v4_gamma_hook_pool:
    """
    Represents a Uniswap V2 plus a v4 gamma protecting hook.
    """
    def __init__(self, parameters, is_zero_gee=False):
        v2_pool = UniswapV2Pool(parameters, is_zero_fee=True)
        self.v2_pool = v2_pool
        self.gamma_factor = parameters['gamma_factor'] if not is_zero_gee else 0
        self.name = f'v4_gamma_pool_g={self.gamma_factor}'
        self.x = v2_pool.x # This is added so during simulation we can access pool.x regardless of whether it's a v4 pool or not
        self.y = v2_pool.y # This is added so during simulation we can access pool.y regardless of whether it's a v4 pool or not
        self.last_trade_fee_x = 0
        self.last_trade_fee_y = 0
        self.initial_x = parameters['initial_x']
        self.initial_y = parameters['initial_y']
        self.last_fee_pct = 0



    def run_gamma_hook(self, amount_in, amount_out, previous_x, previous_y, new_price, token_in_is_0):
        if token_in_is_0:
            r = amount_in/previous_x
            dynamic_fee_pct = (r / (1+r)) * (1+self.gamma_factor)
            amount_fee = amount_out * (dynamic_fee_pct )
            amount_out_net = amount_out - amount_fee
            self.v2_pool.fees_x += amount_fee/2*new_price
            self.v2_pool.fees_y += amount_fee/2

            self.v2_pool.x += amount_fee/2*new_price 
            self.v2_pool.y += amount_fee/2 # THIS ASSUMES HALF OF FEE ATOMICALLY SWAPPED AND ADDED TO OPPOSITE RESERVES
            self.x += amount_fee/2*new_price # This is added so during simulation we can access pool.x regardless of whether it's a v4 pool or not
            self.y += amount_fee/2 # This is added so during simulation we can access pool.y regardless of whether it's a v4 pool or not
            self.last_trade_fee_x = amount_fee/2*new_price # for tracking analytics
            self.last_trade_fee_y = amount_fee/2 # for tracking analytics
            self.last_fee_pct = dynamic_fee_pct

        else:
            r = amount_in/previous_y
            dynamic_fee_pct = (r / (1+r)) * (1+self.gamma_factor)
            amount_fee = amount_out * (dynamic_fee_pct)
            amount_out_net = amount_out - amount_fee

            self.v2_pool.fees_x += amount_fee/2
            self.v2_pool.fees_y += amount_fee/2/new_price
            self.v2_pool.x += amount_fee/2 # THIS ASSUMES HALF OF FEE ATOMICALLY SWAPPED AND ADDED TO OPPOSITE RESERVES
            self.v2_pool.y += amount_fee/2/new_price # THIS ASSUMES HALF OF FEE ATOMICALLY SWAPPED AND ADDED TO OPPOSITE RESERVES
            self.x += amount_fee/2 # This is added so during simulation we can access pool.x regardless of whether it's a v4 pool or not
            self.y += amount_fee/2/new_price # This is added so during simulation we can access pool.y regardless of whether it's a v4 pool or not
            self.last_trade_fee_x = amount_fee/2
            self.last_trade_fee_y = amount_fee/2/new_price
            self.last_fee_pct = dynamic_fee_pct
        return amount_out_net

    def get_price(self):
        return self.v2_pool.get_price()

    def swap(self, amount_in, token_in_is_0):
        """
        Calculates a swap and then applies the gamma hook.
        """
        previous_x = self.v2_pool.x
        previous_y = self.v2_pool.y
        previous_price = previous_x/previous_y
        amount_out = self.v2_pool.swap(amount_in, token_in_is_0)
        new_x, new_y, _, _ = self.v2_pool.get_balances()
        new_price = new_x/new_y
        amount_out_net = self.run_gamma_hook(amount_in, amount_out, previous_x, previous_y, new_price, token_in_is_0)

        return amount_out_net

    def get_balances(self):
        """Returns current balances (x, y, fees_x, fees_y)."""
        return self.v2_pool.get_balances()
        
    def reset(self):
        """Resets the pool to the initial state."""
        self.v2_pool.reset()
        self.x = self.initial_x
        self.y = self.initial_y
        self.last_trade_fee_x = 0
        self.last_trade_fee_y = 0

        
