

#%%
import sys
import os
import numpy as np
import pandas as pd

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Project root added to sys.path: {project_root}")


from src.utils.clean import sweep_swap_amount_range, get_xy_fig
from src.models.v3_pool import UniswapV3Pool

pool = UniswapV3Pool(
    token0=100,
    token1=50,
    price_low=1,
    price_high=4,
    current_price=2,
    fee=0.0
)

print("Initial:", pool.state())

print("\nToken0 in → token1 out:")
out1 = pool.swap(amount_in=10, token_in_is_0=True)
print(f"User gets {out1:.6f} token1")
print(pool.state())

print("\nToken1 in → token0 out:")
out2 = pool.swap(amount_in=5, token_in_is_0=False)
print(f"User gets {out2:.6f} token0")
print(pool.state())
# %%

range = np.arange(0, 100, 10)