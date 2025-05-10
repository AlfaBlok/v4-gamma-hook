# V4 Gamma Hook (Draft v0)

## Summary
This notebook shows research on a V4 Hook design that implements dynamic fees with the goal of preventing impermanent loss and achieving convexity for LPs.

## Problem Statement
Impermanent loss is a critical enemy of liquidity providers. It makes it hard for LPs to hedge their exposure. It results in lower liquidity and ultimately smaller pools and higher volatility. 


## Proposed Solution
A V4 Hook with a dynamic fee that fully cancels out Impermanent Loss. This would allow LPs to fully hedge their position and achieve delta neutrality more easily.
<br>
![Calculation Example](../plots/img00.png "Calculation Example")
## Strategy
We will charge a fee identical to impermanent loss on each swap and reinvest the fee into the pool immediately.

### Example: xy Pool USDC/ETH
Each row in the table represents a new price and tracks balances on x pool and y pool, with impermanent loss (IL) marked in red. 

![Calculation Example](../plots/image0.png "Calculation Example")

Since we want to charge IL as the fee percentage on the incoming swap, we look at IL as a percentage over the swap amount (first yellow column). 

We observe that IL as a percentage of swap size exactly matches the swap size relative to reserve (second yellow column).

![Calculation Example](../plots/IL_O.png "Calculation Example")

This confirms 1:1 relationship. It suggests that if your order is 10% of the reserve, a 10% fee on your order eliminates IL. A circular recurrency emerges from this, but we can resolve.

## Dynamic Fee Derivation
Let us define the following variables:

- $O$: Order size (token quantity sent in swap)
- $R$: Pool reserve (token quantity available)
- $r = \frac{O}{R}$: Order to reserve ratio
- $F$: Fee amount charged
- $f = \frac{F}{O}$: Fee percentage of order size
- $O_{net} = O \cdot (1 - f)$: Net order after fee (token quantity)
- $IL$: Impermanent loss value

We aim to find an expression for $f$ such that the resulting $F$ equals $IL$. We set the fee amount to be paid as $IL$ and use the equality observed in the yellow columns.

**Steps**:

1. Set the fee amount equal to the impermanent loss:  
   $$F = IL$$

2. From the table, the impermanent loss as a percentage of swap size equals the swap size relative to reserve:  
   $$\frac{IL}{O} = \frac{O_{net}}{R}$$

3. Combine the previous two equations:  
   $$\frac{IL}{O} = \frac{F}{O} = \frac{O_{net}}{R}$$

4. Simplify the fee to pay over order size as the fee percentage (our target):  
   $$f = \frac{O_{net}}{R}$$

5. Substitute $O_{net}$ per its definition:  
   $$f = \frac{O \cdot (1 - f)}{R}$$

6. Substitute $\frac{O}{R} = r$ per the definition:  
   $$f = r \cdot (1 - f)$$

7. Rearrange to solve for $f$:  
   $$f = \frac{r}{1 + r}$$

This yields a simple heuristic to calculate a dynamic fee that fully offsets IL: $f = \frac{r}{1 + r}$, where $r$ is the ratio between swap amount and reserve amount.

## Testing Architecture
We validate the pool's functionality with this hook using Python, creating two pool classes:
- **V2 Class**: Implements a flat fee.
- **V4 Class**: Encapsulates an internal V2 pool with a post-swap execution hook logic.

### V2 Pool Swap Method
Mimics Solidity code in V2 for swap execution.
![Calculation Example](../plots/Picture11.png "Calculation Example")


### V4 Pool Swap Method
![Calculation Example](../plots/code00.png "Calculation Example")

Calls the V2 pool's swap function and passes the output to the `run_gamma_hook` function, which implements the dynamic fee calculation: **f = r / (1 + r)**. Half of the fee is deposited back into `reserve0` and the other half into `reserve1`, assuming atomic swap execution. A `gamma_factor` (denoted as **g** or "gee") scales the fee to regulate gamma, distinct from a regular percentage fee.

### V4 Run Gamma Hook Method
The run_gamma_hook simply implements the dynamic fee calculation as discussed:
![Calculation Example](../plots/code0.png "Calculation Example")


## Single Swap Test Results
We test pools by sending single swap orders, starting at a price of **1 ARB = 100 USDC**, across a range of swap sizes relative to pool reserves (0% to 10,000%).

### Price vs Value
![Calculation Example](../plots/image.png "Calculation Example")
The V4 pool with the Gamma hook (green line) behaves identically to a buy-and-hold strategy, as it has no IL. IL is a byproduct of **Gamma**, a measure of the pool value's "acceleration" or "deceleration" due to price changes:
- **V2 pools** (red with fee, blue without) suffer IL due to value deceleration, making them concave with negative gamma.
- **V4 pool with g = 0** (green) is linear, with no curvature and zero gamma.
- **V4 pool with positive g** (purple) accelerates with price, becoming convex with positive gamma, enabling "Impermanent Gain" (IG).

The Gamma hook allows control over the pool's gamma, with zero gamma being a special case. The **g** factor lets LPs adjust gamma above or below zero.

### Gamma Analysis
![Calculation Example](../plots/Picture6.png "Calculation Example")
- V4 pool with **g = 0**: Zero gamma.
- V4 pool with positive **g**: Positive gamma.

### Delta Analysis
![Calculation Example](../plots/Picture8.png "Calculation Example")
- V4 pool with **g = 0**: Constant delta 1. Perfect for delta-neutral hedging.
  
  


### XY Reserve Map
![Calculation Example](../plots/Picture7.png "Calculation Example")
For orders ≥50% of reserves, the fee exceeds the order size, creating a defensive wall against pool depletion.

### Price Comparison
![Calculation Example](../plots/Picture9.png "Calculation Example")
- **V4 Gamma pool prices**:
  - Worse than flat fee pools for very large swaps.
  - Better than flat fee pools for regular swap sizes.
  - No IL across the entire price range.
  - "Impermanent Gain" for positive **g**.
- For order sizes <3% of reserves, V4 Gamma pool fees are cheaper than V2 flat fee pools (assuming a standard 0.03% fee).

### Impermanennt Loss Comparison
![Calculation Example](../plots/Picture10.png "Calculation Example")

### Fee Comparison
![Calculation Example](../plots/Picture111.png "Calculation Example") <br>
Looking at broad ranges Gamma pools are generally more expensive.

But there is a narrow range in which gamma pools are cheaper than flat fee pools:

![Calculation Example](../plots/Picture12.png "Calculation Example")
In the case of 0,3% pools, unsurprisingly the threshold happens at 0,3%.

We can generalize: Gamma pools are cheaper than flat fee pools, when the order relative to the pool size is lower than the flat fee.


We must evaluate what are historical swaps relative to reserves, to see how they compare to the existing flat fee pools available.

<br>

### Real-Life Swap Distribution
Below is actual distribution of swaps for the  **ARB:USDC** pool on L2 Arbitrum (1M TVL) for the last 100K swaps:
![Calculation Example](../plots/Picture14.png "Calculation Example")
Here zooming in the 0.4% range:
![Calculation Example](../plots/Picture15.png "Calculation Example")
Most orders are <0.3% of reserves. Thus, V4 Gamma pools offer cheaper fees for the vast majority of swaps compared to pools of that fee, with V2 being cheaper only for large swaps (>3K swaps on 1M pool).

For larger pools 100M in TVL, the threshold at which Gamma Pool becomes more expensive would be $300K swaps.

In short for big pools only whales would find Gamma pools more expensive.


## Multi-Swap Test
We create a random swap generator matching the **ARB:USDC** swap size distribution and simulate real-life trading.

### Validation
The synthetic order distribution matches historical data.
![Calculation Example](../plots/Picture16.png "Calculation Example")

### Single Sequence Test
Applying a 20K random swap sequence to each pool shows IL is canceled out in Gamma pools, while V2 pools with fees offset some IL but still suffer losses.
![Calculation Example](../plots/Picture17.png "Calculation Example")
![Calculation Example](../plots/Picture18.png "Calculation Example")
![Calculation Example](../plots/Picture19.png "Calculation Example")
![Calculation Example](../plots/Picture1111.png "Calculation Example")
<br>
![Calculation Example](../plots/Picture114.png "Calculation Example")
We add transparency and focus on the cloud area:
![Calculation Example](../plots/Picture115.png "Calculation Example")
While each swap experienced a different fee, most times those fees where lower than equivalent pool's flat fee.
### Impermanent Loss Comparison
![Calculation Example](../plots/image9.png "Calculation Example")


Gamma pools show consistent IL performance in realistic trading environment.

### Monte Carlo Simulation



![Calculation Example](../plots/img0.png "Calculation Example")

Running 1,000 simulations, each with 10,000 swaps, and comparing the final results for all pools suggests:
- Gamma pools eliminate IL.
- Gamma pools have lower average fees.


## Summary and Discussion
A dynamic fee based on the **order size to reserve size ratio** (O/R) enables the V4 hook to fully offset Impermanent Loss.

### Benefits
- Enables LPs to fully hedge via delta-neutral strategies.
- Cheaper fees for most users.
- **g** factor allows convex exposure to the underlying asset.
- Simple computation logic.

### Drawbacks
- Dependency on atomic swaps, which may suffer from slippage.
- Less competitive for very large order sizes.

### Next Steps
- Peer review for feedback and potential flaws.
- Explore alternative designs without atomic swap dependency.
- Evaluate swap dependency implementation options.
- Build a Solidity proof of concept.
- Extend testing to V3 logic. In principle all same principles should apply but need to be tested.

### Open questions / debug
- Why is IL slightly posisitve on swap squences, when it has 0 IL on a single swap basis?
- Would pools remain arbitraged to consensus price in high volatility environments?
