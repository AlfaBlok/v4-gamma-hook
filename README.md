# Uniswap V4 Hook Research: Dynamic Gamma Fee

This repository contains research exploring a dynamic fee mechanism for Uniswap V4 hooks, designed to mitigate impermanent loss (IL) for liquidity providers (LPs).

## Overview

The core of this research is documented in the following markdown file:

*   **[`src/V4_Gamma_Hook.markdown`](docs/V4_Gamma_Hook.markdown)**: This document details the "Gamma Hook" concept, derives the dynamic fee structure ($f = \frac{r}{1 + r}$), and presents simulation results comparing its performance against traditional V2 pools. It explores the potential for eliminating IL and even achieving positive gamma ("Impermanent Gain").

## Simulation & Analysis

The analysis presented in the markdown document is supported by simulations coded in Python.

*   **[`src/notebook.ipynb`](src/notebook.ipynb)**: This Jupyter notebook contains the Python code used to simulate pool behavior, generate comparative plots, and validate the Gamma Hook strategy under various market conditions (single swaps, multi-swap sequences, Monte Carlo).

## Key Concepts Explored

*   **Impermanent Loss (IL) Mitigation:** The primary goal is to design a fee structure that dynamically offsets IL.
*   **Dynamic Fee:** The proposed fee is calculated based on the ratio of the swap size to the pool's reserve ($r = O/R$).
*   **Gamma / Convexity:** The research investigates how the hook affects the pool's value curvature (Gamma) in response to price changes, aiming for zero or positive gamma.
*   **LP Hedging:** By eliminating IL (zero gamma), LPs could potentially achieve delta-neutral positions more easily.
*   **Fee Comparison:** The dynamic fee's cost to swappers is compared against traditional fixed-tier fees.

## Project Status

This is currently a research exploration. The concepts and results are presented in the linked markdown file and notebook.

## Contribute

Please feel free to contribute to the project and participate in the research. Planning to use Github native tooling around the repo for discussions, issues, etc.

## Acknowledgments

Special thanks to oktep for his support and help on this project and the insightful feedback on the earliest versions of this research.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 