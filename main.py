from optimizers.RAdam import RAdam
from tools.tools import BarrierMethodTools

import sympy as sp


# Define symbolic variables
x1, x2, x3 = sp.symbols('x1 x2 x3')

# Define the objective function
f = x1**2 + x2**2 + x3**2 + 2 * sp.cos(x1) + 3 * sp.sin(x2)

# User-defined inequality constraints (g_i(x) <= 0)
constraints = [
    x1**2 + x2**2 + x3**2 - 10,
    sp.exp(x1)-10*x2
]

# Initial guess for the variables
initial_guess = [1.0, 1.0, 1.0]

# Initialize RAdam optimizer
optimizer = RAdam(lr=1e-2)

# Solve the constrained optimization problem using BarrierMethodTools
optimized_params = BarrierMethodTools.optimize_with_adaptive_barrier(
    f, constraints, initial_guess, optimizer, mu_0=1.0, num_iterations=100, mu_decay=0.9
)

# Print the optimized parameters
print("Optimized Parameters:")
print(f"x1 = {optimized_params[0]:.4f}, x2 = {optimized_params[1]:.4f}, x3 = {optimized_params[2]:.4f}")