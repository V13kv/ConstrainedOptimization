import sympy as sp
import numpy as np

# TODO: equalities support (see penalty method combined with barrier method (see book))

class BarrierMethodTools:
    """A static class containing utility functions for the barrier method."""

    @staticmethod
    def create_barrier_function(constraints, epsilon=1e-6):
        """
        Creates the barrier function phi(x) for the given constraints using the logarithm of the product.

        Args:
            constraints (list of sympy expressions): List of inequality constraints (g_i(x) <= 0).
            epsilon (float): Small offset to ensure constraints are strictly less than zero.

        Returns:
            sympy expression: The barrier function phi(x).
        """
        # Add a small offset to the constraints to avoid g_i(x) = 0
        offset_constraints = [g - epsilon for g in constraints]

        # Barrier function: phi(x) = -log(product_of_constraints)
        product_of_constraints = sp.prod(-g for g in offset_constraints)
        phi = -sp.log(product_of_constraints)
        return phi

    @staticmethod
    def create_objective_with_barrier(f, constraints, mu):
        """
        Creates a new objective function with the barrier method applied.

        Args:
            f (sympy expression): The original objective function.
            constraints (list of sympy expressions): List of inequality constraints (g_i(x) <= 0).
            mu (sympy symbol or float): Barrier parameter.

        Returns:
            sympy expression: The new objective function F(x, mu).
        """
        # Create the barrier function
        phi = BarrierMethodTools.create_barrier_function(constraints)

        # New objective function: F(x, mu) = f(x) + mu * phi(x)
        F = f + mu * phi
        return F

    @staticmethod
    def gradient_function(F, variables):
        """
        Calculates the gradient of the function F with respect to the given variables.

        Args:
            F (sympy expression): The function to compute the gradient for.
            variables (list of sympy symbols): List of variables to differentiate with respect to.

        Returns:
            list of sympy expressions: The gradient of F.
        """
        gradient = [sp.diff(F, var) for var in variables]
        return gradient

    @staticmethod
    def is_feasible(constraints, variables, point, epsilon=1e-6):
        """
        Check if a point is feasible (satisfies all constraints).

        Args:
            constraints (list of sympy expressions): List of inequality constraints (g_i(x) <= 0).
            variables (list of sympy symbols): List of variables in the constraints.
            point (list or np.ndarray): The point to check.
            epsilon (float): Tolerance for feasibility (default: 1e-6).

        Returns:
            bool: True if the point is feasible, False otherwise.
        """
        # Convert constraints to numerical functions
        constraint_funcs = [sp.lambdify(variables, g, 'numpy') for g in constraints]

        # Evaluate constraints at the point
        for g_func in constraint_funcs:
            if g_func(*point) >= -epsilon:
                return False  # Constraint violated

        return True  # All constraints satisfied

    @staticmethod
    def find_feasible_point(constraints, variables, initial_guess, optimizer, max_iterations=1000):
        """
        Find a feasible point by minimizing the constraint violations.

        Args:
            constraints (list of sympy expressions): List of inequality constraints (g_i(x) <= 0).
            variables (list of sympy symbols): List of variables in the constraints.
            initial_guess (list or np.ndarray): Initial guess for the variables.
            optimizer: The optimizer to use (e.g., RAdam).
            max_iterations (int): Maximum number of iterations to find a feasible point (default: 1000).

        Returns:
            list of float: A feasible point.
        """
        # Define the objective function as the sum of squared constraint violations
        violation = sum(sp.Max(0, g)**2 for g in constraints)
        violation_func = sp.lambdify(variables, violation, 'numpy')

        # Define the gradient of the violation function
        gradient_violation = [sp.diff(violation, var) for var in variables]
        gradient_violation_func = sp.lambdify(variables, gradient_violation, 'numpy')

        # Convert initial_guess to a list of np.array objects
        params = [np.array([x]) for x in initial_guess]

        # Minimize the violation function
        for iteration in range(1, max_iterations + 1):
            # Extract scalar values from params
            param_values = [p[0] for p in params]

            # Compute gradients
            grads = np.array(gradient_violation_func(*param_values)).reshape(-1, 1)  # Ensure grads is a single array

            # Perform optimization step
            params = optimizer.step(params, grads, iteration)

            # Check if the point is feasible
            if BarrierMethodTools.is_feasible(constraints, variables, [p[0] for p in params]):
                return [p[0] for p in params]

        raise ValueError("Unable to find a feasible point within the maximum number of iterations.")

    @staticmethod
    def optimize_with_adaptive_barrier(f, constraints, initial_guess, optimizer, mu_0=1.0, num_iterations=100, mu_decay=0.9):
        """
        Solve a constrained optimization problem using the barrier method and a given optimizer.

        Args:
            f (sympy expression): The objective function to minimize.
            constraints (list of sympy expressions): List of inequality constraints (g_i(x) <= 0).
            initial_guess (list of floats or np.ndarray of floats): Initial guess for the variables.
            optimizer: The optimizer to use (e.g., RAdam).
            mu_0 (float): Initial barrier parameter (default: 1.0).
            num_iterations (int): Number of optimization iterations (default: 100).
            mu_decay (float): Decay factor for the barrier parameter (default: 0.9).

        Returns:
            list of float: Optimized parameters.
        """
        # Extract variables from the objective function only
        variables = list(f.free_symbols)
        variables.sort(key=lambda var: var.name)  # Sort variables for consistency

        # Check if the initial guess is feasible
        if not BarrierMethodTools.is_feasible(constraints, variables, initial_guess):
            raise ValueError("Initial guess is not in the feasible set!")

        # Initialize the barrier parameter
        mu = mu_0

        # Convert initial_guess to a list of np.array objects
        params = [np.array([x]) for x in initial_guess]

        # Optimization loop
        for iteration in range(1, num_iterations + 1):
            # Create the new objective function with the current barrier parameter
            F_sym = BarrierMethodTools.create_objective_with_barrier(f, constraints, mu)

            # Compute the gradient of F (symbolic)
            gradient_sym = BarrierMethodTools.gradient_function(F_sym, variables)

            # Convert symbolic expressions to numerical functions
            F_numeric = sp.lambdify(variables, F_sym, 'numpy')
            gradient_numeric = sp.lambdify(variables, gradient_sym, 'numpy')

            # Define the objective and gradient functions for the optimizer
            def objective_function(*params):
                return F_numeric(*params)

            def gradient_function_wrapper(*params):
                grads = gradient_numeric(*params)
                return [np.array(grad) for grad in grads]

            # Perform one optimization step
            params = optimizer.step(params, gradient_function_wrapper(*params), iteration)

            # Update the barrier parameter
            mu *= mu_decay

            # Print progress
            if (iteration + 1) % 10 == 0:
                loss = objective_function(*params)
                print(f"Iteration {iteration + 1}: Parameters = {[p[0] for p in params]}, Loss = {loss[0]:.4f}, mu = {mu:.4f}")

        # Return the optimized parameters as a list of floats
        return [p[0] for p in params]