from optimizers.base import OptimizationAlgorithm

import numpy as np


# https://stats.stackexchange.com/questions/232741/why-is-it-important-to-include-a-bias-correction-term-for-the-adam-optimizer-for
class RAdam(OptimizationAlgorithm):
    """Rectified Adam (RAdam) optimizer (see https://arxiv.org/pdf/1908.03265).

    This optimizer dynamically adjusts the learning rate using a rectification term
    to stabilize the variance of the adaptive learning rate, especially in early training.

    Attributes:
        lr (float): Learning rate.
        betas (tuple): Coefficients for computing running averages of gradient and its square.
        eps (float): Term added to the denominator to improve numerical stability.
    """
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        """Initialize the RAdam optimizer.

        Args:
            lr (float): Learning rate (default: 1e-3).
            betas (tuple): Coefficients for computing running averages of gradient and its square
                           (default: (0.9, 0.999)).
            eps (float): Term added to the denominator to improve numerical stability (default: 1e-8).

        Raises:
            ValueError: If `lr`, `betas`, or `eps` are invalid.
        """
        # Parameter validation
        if lr <= 0:
            raise ValueError("Learning rate (lr) must be positive.")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError("Betas must be in the range [0, 1).")
        if eps <= 0:
            raise ValueError("Epsilon (eps) must be positive.")

        # Parameter initialization
        self.lr = lr    # Learning rate
        self.betas = betas  # Coefficients for moment estimates
        self.eps = eps  # Small constant for numerical stability

        # Initialize moment estimates (first and second moments)
        self.m = None  # First moment estimate (mean of gradients)
        self.v = None  # Second moment estimate (uncentered variance of gradients)

        # Asymptotic upper bound for the rectification term
        self.rho_inf = 2 / (1 - betas[1]) - 1

    def solve(self, objective_function, params, gradient_function, num_iterations=100):
        """Solve the optimization problem using RAdam optimizer.

        Args:
            objective_function (callable): The objective function to minimize.
            params (list of np.ndarray): List of parameters to optimize.
            gradient_function (callable): Function to compute gradients of the objective function.
            num_iterations (int): Number of optimization iterations (default: 100).

        Returns:
            list of np.ndarray: Optimized parameters.

        Raises:
            TypeError: If `objective_function`, `gradient_function` are invalid.
            ValueError: If `num_iterations` is invalid.
        """
        # Validate objective_function
        if not callable(objective_function):
            raise TypeError("objective_function must be a callable function.")

        # Validate gradient_function
        if not callable(gradient_function):
            raise TypeError("gradient_function must be a callable function.")

        # Validate num_iterations
        if not isinstance(num_iterations, int) or num_iterations <= 0:
            raise ValueError("num_iterations must be a positive integer.")

        # Initialize moment estimates
        self._initialize_moments(params)

        # Optimization loop
        for iteration in range(1, num_iterations + 1):
            # Compute gradients
            grads = list(gradient_function(*params))

            # Perform optimization step
            params = self.step(params, grads, iteration)

            # Print progress
            if (iteration + 1) % 10 == 0:
                loss = objective_function(*params)
                print(f"Iteration {iteration + 1}: x = {params[0][0]:.4f}, y = {params[1][0]:.4f}, Loss = {loss[0]:.4f}")

        return params

    def step(self, params, grads, t):
        """Perform a single optimization step using RAdam.

        Args:
            params (list of np.ndarray): Current parameters.
            grads (list of np.ndarray): Gradients of the objective function with respect to the parameters.
            t (int): Current iteration number (used for bias correction).

        Returns:
            list of np.ndarray: Updated parameters.

        Raises:
            AssertionError: If parameter and gradient shapes do not match.
        """
        # Ensure we have gradient for each parameter (shapes are equal)
        assert len(params) == len(grads), "Parameter and gradient shapes must match."

        # Lazy initialization of moment estimates
        if self.m is None or self.v is None:
            self._initialize_moments(params)

        for i, (param, grad) in enumerate(zip(params, grads)):
            # Ensure gradients and parameters have the same shape (needed for multidimensional parameters such as matricies)
            assert param.shape == grad.shape, "Parameter and gradient shapes must match."

            # Update biased first moment estimate
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad

            # Update biased second moment estimate
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (grad ** 2)

            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.betas[0] ** t)

            # Rectification term
            rho_t = self.rho_inf - 2 * t * (self.betas[1] ** t) / (1 - self.betas[1] ** t)

            if rho_t > 4:
                # Rectified update
                l_t = np.sqrt( (1 - self.betas[1]**t) / (self.v[i] + self.eps) )
                r_t = np.sqrt(
                    ((rho_t - 4) * (rho_t - 2) * self.rho_inf) / ((self.rho_inf - 4) * (self.rho_inf - 2) * rho_t)
                )
                param_update = r_t * m_hat * l_t
            else:
                # Unrectified update (similar to Adam)
                param_update = m_hat

            # Update parameter
            param -= self.lr * param_update

        return params

    def _initialize_moments(self, params):
        """
        Initialize moment estimates (self.m and self.v) to zero vectors.

        Args:
            params (list of np.ndarray): Current parameters.
        """
        self.m = [np.zeros_like(p) for p in params]  # First moment estimate
        self.v = [np.zeros_like(p) for p in params]  # Second moment estimate