from abc import ABC, abstractmethod


class OptimizationAlgorithm(ABC):
    """
    Base class for all optimization algorithms.
    All algorithms should inherit from this class and implement the solve method.
    Output: OptimizationResult
    """
    @abstractmethod
    def solve(self, objective_function, initial_guess, **kwargs):
        """
        Solves the optimization problem.

        :param objective_function: The objective function to minimize.
        :param initial_guess: The initial guess for the optimization.
        :param kwargs: Additional optimizer-specific parameters.
        :return: The result of the optimization.
        """
        pass