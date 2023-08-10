
import logging
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from scipy.stats import t, norm, qmc
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

logger = logging.getLogger(__name__)


class Simulate:
    """
    Class for Monte Carlo simulation to generate correlated uniform variables. 
    """
    def __init__(self, copula, n_simulations, countries, N):
        """
        Params:
            copula (pymodule.copula.Copula): a fitted Copula object.
            n_simulations (int): Number of simulations to run.
            countries (list): list of countries.
            N (int): Number of constituents in basket.
        """
        self.copula = copula
        self.n_simulations = n_simulations
        self.countries = countries
        self.N = N
        self.Z = self.__generate_random_numbers()
    
    def gaussian(self, A=None):
        """
        Run MC on Gaussian copula.

        Params:
            A (np.matrix): Custom Cholesky decomposed matrix.

        Returns:
            df_U_sim (pd.DataFrame): simulated uniform variables. 
        """
        logger.info("Running MC simulation using Gaussian copula.")
        if not A:
            A = self.copula.gaussian()
        X = A @ self.Z
        df_U_sim = pd.DataFrame(data=norm.cdf(X), index=self.countries).T
        return df_U_sim
    
    def students_t(self, method, A=None):
        """
        Run MC on Student's t copula.

        Params:
            method (string): Correlation method ("spearman" or "kendall")
            A (np.matrix): Custom correlation matrix.

        Returns:
            df_U_sim (pd.DataFrame): simulated uniform variables. 
        """
        logger.info(f"Running MC simulation using Student's T ({method.title()}) copula.")
        if not A:
            A = self.copula.students_t(method)
        v = self.copula.calibrate_students_t_degrees_of_freedom(A, len(self.countries))
        print(f"Calibrated Student's T - {method.title()} copula to {v} degrees of freedom")
        chi_squared_rv = np.random.chisquare(v, self.n_simulations)
        Y = self.Z / np.sqrt(chi_squared_rv / v)
        X = A @ Y
        df_U_sim = pd.DataFrame(data=t.cdf(X, v), index=self.countries).T
        return df_U_sim
    
    def __generate_random_numbers(self):
        """Generate random numbers using Halton sequencing."""
        robjects.r("""
        library('randtoolbox')
        generate_normal_variables <- function(n, dim) {
            halton_seq <- halton(n, dim)
            halton_seq_t <- t(halton_seq)
            standard_normal_seq <- qnorm(halton_seq_t)
            return(standard_normal_seq)
        }
        """)
        generate_normal_variables = robjects.globalenv["generate_normal_variables"]
        results = generate_normal_variables(self.n_simulations, self.N)
        results = pandas2ri.ri2py(results)
        return results

