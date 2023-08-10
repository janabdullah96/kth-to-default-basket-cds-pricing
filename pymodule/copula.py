
import logging
import math
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import scipy.special as sp
import matplotlib.pyplot as plt
import pymodule.grapher as gp
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from scipy.stats import norm, t
from IPython.display import display, Markdown

logger = logging.getLogger(__name__)


class Copula:
    """
    Class to build copulae. 
    """
    def __init__(self, loader, countries, N, display=False, correlation_factor=1):
        """
        Params:
            loader (pymodule.loader.Loader): a fitted Loader object.
            countries (list): list of countries.
            N (int): Number of constituents in basket.
        """
        self.df = loader.df_cds_historical
        self.countries = countries
        self.df_X_hist = None
        self.df_U_hist = None
        self.df_Z_hist = None
        self.display = display
        self.correlation_factor = correlation_factor
        self.__display_headers = ['' for _ in range(N)]
    
    def gaussian(self):
        """Run Cholesky decomposition on Gaussian correlation matrix"""
        logger.info("Running Cholesky decomposition on Gaussian correlation matrix")
        corr = self.df_Z_hist.corr(method="pearson")
        if self.correlation_factor != 1:
            corr = self.__factor_correlation_matrix(corr)
        A = np.linalg.cholesky(corr)
        if self.display:
            self.__display_matrices(corr, A, "Gaussian")
        return A
    
    def students_t(self, method):
        """
        Run Cholesky decomposition on Student's T correlation matrix

        Params:
            method (string): Correlation method ("spearman" or "kendall")
        """
        logger.info(f"Running Cholesky decomposition on Student's T {method.title()} correlation matrix")
        if method.lower() == "spearman":
            corr = self.df_U_hist.corr(method=method).applymap(lambda p: 2*math.sin((math.pi/6)*p))
        elif method.lower() == "kendall":
            corr = self.df_X_hist.corr(method=method).applymap(lambda p: math.sin((math.pi/2)*p))
        else:
            raise KeyError
        if self.correlation_factor != 1:
            corr = self.__factor_correlation_matrix(corr)
        A = np.linalg.cholesky(corr)
        if self.display:
            self.__display_matrices(corr, A, f"Student's T - {method.title()}")
        return A
    
    def calibrate_students_t_degrees_of_freedom(self, A, N, v_lower=1, v_upper=25):
        """
        Calibrates the degrees of freedom (v) for Student's t copula.

        Parameters:
            A (np.matrix): Cholesky decomposed correlation matrix
            v_lower (int, optional): The lower bound of the degrees of freedom to consider. Defaults to 1.
            v_upper (int, optional): The upper bound of the degrees of freedom to consider. Defaults to 25.
            N (int): Number of countries

        Returns:
            int: The calibrated degrees of freedom (v).
        """
        logger.info(f"Calibrating degrees of freedom for Student's T copula using MLE.")
        v_dict = {}
        for v in range(v_lower, v_upper):
            v_dict[v] =  np.sum(self.df_U_hist.apply(lambda row: self.__calculate_log_c(row, A, v, N), axis=1))[0][0]
        v = max(v_dict, key=v_dict.get)
        if self.display:
            df = pd.DataFrame(v_dict, index=["log-likelihood"]).T
            gp.graph(
                df=df,
                cols=df.columns,
                kind=plt.plot,
                figsize=(9.5, 4),
                title=r"Degrees of freedom calibration",
                xlabel="v",
                ylabel="Log-likelihood",
                lw=0.75
            )
        return v
            
    def fit(self):
        """Set attributes"""
        self.__set_X_hist()
        self.__set_U_hist()
        self.__set_Z_hist()
    
    def __set_X_hist(self):
        """Set historical CDS changes dataframe as attribute"""
        logger.info("Setting dataframe of historical CDS spread changes.")
        self.df_X_hist = self.df.diff().dropna().apply(self.__impute_outliers)
    
    def __set_U_hist(self):
        """Convert historical CDS changes to uniformly distributed data and set as attribute"""
        logger.info("Converting historical CDS spread changes to pseudo samples with KS Density estimation.")
        U_hist_dict = {country: self.__ksdensity(self.df_X_hist[country].dropna())
                       for country in self.countries}
        self.df_U_hist = pd.DataFrame(U_hist_dict, index=self.df_X_hist.index)
        
    def __set_Z_hist(self):
        """Convert uniformly distributed data to standard normal and set as attribute"""
        logger.info("Converting pseudo samples of historical CDS spread changes to standard normal.")
        self.df_Z_hist = self.df_U_hist.apply(norm.ppf)
    
    @staticmethod
    def __impute_outliers(series, lower_limit=0.01, upper_limit=0.99):
        """
        Impute the outliers of the series with the mean of the subseries.
        I.e on the positive change side, impute outlier with the mean of all positive changes
        and on the negative change side, impute outlier with the mean of all negative changes.

        Params:
            series (pd.Series): the series to clean
            lower_limit (float, optional): The lower percentile values to impute. Defaults to 1%.
            upper_limit (float, optional): The upper percentile values to impute. Defaults to 99%. 

        returns:
            pd.Series: Cleaned series. 
        """
        series[series <= series.quantile(lower_limit)] = series[series < 0].mean()
        series[series >= series.quantile(upper_limit)] = series[series > 0].mean()
        return series
    
    @staticmethod
    def __ksdensity(data):
        """
        Run KS Density estimation with R 'ks' library. 

        Params:
            data (pd.Series): series to run KS Density estimation on.
        
        Returns:
            result (series): series with transformed data.
        """
        pandas2ri.activate()
        r_function = robjects.r(
            """
            library('ks')
            pseudo.samples = function(X){
                Fhat <- ks::kcde(X, h=0.001)
                return(predict(Fhat, x=X))
            }
            """
        )
        r_vector = robjects.FloatVector(data)
        result = r_function(r_vector)
        result = pandas2ri.ri2py(result)
        return result
    
    @staticmethod
    def __calculate_log_c(row, A, v, N):
        """
        Calculate the logarithm of the c value based on the given parameters.

        Args:
            row (pd.Dataframe row): Sample uniform variables for each country.
            A (np.matrix): Cholesky decomposed correlation matrix
            v (float): Degrees of freedom
            N (int): Number of countries

        Returns:
            float: The logarithm of the c value.
        """
        U = np.matrix(row)
        term1 = 1 / np.sqrt(np.linalg.det(A))
        term2 = sp.gamma((v + N) / 2) / sp.gamma(v / 2)
        term3 = (sp.gamma(v / 2) / sp.gamma((v + 1) / 2)) ** N
        term4_numerator = (1 + ((t.ppf(U, v) @ np.linalg.inv(A) @ t.ppf(U.T, v)) / v)) ** -((v + N) / 2)
        term_4_denominator = 1
        for u in np.array(U)[0]:
            factor = (1 + ((t.ppf(u, v)**2) / v)) ** -((v + 1) / 2)
            term_4_denominator*=factor
        term4 = term4_numerator / term_4_denominator
        log_c = np.log(term1 * term2 * term3 * term4)
        return log_c
    
    def __factor_correlation_matrix(self, df_corr):
        """
        Bump up or down all values in a correlation matrix (except for the diagonal)
        by specified factor set during instantiation. 
        (Factor of 1 means no change)

        Params:
            df_corr (pd.DataFrame): Original correlation matrix
        
        Returns:
            pd.DataFrame: Updated correlation matrix
        """
        corr = df_corr.values.copy()
        upper_tri_indices = np.triu_indices(corr.shape[0], k=1)
        lower_tri_indices = np.tril_indices(corr.shape[0], k=-1)
        corr[upper_tri_indices] *= self.correlation_factor
        corr[lower_tri_indices] *= self.correlation_factor
        df_corr_updated = pd.DataFrame(corr, columns=df_corr.columns, index=df_corr.index)
        df_corr_updated = df_corr_updated.applymap(lambda x: 0.99 if x > 1 else x)
        df_corr_updated = df_corr_updated.applymap(lambda x: -0.99 if x < -1 else x)
        return df_corr_updated

    def __display_matrices(self, corr, A, method):
        display(Markdown(f"\n#### Correlation Matrix (*{method}*)"))
        display(corr)
        display(Markdown(f"\n#### Cholesky Decomposed Matrix (*{method}*)"))
        display(pd.DataFrame(A, index=self.__display_headers, columns=self.__display_headers))
