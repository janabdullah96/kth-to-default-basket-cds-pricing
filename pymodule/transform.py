
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PseudoSampleToDefaultTimeTransformer:
    """
    Class for transforming pseudo-samples to default times.
    """
    def __init__(self, N, bootstrapper):
        """
        Params:
            N (int): Number of constituents in basket. 
            bootstrapper (pymodule.boostrap.Bootstrapper): a fitted Bootstrapper object.
        """
        self.df_hazard_rates = bootstrapper.df_hazard_rates
        self.df_hazard_rates_cumulative = bootstrapper.df_hazard_rates.cumsum()
        self.df_survival_probabilities = bootstrapper.df_survival_probabilities
        self.N = N
        
    def construct_kth_to_default_times_table(self, df):
        """
        Constructs a DataFrame of default times for each country.

        Params:
            df (pd.DataFrame): Dataframe of historically simulated uniform random variables for each country.

        Returns:
            pd.DataFrame: DataFrame of default times.
        """
        logger.info("Converting pseudo-samples to default times.")
        default_time_frames = []
        df_U_sim_transformed = abs(np.log(1 - df))
        for country in df_U_sim_transformed.columns:
            df_country = df_U_sim_transformed[[country]].copy()
            mask = df_country[country] >= self.df_hazard_rates_cumulative[country].max()
            df_country.loc[mask, country] = None
            df_country.loc[~mask, country] = df_country.loc[~mask, country].map(lambda u: self.__convert_pseudo_sample_to_default_time(u, country))
            default_time_frames.append(df_country)
        df = pd.concat(default_time_frames, axis=1)
        df = df.apply(self.__sort_default_times, axis=1)
        df = round(df, 2)
        df.columns = [i for i in range(1, self.N+1)]
        df.fillna(0, inplace=True)
        return df
    
    def __convert_pseudo_sample_to_default_time(self, u, country):
        """
        Converts uniform distributed pseudo-sample to default time (constrained to 0.25 as the minimum)

        Params:
            u (float): |log(1-u)| of pseudo-sample
            country (string): name of country

        Returns:
            t (float): time to default
        """
        t_m = self.df_hazard_rates_cumulative.loc[self.df_hazard_rates_cumulative[country] >= u].index.min()
        t_m_minus_1 = t_m - 1
        lambda_m = self.df_hazard_rates.loc[self.df_hazard_rates.index == t_m, country].values[0]
        P_m_minus_1 = self.df_survival_probabilities.loc[self.df_survival_probabilities.index == t_m_minus_1, country].values[0]
        delta_t = -(1 / lambda_m) * np.log((1-u) / P_m_minus_1)
        t = t_m_minus_1 + delta_t
        if t >= 0.25 and t <= 5:
            return t
        else:
            return None
    
    @staticmethod
    def __sort_default_times(row):
        """
        Sort default times from earliest to latest going from left to right.
        NULLs are all pushed to the right.

        Params:
            row (list): row vector of default times

        returns:
            pd.Series: sorted row vectors
        """
        sorted_row = sorted(row.dropna()) 
        sorted_row.extend([np.nan] * (len(row) - len(sorted_row))) 
        return pd.Series(sorted_row, index=row.index)
    
