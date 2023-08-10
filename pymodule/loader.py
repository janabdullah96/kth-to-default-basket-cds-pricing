
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class Loader:
    """
    Class for loading datasets.
    """
    def __init__(self, countries, spread_factor=1, curve_factor=1):
        """
        Params:
            countries (list): Names of countries to load
            spread_factor (float): Factor to apply to all historical CDS spreads
            curve_factor (float): factor to apply to all credit curve points
        """
        self.countries = countries
        self.df_cds_historical = None
        self.df_curves = None
        self.df_discount_curve = None
        self.spread_factor = spread_factor
        self.curve_factor = curve_factor
        
    def fit(self):
        """Set attributes"""
        self.__load_historical_cds_spreads()
        self.__load_credit_curves()
        self.__load_discount_curve()
        
    def __load_historical_cds_spreads(self):
        logger.info("Loading historical CDS spreads.")
        historical_spreads_dir = "data/historical_spreads/"
        frames = []
        for country in self.countries:
            df = pd.read_csv(f"{historical_spreads_dir}{country}.csv")
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            frames.append(df)
        df = pd.DataFrame(index=set([j for i in [df_country.index.dropna() for df_country in frames] for j in i]))
        for df_country in frames:
            df = df.join(df_country, how="left")
        df = df.resample("W").last()
        df *= self.spread_factor
        self.df_cds_historical = df
    
    def __load_credit_curves(self):
        logger.info("Loading credit curves.")
        df = pd.read_csv("data/cds_curves.csv").set_index("tenor(y)") / 10000
        df.index = df.index.astype(int)
        df = df.loc[df.index >= 1].copy()
        df *= self.curve_factor
        self.df_curves = df
    
    def __load_discount_curve(self):
        logger.info("Loading discount curve.")
        df = pd.read_csv("data/yield_curve.csv").set_index("tenor(y)") / 100
        df.loc[:, "discount_factor"] = np.exp(-df["zero_cpn_yield"] * df.index)
        self.df_discount_curve = df
    