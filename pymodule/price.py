
import logging
import numpy as np
import pandas as pd
from scipy import interpolate

logger = logging.getLogger(__name__)


class Pricer:
    """
    Class for pricing default and premium legs using simulated default times.
    """
    def __init__(self, RR, N, T, bootstrapper):
        """
        Params:
            RR (float): Recovery rate.
            N (int): Number of constituents in basket.
            T (int): Max tenor.
            bootstrapper (pymodule.boostrap.Bootstrapper): a fitted Bootstrapper object.
        """
        self.RR = RR
        self.discount_curve = bootstrapper.interpolated_discount_curve
        self.N = N
        self.n = 1 / self.N
        self.T = T
        self.discounted_full_premium = self.discount_curve.get(self.T) * self.T

    def construct_fair_spread_table(self, df, distribution):
        """
        Constructs table of fair spreads using with and without removal methodologies for premium leg.

        Params:
            df (pd.DataFrame): DataFrame containing default times data.
            distribution (string): The type of distribution being used. For labeling purposes. 

        Returns:
            pd.DataFrame: Dataframe of fair spreads.
        """
        logger.info(f"Constructing {distribution.replace('_', ' ').title()} fair spread table")
        frames = []
        for k in df.columns:
            default_leg_pv = self.__price_default_leg(df.copy(), k)
            spread_without_removal = default_leg_pv / self.__price_premium_leg_without_removal(df.copy(), k) 
            if k > 1:
                spread_with_removal = default_leg_pv / self.__price_premium_leg_with_removal(df.copy(), k) 
            else:
                spread_with_removal = None
            df_country = pd.DataFrame(
                index=["spread_without_removal", "spread_with_removal"], 
                data=[spread_without_removal, spread_with_removal],
                columns=[k]
            )
            frames.append(df_country)
        df_spreads = pd.concat(frames, axis=1)
        df_spreads.index = pd.MultiIndex.from_product(
            [[distribution], ["spread_without_removal", "spread_with_removal"]],
            names=["distribution_type", "premium_leg_spread_type"]
        )
        return df_spreads

    def __price_default_leg(self, df, k):
        """
        Params:
            df (pd.DataFrame): DataFrame containing default times data.
            k (float or int): kth to default swap to price
        Returns:
            float: The price of the default leg.
        """
        mask = df[k] != 0
        df.loc[mask, k] = df.loc[mask, k].map(lambda t: (1 - self.RR) * self.discount_curve.get(t) * self.n)
        return np.mean(df[k])

    def __price_premium_leg_without_removal(self, df, k):
        """
        Params:
            df (pd.DataFrame): DataFrame containing default times data.
            k (float or int): kth to default swap to price
        Returns:
            float: The price of the premium leg without removal.
        """
        mask = df[k] == 0
        df.loc[mask, k] = self.discounted_full_premium
        df.loc[~mask, k] = df.loc[~mask, k].map(lambda t: self.discount_curve.get(t) * t)
        return np.mean(df[k])

    def __price_premium_leg_with_removal(self, df, k):
        """
        Params:
            df (pd.DataFrame): DataFrame containing default times data.
            k (float or int): kth to default swap to price
        Returns:
            float: The price of the premium leg with removal.
        """
        mask = df[k] == 0
        df_with_default_index = df.loc[~mask].index
        df.loc[mask, k] = self.discounted_full_premium
        for i, row in df.iloc[df_with_default_index].iterrows():
            t = row[k]
            ts = [0] + [row[kn] for kn in range(1, k + 1)]
            PL = 0
            NP = 1
            for i, t in enumerate(ts[1:], 1):
                discount_factor = self.discount_curve.get(t)
                marginal_default_time = t - ts[i - 1]
                PL += discount_factor * marginal_default_time * NP
                NP -= self.n
            df.loc[df.index == i, k] = PL
        return np.mean(df[k])

