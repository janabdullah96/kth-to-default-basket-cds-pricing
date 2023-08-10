
import logging
import numpy as np
import pandas as pd
from scipy import interpolate

logger = logging.getLogger(__name__)


class Bootstrapper:
    """
    Class for bootstrapping and interpolating term structures and curves.
    """
    def __init__(self, loader, RR):
        """
        Params:
            loader (pymodule.loader.Loader): a fitted Loader object.
            RR (float): Recovery rate.
        """
        self.df = loader.df_curves
        self.df_discount_curve = loader.df_discount_curve
        self.RR = RR
        self.countries = self.df.columns
        self.df_formatted = self.__format_df()
        self.df_survival_probabilities = None
        self.df_hazard_rates = None
        self.interpolated_discount_curve = None
    
    def fit(self):
        """Set attributes"""
        self.__construct_survival_probabilities_term_structure_table()
        self.__construct_hazard_rates_term_structure_table(self.df_survival_probabilities)
        self.__interpolate_discount_curve()

    def __construct_hazard_rates_term_structure_table(self, df):
        """
        Construct hazard rates term structure table for all countries.

        Params:
            df (pd.DataFrame): Dataframe containing survival probability term structures for different countries.

        Returns:
            pd.DataFrame: Hazard rates term structure table.
        """
        logger.info("Constructing hazard rates term structure table.")
        frames = []
        df["dt"] = [0] + np.diff(df.index).tolist()
        for country in self.countries:
            df_hazard_rates = df[["dt", country]].rename(columns={country: "survival_probability"})
            df_hazard_rates = pd.DataFrame(
                data=self.__bootstrap_hazard_rates(df_hazard_rates),
                index=df.index,
                columns=[country]
            )
            frames.append(df_hazard_rates)
        self.df_hazard_rates = pd.concat(frames, axis=1)

    def __construct_survival_probabilities_term_structure_table(self):
        """
        Construct survival probabilities term structure table for all countries.

        Returns:
            pd.DataFrame: Dataframe containing the survival probabilities for different countries.
        """
        logger.info("Constructing survival probabilities term structure table.")
        frames = []
        for country in self.countries:
            df = self.df_formatted[["tenor(y)", country, "discount_factor", "dt"]].rename(columns={country: "spread"})
            df_survival_probabilities = pd.DataFrame(
                data=self.__bootstrap_survival_probabilities(df).tolist(),
                index=self.df_formatted.index,
                columns=[country]
            )
            frames.append(df_survival_probabilities)
        self.df_survival_probabilities = pd.concat(frames, axis=1)
    
    def __interpolate_discount_curve(self, col="discount_factor", T=5, t_min=0.25):
        """
        Interpolates the discount curve using log-linear interpolation.

        Parameters:
            col (str, optional): Name of the column containing the discount factors. Defaults to "discount_factor".
            T (int, optional): Maximum time value for interpolation. Defaults to 5.
            t_min (float, optional): Minimum time value for interpolation. Defaults to 0.25.

        Returns:
            Dict[float, float]: A dictionary representing the continuously interpolated discount curve.
                                Keys are time values (rounded to 2 decimal places),
                                and values are corresponding discount factors.
        """
        logger.info("Interpolating discount curve with log-linear interpolation.")
        f = interpolate.interp1d(self.df_discount_curve.index, np.log(self.df_discount_curve[col]), kind="linear")
        t = t_min
        interpolated_discount_curve = {}
        while t <= T:
            interpolated_discount_curve[round(t, 2)] = np.exp(f(t))
            t += 0.01
        self.interpolated_discount_curve = interpolated_discount_curve

    def __format_df(self):
        """
        Format the input dataframe by adding necessary columns and sorting the data.

        Returns:
            pd.DataFrame: Formatted dataframe.
        """
        df = self.df.join(self.df_discount_curve, how="left")
        df = pd.concat(
            [pd.DataFrame([[0] * len(df.columns)], columns=df.columns), df],
            axis=0
        )
        df.index.name = "tenor(y)"
        df.reset_index(inplace=True)
        df.sort_values(by="tenor(y)", inplace=True)
        df["dt"] = df["tenor(y)"].diff().fillna(0)
        return df

    def __bootstrap_survival_probabilities(self, df):
        """
        Bootstrap survival probabilities for all tenors.

        Params:
            df (pd.DataFrame): Dataframe containing spreads, discount factors, time steps and tenor mark for given country.

        Returns:
            pd.Series: Series of survival probabilities for the given country.
        """
        L = 1 - self.RR
        df.loc[df.index == 0, "survival_probability"] = 1
        df.loc[df.index == 1, "survival_probability"] = L / (L + df["dt"] * df["spread"])
        for T, row in df.loc[df.index > 1].iterrows():
            T = int(T)
            # create dictionary and store variables from previous tenors for easy lookups
            tenor_vars = {}
            for i, row in df.loc[df.index <= T].iterrows():
                tenor_vars[i] = {}
                for col in ["spread", "discount_factor", "dt", "survival_probability"]:
                    tenor_vars[i][col] = row[col]

            term1_numerator = 0
            term1_denominator = tenor_vars[T]["discount_factor"] * (L + tenor_vars[T]["dt"] * tenor_vars[T]["spread"])
            term2_numerator = tenor_vars[T - 1]["survival_probability"] * L
            term2_denominator = L + tenor_vars[T]["dt"] * tenor_vars[T]["spread"]

            for n in range(1, T):
                term1_numerator += tenor_vars[n]["discount_factor"] * (
                        L * tenor_vars[n - 1]["survival_probability"]
                        - (L + tenor_vars[n]["dt"] * tenor_vars[T]["spread"])
                        * tenor_vars[n]["survival_probability"]
                )

            survival_probability = (term1_numerator / term1_denominator) + (term2_numerator / term2_denominator)
            df.loc[df.index == T, "survival_probability"] = survival_probability
        return df["survival_probability"]

    @staticmethod
    def __bootstrap_hazard_rates(df):
        """
        Get hazard rates using the log-ratio of survival probabilities.

        Params:
            df (pd.DataFrame): Dataframe containing dt and survival probabilities for given country.

        Returns:
            pd.Series: Series of piecewise constant hazard rates for the given country.
        """
        return (-1 / df["dt"]) * np.log(df["survival_probability"] / df["survival_probability"].shift(1))

