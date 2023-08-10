
import sys
import logging
import pandas as pd
from argparse import ArgumentParser
from pymodule.loader import Loader
from pymodule.bootstrap import Bootstrapper
from pymodule.copula import Copula
from pymodule.transform import PseudoSampleToDefaultTimeTransformer
from pymodule.simulate import Simulate
from pymodule.price import Pricer

pd.set_option("display.max_columns", None)  
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.max_colwidth", None)

def main(
    n_simulations=10000, 
    RR=0.4, 
    correlation_factor=1, 
    spread_factor=1, 
    curve_factor=1, 
    display=False, 
    return_df=False
    ):
    countries = ["japan", "china", "thailand", "singapore", "malaysia"]
    N = len(countries)
    print("Running CDS kth-to-default basket pricing.")
    print(f"""
    Number of Simulations: {n_simulations} 
    RR: {RR} 
    Correlation Factor: {correlation_factor} 
    Spread Factor: {spread_factor} 
    Curve Factor: {curve_factor}
    """)
    loader = Loader(countries=countries, spread_factor=spread_factor, curve_factor=curve_factor)
    loader.fit()
    bootstrapper = Bootstrapper(loader=loader, RR=RR)
    bootstrapper.fit()
    copula = Copula(loader=loader, countries=countries, N=N, correlation_factor=correlation_factor)
    copula.fit()
    simulate = Simulate(copula=copula, n_simulations=n_simulations, countries=countries, N=N)
    transformer = PseudoSampleToDefaultTimeTransformer(N=N, bootstrapper=bootstrapper)
    pricer = Pricer(RR=RR, N=N, bootstrapper=bootstrapper)
    df_gaussian_spreads = simulate.gaussian(
    ).pipe(
        transformer.construct_kth_to_default_times_table
    ).pipe(
        pricer.construct_fair_spread_table,
        distribution="gaussian"
    )
    df_t_spearman_spreads = simulate.students_t(
        method="spearman"
    ).pipe(
        transformer.construct_kth_to_default_times_table
    ).pipe(
        pricer.construct_fair_spread_table,
        distribution="students_t_spearman"
    )
    df_t_kendall_spreads = simulate.students_t(
        method="kendall"
    ).pipe(
        transformer.construct_kth_to_default_times_table
    ).pipe(
        pricer.construct_fair_spread_table,
        distribution="students_t_kendall"
    )
    df = pd.concat([df_gaussian_spreads, df_t_spearman_spreads, df_t_kendall_spreads])
    if display:
        df_display = df.applymap(lambda x: str(round(x*10000, 2)) + " bps" if pd.notnull(x) else " ")
        print(f"\n{'='*50}\nFinalized Spread Pricing Table\n", df_display)
    if return_df:
        return df

def setup_logging():
    logger = logging.getLogger(__name__)
    if logger.hasHandlers():
        return logger
    format_ = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        format=format_,
        datefmt=date_format,
        level=logging.DEBUG)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(
        logging.Formatter(
            fmt=format_,
            datefmt=date_format))
    logger.addHandler(stream_handler)
    return logger


if __name__ == "__main__":
    logger = setup_logging()
    parser = ArgumentParser()
    parser.add_argument("-n_sims", 
                        "--n_simulations", 
                        nargs=1,
                        type=int, 
                        default=[10000],
                        help="Enter number of simulations to run")
    parser.add_argument("-rr", 
                        "--RR", 
                        nargs=1,
                        type=float, 
                        default=[0.4],
                        help="Enter standard recovery rate to use")
    parser.add_argument("-cf", 
                        "--correlation_factor", 
                        nargs=1,
                        type=float, 
                        default=[1],
                        help="Enter correlation factor to use")
    parser.add_argument("-sf", 
                        "--spread_factor", 
                        nargs=1,
                        type=float, 
                        default=[1],
                        help="Enter spread factor to use")
    parser.add_argument("-crvf", 
                        "--curve_factor", 
                        nargs=1,
                        type=float, 
                        default=[1],
                        help="Enter curve factor to use")
    parsed_args = vars(parser.parse_args())
    n_simulations = parsed_args.get("n_simulations")[0]
    RR = parsed_args.get("RR")[0]
    correlation_factor = parsed_args.get("correlation_factor")[0]
    spread_factor = parsed_args.get("spread_factor")[0]
    curve_factor = parsed_args.get("curve_factor")[0]
    try:
        main(
            n_simulations=n_simulations, 
            RR=RR, 
            correlation_factor=correlation_factor,
            spread_factor=spread_factor,
            curve_factor=curve_factor,
            display=True
            )
    except Exception as e:
        logger.exception(e)

