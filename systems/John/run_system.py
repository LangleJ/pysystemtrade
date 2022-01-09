import matplotlib

matplotlib.use("TkAgg")

from syscore.objects import arg_not_supplied

# from sysdata.sim.csv_futures_sim_data import csvFuturesSimData
from sysdata.sim.db_futures_sim_data import dbFuturesSimData
from sysdata.config.configdata import Config

from systems.forecasting import Rules
from systems.basesystem import System
from systems.forecast_combine import ForecastCombine
from systems.provided.rob_system.forecastScaleCap import volAttenForecastScaleCap
from systems.provided.rob_system.rawdata import myFuturesRawData
from systems.positionsizing import PositionSizing
from systems.portfolio import Portfolios
from systems.provided.dynamic_small_system_optimise.portfolio_weights_stage import (
    portfolioWeightsStage,
)
from systems.provided.dynamic_small_system_optimise.optimised_positions_stage import (
    optimisedPositions,
)
from systems.provided.dynamic_small_system_optimise.risk import Risk
from systems.provided.dynamic_small_system_optimise.accounts_stage import (
    accountForOptimisedStage,
)

def futures_system(
    data=None, config=None, trading_rules=arg_not_supplied, log_level="on"
):
 
     if data is None:
        sim_data = dbFuturesSimData()
    
     if config is None:
        config_filename="systems.john.config.yaml"
        config = Config(config_filename)

    system = System(
        [
            Risk(),
            accountForOptimisedStage(),
            optimisedPositions(),
            portfolioWeightsStage(),
            Portfolios(),
            PositionSizing(),
            myFuturesRawData(),
            ForecastCombine(),
            volAttenForecastScaleCap(),
            Rules(),
        ],
        sim_data,
        config,
    )
    system.set_logging_level("on")

    return system

if __name__ == "__main__":
    plots_path = 'C:\\Quant\\pysystemtrade\\systems\\John'
    system = futures_system()
    stats = system.accounts.portfolio().stats()
    system.portfolio.get_instrument_correlation_matrix()

    system.cache.pickle("systems.John.system.pck")

    stats_dict = dict(stats[0])
    print_stats(stats)

    df = system.accounts.portfolio().curve()
    filepath = os.path.join(plots_path, 'PandL.png')
    title = f"PandL, Sharpe:{stats_dict['sharpe']}"
    plot_lines_df(filepath, df, title)

    df = system.portfolio.get_instrument_weights()
    filepath = os.path.join(plots_path, 'InstrumentWeightsTime.png')
    title = f"Instrument Weights Over Time"
    plot_lines_df(filepath, df, title)

    df = system.portfolio.get_instrument_weights().iloc[-1].sort_values()
    filepath = os.path.join(plots_path, 'InstrumentWeights.png')
    title = f"Instrument Weights"
    plot_bars_series(filepath, df, title)

    instruments = system.get_instrument_list()


    
    max_instrument_weight = 0.1
    notional_starting_IDM = 1.0
    minimum_instrument_weight_idm = max_instrument_weight * notional_starting_IDM



