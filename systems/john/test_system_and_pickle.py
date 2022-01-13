from spreadbetting_system import spreadbettingSystem
from syscore.fileutils import get_filename_for_package, get_resolved_pathname
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from systems.john.analyse_estimated_system import plot_lines_df, plot_bars_series

if __name__ == "__main__":
    plots_path                  = 'private.spreadbettingSystem_100k_test.plots'
    postions_path               = 'private.spreadbettingSystem_100k_test.plots.positions'

    pickle_filepath             = 'private.spreadbettingSystem_100k_test.test_system.pkl'
    config_path                 = "systems.john.config.yaml"
    forecast_weights_filepath   = "systems.john.forecast_weights.yaml"      # forecast_weights:
    forecast_scalars_filepath   = "systems.john.forecast_scalars.yaml"      # forecast_scalars:
    inst_weights_filepath       = "systems.john.inst_weights.yaml"          # instrument_weights:
    forcast_div_mult_filepath   = "systems.john.forecast_div_mult.yaml"     # forecast_div_multiplier:

    system = spreadbettingSystem([config_path, forecast_weights_filepath, forecast_scalars_filepath, inst_weights_filepath, forcast_div_mult_filepath])
    
    stats = system.accounts.portfolio().stats()
    system.portfolio.get_instrument_correlation_matrix()

    system.cache.pickle(pickle_filepath)

    # Make some folders
    plots_path = get_resolved_pathname(plots_path)
    if not os.path.exists(plots_path):
        os.mkdir(plots_path)

    postions_path = get_resolved_pathname(postions_path)
    if not os.path.exists(postions_path):
        os.mkdir(postions_path)


    # Get some iterables
    stats_dict = dict(stats[0])
    instruments = system.get_instrument_list()
    rule_variations = system.config.trading_rules.keys()

    # Plot PandL
    df = system.accounts.portfolio().curve()
    filepath = os.path.join(plots_path, 'PandL.png')
    title = f"PandL, Sharpe:{stats_dict['sharpe']}"
    plot_lines_df(filepath, df, title)

    # plot the position over time for each instrument
    for instrument in instruments:
        df = pd.DataFrame()
        notional_posn = system.portfolio.get_notional_position(instrument)
        notional_posn.name= 'notional position'
        buffers = system.portfolio.get_buffers_for_position(instrument)
        buffered_posn = system.accounts.get_buffered_position(instrument)
        buffered_posn.name = 'buffered position'
        df = pd.concat([notional_posn, buffers, buffered_posn], axis=1)
        filepath = os.path.join(postions_path, f'Postion_{instrument}.png')
        title = instrument
        plot_lines_df(filepath, df, title)

    # plot the sharpe ratio for each instrument
    inst_sharpe_series = pd.Series()
    for instrument in instruments:
        sharpe = system.accounts.pandl_for_instrument(instrument).sharpe()
        inst_sharpe_series.loc[instrument] = sharpe
    inst_sharpe_series.sort_values(ascending=False, inplace=True)
    median_inst_sharpe = inst_sharpe_series.median()
    filepath = os.path.join(plots_path, 'Instrument Sharpe.png')
    plot_bars_series(filepath, inst_sharpe_series, f'Instrument Sharpe Ratios, median is {median_inst_sharpe}')

    print('Done')