import os
import pandas as pd
import numpy as np

from spreadbetting_system import spreadbettingSystem
from syscore.fileutils import get_filename_for_package

if __name__ == "__main__":
    plots_path = 'private.spreadbettingSystem.plots'
    pickle_filepath = 'private.spreadbettingSystem.system.pkl'
    config_path = "systems.john.estimation_config.yaml"

    plots_path = get_filename_for_package(plots_path)
    if not os.path.exists(plots_path):
        os.mkdir(plots_path)

    system = spreadbettingSystem(config_path)
    stats = system.accounts.portfolio().stats()
    system.portfolio.get_instrument_correlation_matrix()

    system.cache.pickle(pickle_filepath)

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



