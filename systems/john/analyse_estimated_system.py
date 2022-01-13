
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

from spreadbetting_system import spreadbettingSystem
from syscore.fileutils import get_filename_for_package, get_resolved_pathname

def plot_lines_df(png_filepath, df, title):
    fig = plt.figure()
    plt.plot(df)
    if isinstance(df, pd.DataFrame):
        plt.legend(df.columns)
    plt.title(title)
    fig.savefig(png_filepath)
    plt.close()
    print(f'Wrote: {png_filepath}\n')

def plot_bars_series(png_filepath, df, title):
    assert(df, pd.Series)
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    y_pos = np.arange(len(df.index))
    ax.barh(y_pos, df)
    ax.set_yticks(y_pos, labels=df.index)
    plt.title(title)
    fig.savefig(png_filepath)
    plt.close()
    print(f'Wrote: {png_filepath}\n')

def output_inst_rule_cost_csv(system, instruments, rule_variations, filename_p):
    print('Outputting csv with costs for each trading rule')
    filename = get_filename_for_package( filename_p )
    rule_costs_df = pd.DataFrame(columns = rule_variations)
    for instrument in instruments:
        inst_rule_costs = []
        inst_dict ={}
        for rule in rule_variations:
            cost = system.accounts.get_SR_cost_for_instrument_forecast(instrument, rule)
            inst_rule_costs.append(cost)
        rule_costs_df.loc[instrument] = inst_rule_costs
    rule_costs_df.to_csv(filename)
    print(f'Wrote: {filename}\n')
    return rule_costs_df

def output_forecast_weights_yaml(rule_cost_df, max_inst_sr_cost, max_rule_sr_cost, hc_fc_weights_filepath, fc_weight_yaml_p):
    print('Outputting forecast weights as a yaml file') 
    hc_fc_weights_filepath = get_filename_for_package( hc_fc_weights_filepath_p )
    hc_fc_weights_df = pd.read_csv(hc_fc_weights_filepath, index_col=0)['weights']
    
    fc_weight_yaml = get_filename_for_package( fc_weight_yaml_p )
    
    forecast_wgt_dict = {}
    bad_markets = []
    for instrument in instruments:
        instr_median = rule_costs_df.loc[instrument].median()
        if instr_median > max_inst_sr_cost:    
            bad_markets.append(instrument)
            
        inst_dict = {}
        rule_weights = rule_costs_df.loc[instrument]
        bad_rules = rule_weights[rule_weights > max_rule_sr_cost].index
        rule_weights.loc[bad_rules] = 0
        weight_sum = rule_weights.sum()
        if weight_sum > 0:
            rule_weights_norm = (rule_weights/ weight_sum).round(4)      
        else:
            rule_weights_norm[:] = 0

        forecast_wgt_dict.update({instrument:rule_weights_norm.to_dict()})

    forecast_wgt_dict = {'forecast_weights':forecast_wgt_dict}
    forecast_wgt_dict['bad_markets']= bad_markets
    wgt_yaml = yaml.dump(forecast_wgt_dict)
    with open(fc_weight_yaml, 'w') as f:
        f.write(wgt_yaml)
    print(f'Wrote: {fc_weight_yaml}\n')

def output_forecast_scalars_yaml(rule_variations, instruments, system):
    print('Outputting forecast scalars as a yaml file')
    forecast_scalar_dict = {}
    for rule in rule_variations:
        series = pd.Series()
        for instrument in instruments:
            series.loc[instrument] = system.forecastScaleCap.get_forecast_scalar(instrument, rule).mean() # or last??
        forecast_scalar_dict[rule] = round(np.float(series.median()), 4)
    forecast_scalar_dict = {'forecast_scalars' : forecast_scalar_dict}
    forecast_scalar_yaml = yaml.dump(forecast_scalar_dict)
    forecast_scalar_yaml_filepath = get_filename_for_package(forecast_scalar_yaml_filepath_p)
    with open(forecast_scalar_yaml_filepath, 'w') as f:
        f.write(forecast_scalar_yaml)
    print(f'Wrote: {forecast_scalar_yaml_filepath}\n')

def output_inst_weights_yaml(hc_inst_weights_filepath_p, inst_weight_yaml_filepath_p):
    print('Outputting instrument weights as a yaml file')
    inst_weights_dict = {}
    hc_inst_weights_filepath = get_filename_for_package( hc_inst_weights_filepath_p )
    hc_inst_weights_df = pd.read_csv(hc_inst_weights_filepath, index_col=0)['weights'].round(4)
    hc_inst_weights_dict = hc_inst_weights_df.to_dict()
    hc_inst_weights_dict = {'instrument_weights' : hc_inst_weights_dict}
    hc_inst_weights_yaml = yaml.dump(hc_inst_weights_dict)
    inst_weight_yaml_filepath = get_filename_for_package(inst_weight_yaml_filepath_p)
    with open(inst_weight_yaml_filepath, 'w') as f:
        f.write(hc_inst_weights_yaml)
    print(f'Wrote: {inst_weight_yaml_filepath}\n')

def output_forecast_div_mult(system, instruments, forcast_div_mult_filename_p):
    print('Outputting forcast Diversification Multilier as yaml')
    forecast_div_mult_dict = {}
    for instrument in instruments:
        forecast_div_mult = system.combForecast.get_forecast_diversification_multiplier(instrument)[-1]
        forecast_div_mult_dict[instrument] = round(np.float(forecast_div_mult), 4)
    forecast_div_mult_dict = {'forecast_div_multiplier' : forecast_div_mult_dict}
    forecast_div_mult_yaml = yaml.dump(forecast_div_mult_dict)
    forcast_div_mult_filename = get_filename_for_package(forcast_div_mult_filename_p)
    with open(forcast_div_mult_filename, 'w') as f:
        f.write(forecast_div_mult_yaml)
    print(f'Wrote: {forcast_div_mult_filename}\n')

if __name__ == "__main__":
    # filepaths to output to
    plots_path                      = 'private.spreadbettingSystem_100k.plots'
    postions_path                   = 'private.spreadbettingSystem_100k.plots.positions'
    pickle_filepath                 = 'private.spreadbettingSystem_100k.estimated_system.pkl'
    rule_costs_df_filename_p        = "private.spreadbettingSystem_100k.rule_costs.csv"
    
    config_path                     = "systems.john.estimation_config.yaml"
    hc_fc_weights_filepath_p        = "systems.john.handcraft_rule_weights.csv"  
    fc_weight_yaml_p                = "systems.john.forecast_weights.yaml"  
    forecast_scalar_yaml_filepath_p = "systems.john.forecast_scalars.yaml"
    hc_inst_weights_filepath_p      = "systems.john.handcraft_inst_weights.csv"
    inst_weight_yaml_filepath_p     = "systems.john.inst_weights.yaml"
    
    forecast_div_mult_filepath_p    = "systems.john.forecast_div_mult.yaml"     # forecast_div_multiplier:

    # Make some folders
    plots_path = get_resolved_pathname(plots_path)
    if not os.path.exists(plots_path):
        os.mkdir(plots_path)

    postions_path = get_resolved_pathname(postions_path)
    if not os.path.exists(postions_path):
        os.mkdir(postions_path)

    # create the system object and load the pickle
    system = spreadbettingSystem(config_path)
    system.cache.unpickle(pickle_filepath)

    # Get some stats
    stats = stats = system.accounts.portfolio().stats()
    stats_dict = dict(stats[0])
    instruments = system.get_instrument_list()
    rule_variations = system.config.trading_rules.keys()

    # Output csv with costs for each trading rule
    rule_costs_df = output_inst_rule_cost_csv(system, instruments, rule_variations, rule_costs_df_filename_p)

    # Output forecast weights as a yaml file
    max_inst_sr_cost = 0.2
    max_rule_sr_cost = 0.1
    output_forecast_weights_yaml(rule_costs_df, max_inst_sr_cost, max_rule_sr_cost, hc_fc_weights_filepath_p, fc_weight_yaml_p)
    
    # Output forecast scalars as a yaml file
    output_forecast_scalars_yaml(rule_variations, instruments, system)

    # Output instrument weights as a yaml file
    output_inst_weights_yaml(hc_inst_weights_filepath_p, inst_weight_yaml_filepath_p)

    # Output forcast Diversification Multilier as yaml
    output_forecast_div_mult(system, instruments, forecast_div_mult_filepath_p)
    
    # Plot PandL
    df = system.accounts.portfolio().curve()
    filepath = os.path.join(plots_path, 'PandL.png')
    title = f"PandL, Sharpe:{stats_dict['sharpe']}"
    plot_lines_df(filepath, df, title)

    # plot instruments weights estmate over time
    df = system.portfolio.get_instrument_weights()
    filepath = os.path.join(plots_path, 'InstrumentWeightsTime.png')
    title = f"Instrument Weights Over Time"
    plot_lines_df(filepath, df, title)

    df = system.portfolio.get_instrument_weights().iloc[-1].sort_values()
    filepath = os.path.join(plots_path, 'InstrumentWeights.png')
    title = f"Instrument Weights"
    plot_bars_series(filepath, df, title)

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

    # plot the sharpe ratio for each trading rule
    w_rule_sharpe_series = pd.Series()
    for rule in rule_variations:
        sharpe = system.accounts.pandl_for_trading_rule_weighted(rule).sharpe()
        w_rule_sharpe_series.loc[instrument] = sharpe
        w_rule_sharpe_series.sort_values(ascending=False, inplace=True)
        filepath = os.path.join(plots_path, 'Weighted Trading Rule Sharpe.png')
        plot_bars_series(filepath, w_rule_sharpe_series, 'Weighted Trading Rule Sharpe Ratios')

   ## plot the (crude and innacurate!) sharpe ratio for each trading rule
   #uw_rule_sharpe_series = pd.Series()
   #for rule in rule_variations:
   #    sharpe = system.accounts.pandl_for_trading_rule_unweighted(rule).sharpe()
   #    uw_rule_sharpe_series.loc[rule] = sharpe
   #    uw_rule_sharpe_series.sort_values(ascending=False, inplace=True)
   #    filepath = os.path.join(plots_path, 'Unweighted Trading Rule Sharpe.png')
   #    plot_bars_series(filepath, uw_rule_sharpe_series, 'Unweighted Trading Rule Sharpe Ratios')





