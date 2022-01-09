from syscore.genutils import true_if_answer_is_yes
from syscore.objects import arg_not_supplied

from sysdata.arctic.arctic_futures_per_contract_prices import (
    arcticFuturesContractPriceData,
)
from sysdata.mongodb.mongo_roll_data import mongoRollParametersData
from sysobjects.roll_calendars import rollCalendar
from sysdata.csv.csv_roll_calendars import csvRollCalendarData
from sysproduction.data.prices import get_valid_instrument_code_from_user

from sysdata.mongodb.mongo_futures_instruments import mongoFuturesInstrumentData
from sysdata.mongodb.mongo_roll_data import mongoRollParametersData

import pandas as pd
import os


"""
Generate a 'best guess' roll calendar based on some price data for individual contracts

"""


def build_and_write_roll_calendar(
    instrument_code,
    output_datapath=arg_not_supplied,
    check_before_writing=True,
    input_prices=arg_not_supplied,
    input_config=arg_not_supplied,
):
    status_dict={}

    if output_datapath is arg_not_supplied:
        print(
            "*** WARNING *** This will overwrite the provided roll calendar. Might be better to use a temporary directory!"
        )
    else:
        print("Writing to %s" % output_datapath)

    if input_prices is arg_not_supplied:
        prices = arcticFuturesContractPriceData()
    else:
        prices = input_prices

    if input_config is arg_not_supplied:
        rollparameters = mongoRollParametersData()
    else:
        rollparameters = input_config

    csv_roll_calendars = csvRollCalendarData(output_datapath)

    dict_of_all_futures_contract_prices = prices.get_all_prices_for_instrument(
        instrument_code
    )
    dict_of_futures_contract_prices = dict_of_all_futures_contract_prices.final_prices()
    num_contracts = len(dict_of_futures_contract_prices.keys())
    status_dict['num_contracts'] = num_contracts

    roll_parameters_object = rollparameters.get_roll_parameters(instrument_code)

    # might take a few seconds
    print("Prepping roll calendar... might take a few seconds")
    roll_calendar = rollCalendar.create_from_prices(
        dict_of_futures_contract_prices, roll_parameters_object
    )
    status_dict['num_rolls'] = len(roll_calendar)
    
    mrpd = mongoRollParametersData()
    hold_rollcycles = len(mrpd.get_roll_parameters(instrument_code).hold_rollcycle)
    priced_rollcycles = len(mrpd.get_roll_parameters(instrument_code).priced_rollcycle)
    
    expected_rolls = int(num_contracts * hold_rollcycles/priced_rollcycles)-2
    status_dict['expected_rolls'] = expected_rolls

    # checks - this might fail
    monotonic = roll_calendar.check_if_date_index_monotonic()

    if monotonic is False:
        print('Not Monotonic, fixing')
        ix_to_remove = []
        for ix in range(len(roll_calendar)-1):
            if  roll_calendar.index[ix] == roll_calendar.index[ix+1]:
                ix_to_remove.append(roll_calendar.index[ix])
        roll_calendar.drop(ix_to_remove, inplace = True)
        monotonic = roll_calendar.check_if_date_index_monotonic()
        if monotonic:
            print('Fixed')
        else
            print('NOT Fixed, you need to investigate this properly')

    status_dict['monotonic'] = monotonic

    # this should never fail
    dates_valid = roll_calendar.check_dates_are_valid_for_prices(dict_of_futures_contract_prices)
    status_dict['dates_valid'] = dates_valid

    # Write to csv
    # Will not work if an existing calendar exists

    if check_before_writing:
        check_happy_to_write = true_if_answer_is_yes(
            "Are you ok to write this csv to path %s/%s.csv? [might be worth writing and hacking manually]?"
            % (csv_roll_calendars.datapath, instrument_code)
        )
    else:
        check_happy_to_write = True

    if check_happy_to_write:
        print("Adding roll calendar")
        csv_roll_calendars.add_roll_calendar(
            instrument_code, roll_calendar, ignore_duplication=True
        )
    else:
        print("Not writing")

    if (status_dict['monotonic'] is False) or (status_dict['dates_valid'] is False) or (status_dict['num_rolls'] < ((status_dict['expected_rolls']-1))):
        status_dict['Investigate'] = 'INVESTIGATE'
    else:
        status_dict['Investigate'] = ''
    return status_dict


def check_saved_roll_calendar(
    instrument_code, input_datapath=arg_not_supplied, input_prices=arg_not_supplied
):

    if input_datapath is None:
        print(
            "This will check the roll calendar in the default directory : are you are that's what you want to do?"
        )

    csv_roll_calendars = csvRollCalendarData(input_datapath)

    roll_calendar = csv_roll_calendars.get_roll_calendar(instrument_code)

    if input_prices is arg_not_supplied:
        prices = arcticFuturesContractPriceData()
    else:
        prices = input_prices

    dict_of_all_futures_contract_prices = prices.get_all_prices_for_instrument(instrument_code)
    dict_of_futures_contract_prices = dict_of_all_futures_contract_prices.final_prices()

    print(roll_calendar)

    # checks - this might fail
    roll_calendar.check_if_date_index_monotonic()

    # this should never fail
    roll_calendar.check_dates_are_valid_for_prices(dict_of_futures_contract_prices)

    return roll_calendar


if __name__ == "__main__":

    datapath = 'C:\\Quant\\pysystemtrade\\private\\roll_calendars'
    status_df_filepath = os.path.join(datapath, '_status_df.csv')

    mfid = mongoFuturesInstrumentData()
    all_inst_codes = mfid.get_list_of_instruments()

    inst_code_in  = input("Enter intrument_code or press enter to do them all..")

    if inst_code_in != '':  # Just doing one then...
        inst_codes = []
        chosen_code =''
        for inst_code in all_inst_codes:
            if inst_code.lower().startswith(inst_code_in.lower()):
                resp = input(f'{inst_code} ? Enter for yes, n for no')
                if resp !='n':
                    chosen_code =inst_code
                    inst_codes.append(chosen_code)
                    break;
                else:
                    print('No')
        if chosen_code == '':
            print('Exiting...')
            quit()
        input(f'Doing {chosen_code}, Enter for yes')
    else:   # Do the lot
        inst_codes = all_inst_codes
        input(f'Doing all {len(all_inst_codes)} instruments, enter to continue or CTL+C to abandon')



    status_df = None
    if len(inst_codes) ==1:  # we're only doing one, so update the status file
        if os.path.exists(status_df_filepath):
            status_df = pd.read_csv(status_df_filepath, index_col=0 )

    if status_df is None:   # We're doing them all, start with an empty df
        status_df = pd.DataFrame(columns={'monotonic':bool, 'dates_valid':bool, 'num_contracts':int, 'num_rolls':int, 'expected_rolls':int, 'Investigate':str})
        status_df.index.name = 'instrument_code'

    for instrument in inst_codes:
        status_dict = build_and_write_roll_calendar(instrument, output_datapath=datapath, check_before_writing=False)

        status_df.loc[instrument] = [status_dict[col] for col in status_df.columns]
        #status_df.loc[instrument] = status_dict ## YES, this does need doing twice! https://github.com/pandas-dev/pandas/issues/17072

        
        status_df.to_csv(status_df_filepath)

    num_investigate = (status_df['Investigate']=='INVESTIGATE').sum()
    print(f'Wrote {status_df_filepath}')
    print(f'{num_investigate} Roll Calendars need investigation')
    print('Done')
