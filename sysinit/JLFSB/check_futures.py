import sys, os
from typing import ClassVar
from numpy.lib.function_base import average

import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib
matplotlib.use('Agg')
from syscore.dateutils import month_from_contract_letter
import datetime as dt
import shutil

def scid_to_df(filepath):
    names = 'DateTime','Open','High','Low','Close','NumTrades','TotalVolume','BidVolume','AskVolume'
    offsets = 0, 8, 12, 16, 20, 24, 28, 32, 36
    formats = 'u8', 'f4', 'f4', 'f4', 'f4', 'i4', 'i4', 'i4', 'i4'
    dt = np.dtype({'names':names, 'offsets':offsets, 'formats':formats}, align=True)
    try:
        file = open(filepath, 'rb')
        filetype    = file.read(4)
        headersize  = np.fromfile(file, 'i4', count=1)[0]
        recordSize  = np.fromfile(file, 'i4', count=1)[0]
        version     = np.fromfile(file, 'i2', count=1)[0]


        file.seek(headersize, os.SEEK_SET)

        df = pd.DataFrame(np.fromfile(file, dt))
        if not df.empty:
            df.index = pd.to_datetime(df['DateTime'], origin='1899-12-30', unit='us')
        del df['DateTime']
    except:
        print('Error opening ' + filepath)
        return None

    return df

def sort_by_contract_date(dly_files):
    df= pd.DataFrame()
    for filepath in dly_files:

        # String mangling for contract details
        contract = os.path.split(filepath)[-1].split('.')[0]
          
        yearcode = int(contract.split('-')[0][-2:])
        if yearcode > 50:
            year_int = 1900 + yearcode
        else:
            year_int = 2000 + yearcode

        month_code = contract.split('-')[0][-3]
        month_int = month_from_contract_letter(month_code)
        contract_date = dt.date(year_int, month_int, 1)

        df = df.append({'filepath':filepath, 'contract_date':contract_date}, ignore_index=True)
    df = df.sort_values('contract_date', ascending=True)
    return list(df['filepath'].values)

if __name__ == "__main__":
    futures_config_filepath = 'C:\\Quant\\pysystemtrade\\private\\sierra_futures.csv'
    rawcsv_path     = 'G:\\Sierra'
    results_path    = 'C:\\Quant\\pysystemtrade\\private'
    adjcsv_path     = 'C:\\Quant\\pysystemtrade\\private\\jlfsb_contract_csvs'
    config_path     = 'C:\\Quant\\pysystemtrade\\data\\futures_spreadbet\\csvconfig'

    futures_config_df = pd.read_csv(futures_config_filepath, index_col = 'FutureSymbol')
    total_files = 0

    summary_df = pd.DataFrame(columns=['symbol', 'name', 'scid files', 'bad scid files', 'scid first date', 'scid years', 'scid last date', 'months', 'dly files', 'bad dly files', 'dly first date', 'dly last date', 'dly years', 'avg dly volume', 'ExpiryOffset', 'RollOffsetDays','CarryOffset'])
    detail_df = pd.DataFrame(columns=['symbol','contract', 'new file name', 'filetype', 'start date', 'end date', 'first day', 'last day', 'first wk day', 'last wk day', 'month', 'volume', 'duration', 'ExpiryOffset'])
    volume_df = pd.DataFrame(columns=['F','G','H','J','K','M','N','Q','U','V','X','Z'])
    count_df = pd.DataFrame(columns=['F','G','H','J','K','M','N','Q','U','V','X','Z'])
    roll_config_df = pd.DataFrame()
    price_and_hold_df = pd.DataFrame(columns=['priced', 'hold'])
    symbols_missing_data = []
    for symbol in futures_config_df.index:
        
        #if not symbol == 'HHI?##-HKFE': # debug a specific contract
        #    continue

        print('Starting ' + symbol)

        # Initialise some stuff fresh for this
        months =[]
        bad_scids = 0
        bad_dlys =0
        summary_dict ={}
        volume_df.loc[symbol]=0
        count_df.loc[symbol]=0
        dailyVolume = []
        open_int_dict ={}
        volatility_dict ={}
        maxContractLen = 0
        df_last = None

        # Glob dly files which are associated with this symbol
        search_symbol = symbol.replace('#','?')
        dly_search_path = os.path.join(rawcsv_path, search_symbol + '.dly')
        dly_files = glob(dly_search_path)
        
        dly_files = sort_by_contract_date(dly_files)

        for idx in range(len(dly_files)):
            filepath = dly_files[idx]
            print('Opening ' + filepath)

            # String mangling for contract details
            contract = os.path.split(filepath)[-1].split('.')[0]
          
            yearcode = int(contract.split('-')[0][-2:])
            if yearcode > 50:
                year_int = 1900 + yearcode
            else:
                year_int = 2000 + yearcode

            month_code = contract.split('-')[0][-3]
            month_int = month_from_contract_letter(month_code)
            contract_date = dt.date(year_int, month_int, 1)

            if not month_code in months:
                months += month_code

            # Actually open the dly 
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            # get rid of whitespace in column headers
            df.rename(columns=dict([(col, col.replace(' ','')) for col in df.columns]), inplace=True) 

            if df is None:
                bad_dlys += 1

            if not df.empty:
                dailyVolume+= [df['Volume'].mean()]

                volume_df.loc[symbol][month_code] = volume_df.loc[symbol][month_code] + df['Volume'].sum()
                count_df.loc[symbol][month_code] = count_df.loc[symbol][month_code] + 1

                open_int_dict[contract] = df['OpenInterest'].values

                # fix any weird prices 
                df_len = len(df)
                outliers = (np.abs(stats.zscore(df['Close'])) > 3)
                df[outliers] = np.nan
                df.fillna(method='ffill',inplace=True)
                assert(len(df) ==df_len)

                # Store the voltaility in here for plotting all contracts later
                volatility_dict[contract] = df['Close'].ewm(span=40, ).std()/df['Close'].ewm(span=40,).mean()
                # annualise volatility
                volatility_dict[contract] = volatility_dict[contract] * (np.sqrt(256) / np.sqrt(20))

                # Store this so we can pad the front later and plot with aligned end dates, or close to
                if len(open_int_dict[contract]) > maxContractLen:
                    maxContractLen = len(open_int_dict[contract])

                # write csv out in the right format for Pysystemtrade
                datecode = str(year_int) + "{0:02d}".format(month_int)
                Instrument = futures_config_df.loc[symbol]['Instrument']
                new_file_name = "%s_%s00.csv" % (Instrument, datecode)
                new_file_path = os.path.join(adjcsv_path, new_file_name)


                del df['OpenInterest'] # Psystemtrade doesn't need this
                    # Convert ti IG style pricing
                multiplier = futures_config_df.loc[symbol]['PointSize']
                if futures_config_df.loc[symbol]['Inverse']:
                    df['Open']  = multiplier / df['Open']
                    df['High']  = multiplier / df['High']
                    df['Low']   = multiplier / df['Low']
                    df['Close'] = multiplier / df['Close']
                else:
                    df['Open']  = multiplier * df['Open']
                    df['High']  = multiplier * df['High']
                    df['Low']   = multiplier * df['Low']
                    df['Close'] = multiplier * df['Close']
                    
                if df_last is not None:
                    wanter_overlap = 5
                    gap_start = df_last.index[-1]
                    gap_stop = df.index[0]
                    if ((gap_stop - gap_start).days > -wanter_overlap):
                        # Theres no everlap, create some overlap so that we can roll before the end of the last contract                       
                        # append old values at the front

                        df = df_last.loc[df_last.index[-wanter_overlap:]].append(df)
                        

                df = df[~df.index.duplicated(keep='last')]  # keep any duplicate data that wasn't just appeneded above
                df.sort_index(ascending=True, inplace=True)
                bdate_range = pd.bdate_range(df.index[0], df.index[-1])
                df = df.reindex(bdate_range, method = None)
                df = df.interpolate()
                df.index.name = 'Date'
                assert(df.index.is_monotonic_increasing)
                df.to_csv(new_file_path, float_format='%g')
                df_last = df.copy(deep=True)
                print(f'Wrote {new_file_path} ')

                # Store the detail about this contract
                detail_dict ={}
                detail_dict['symbol']       = symbol
                detail_dict['contract']     = contract
                detail_dict['new file name']= new_file_name
                detail_dict['filetype']     = 'dly'
                detail_dict['start date']   = df.index[0]
                detail_dict['end date']     = df.index[-1]
                detail_dict['first day']    = df.index[0].day
                detail_dict['last day']     = df.index[-1].day
                detail_dict['first wk day'] = df.index[0].weekday()
                detail_dict['last wk day']  = df.index[-1].weekday()
                detail_dict['month']        = month_code
                detail_dict['volume']       = df['Volume'].sum()
                detail_dict['duration']     = int((df.index[-1] - df.index[0]).days)
                if df.index[-1].date() < (dt.date.today()-dt.timedelta(days=5)):
                    detail_dict['ExpiryOffset'] = (df.index[-1].date() - contract_date).days
                else:
                    detail_dict['ExpiryOffset'] = np.nan # exclude end date if the contract is possibly still going
                detail_df = detail_df.append(detail_dict, ignore_index=True)

        future_name = futures_config_df.loc[symbol]['FutureName']   # get this for labelling
        # Plot Open Interest of the life of each contract
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(19.2, 10.9))
        for contract in open_int_dict.keys():
            open_int_dict[contract] = np.pad(open_int_dict[contract], ((maxContractLen-len(open_int_dict[contract]),0),))     
            ax1.plot(open_int_dict[contract][-31:])
            ax2.plot(open_int_dict[contract])
        futureSym = contract.split('-')[0][:-3]
        fig.suptitle('Open Interest' + symbol + ' ' + future_name)
        fig.savefig(os.path.join(results_path, 'OpenInterest', futureSym + '.png'))
        plt.close()

        # Plot rolling 20 day volatility over the life of each contract
        fig = plt.figure(figsize=(19.2, 10.9))
        for contract in volatility_dict.keys():
            volatility_dict[contract] = np.pad(volatility_dict[contract], ((maxContractLen-len(volatility_dict[contract]),0),))     
            plt.plot(volatility_dict[contract])
        mean_vol = pd.DataFrame.from_dict(volatility_dict,orient = 'index').reset_index()
        mean_vol[mean_vol==0] = np.nan
        mean_vol = mean_vol.mean().values
        plt.plot(mean_vol, 'k', linewidth=6)
        futureSym = contract.split('-')[0][:-3]
        fig.suptitle('Rolling volatility' + symbol + ' ' + future_name)
        fig.savefig(os.path.join(results_path, 'Volatility', futureSym + '.png'))
        plt.close()

        # prepare this now we have the data
        priced_months = ''.join(e for e in months)
        num_months = len(priced_months)
        max_contract_interval = int(np.round(365/num_months/30,0)*30)

        # Estimate when  to roll
        roll_vol_threshold = np.nanmedian(mean_vol)
        RollOffsetDays = 0
        while mean_vol[RollOffsetDays-1] < roll_vol_threshold:
            RollOffsetDays-= max_contract_interval

        if RollOffsetDays > -5:
            RollOffsetDays = -5

        # Now we can have a stab at this      
        CarryOffset = -1 if -RollOffsetDays >= max_contract_interval else 1
        
        # Process SCID file similarly
        scid_files =[]
        if False:
            scid_search_path = os.path.join(rawcsv_path, search_symbol + '.scid')
            scid_files = glob(scid_search_path)
        
            for filepath in scid_files:
                print('Opening ' + filepath)

                contract = os.path.split(filepath)[-1].split('.')[0]
                month_code = contract.split('-')[0][-3]
                if not month_code in months:
                    months += month_code

                df = scid_to_df(filepath)

                if df is None:
                    bad_scids += 1
                else:
                    if not df.empty:
                        detail_dict ={}
                        detail_dict['symbol']       = symbol
                        detail_dict['contract']     = contract
                        detail_dict['filetype']     = 'scid'
                        detail_dict['start date']   = df.index[0]
                        detail_dict['end date']     = df.index[-1]
                        detail_dict['first day']    = df.index[0].day
                        detail_dict['last day']     = df.index[-1].day
                        detail_dict['month']        = month_code
                        detail_dict['volume']       = df['TotalVolume'].sum()
                
                        detail_df = detail_df.append(detail_dict, ignore_index=True)

            summary_dict['scid files']         = len(scid_files)
            summary_dict['bad scid files']     = bad_scids
            summary_dict['scid first date']    = detail_df[ (detail_df['symbol']==symbol) & (detail_df['filetype']=='scid') ]['start date'].min().date()
            summary_dict['scid last date']     = detail_df[ (detail_df['symbol']==symbol) & (detail_df['filetype']=='scid') ]['end date'].max().date()    
            summary_dict['scid years']         = round(((summary_dict['scid last date']-summary_dict['scid first date']).days)/365, 1)
       
        
        summary_dict['symbol']             = symbol
        summary_dict['name']               = future_name
        summary_dict['dly files']          = len(dly_files)
        summary_dict['bad dly files']      = bad_dlys
        summary_dict['dly first date']     = detail_df[ (detail_df['symbol']==symbol) & (detail_df['filetype']=='dly') ]['start date'].min().date()
        summary_dict['dly last date']      = detail_df[ (detail_df['symbol']==symbol) & (detail_df['filetype']=='dly') ]['end date'].max().date()
        summary_dict['dly years']          = round(((summary_dict['dly last date']-summary_dict['dly first date']).days)/365, 1)
        summary_dict['avg dly volume']     = round(np.array(dailyVolume).mean(),0)
        summary_dict['months']             = priced_months
        summary_dict['ExpiryOffset']       = round(detail_df[ (detail_df['symbol']==symbol) & (detail_df['filetype']=='dly') ]['ExpiryOffset'].mean())
        summary_dict['RollOffsetDays']     = RollOffsetDays
        summary_dict['CarryOffset']        = CarryOffset

        summary_df = summary_df.append(summary_dict, ignore_index=True)

        total_files += len(scid_files) + len(dly_files)

        detail_df.to_excel(os.path.join(results_path,'detail_df.xlsx'))
        count_df[count_df==0] = np.nan
        
        _volume_df = volume_df / (count_df)

        cols = _volume_df.columns
        priced=''
        hold=''
        price_and_hold_df.loc[symbol]='' # add a new row
        for col in cols:
            if not np.isnan(_volume_df.loc[symbol][col]):
                priced += col
                if _volume_df.loc[symbol][col] > 0.3:
                    hold += col
        price_and_hold_df.loc[symbol]['priced'] = priced
        price_and_hold_df.loc[symbol]['hold'] = hold

        _volume_df = pd.concat([_volume_df, price_and_hold_df], axis=1)
        _volume_df.to_excel(os.path.join(results_path,'volume_df.xlsx'))

        _summary_df = summary_df.copy(deep=True)
        _summary_df.index = _summary_df['symbol']
        _summary_df = pd.concat([_summary_df, price_and_hold_df], axis=1)
        del(_summary_df['symbol'])
        _summary_df.to_excel(os.path.join(results_path,'summary_df.xlsx'))

        ix = _summary_df.index
        # Construct and write roll_config
        roll_config_df = pd.DataFrame()
        roll_config_df['HoldRollCycle']  = _summary_df['hold']
        roll_config_df['RollOffsetDays'] = _summary_df['RollOffsetDays']
        roll_config_df['CarryOffset']    = _summary_df['CarryOffset']
        roll_config_df['PricedRollCycle']= _summary_df['priced']
        roll_config_df['ExpiryOffset']   = _summary_df['ExpiryOffset']
        roll_config_df['Instrument']     = futures_config_df.loc[ix]['Instrument']
        roll_config_df.index             = roll_config_df['Instrument']
        del roll_config_df['Instrument']
        roll_config_df.to_csv(os.path.join(config_path,'JLSB_roll_config.csv'))

        # construct and write instrument_config
        inst_config_df = pd.DataFrame()
        inst_config_df['Instrument']    = futures_config_df.loc[ix]['Instrument']
        inst_config_df['Pointsize']     = futures_config_df.loc[ix]['PointSize']
        inst_config_df['PerTrade']      = 0
        inst_config_df['AssetClass']    = futures_config_df.loc[ix]['AssetClass']
        inst_config_df['PerBlock']      = 0
        inst_config_df['Description']   = futures_config_df.loc[ix]['Description']
        inst_config_df['Percentage']    = 0
        inst_config_df['Currency']      = 'GBP'
        inst_config_df['Slippage']      = futures_config_df.loc[ix]['Slippage']
        inst_config_df.index            = inst_config_df['Instrument']
        del inst_config_df['Instrument']
        inst_config_df.to_csv(os.path.join(config_path,'JLSB_instrument_config.csv'))

    
    print(f'Done. Process {total_files} files')