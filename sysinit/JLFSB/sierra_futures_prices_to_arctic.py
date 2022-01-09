
from sysdata.csv.csv_futures_contract_prices import ConfigCsvFuturesPrices
import os
import pandas as pd

from syscore.fileutils import (
    files_with_extension_in_resolved_pathname,
    get_resolved_pathname,
)
from syscore.dateutils import month_from_contract_letter

from sysinit.futures.contract_prices_from_csv_to_arctic import (
    init_arctic_with_csv_futures_contract_prices,
)

sierra_csv_config = ConfigCsvFuturesPrices(
    input_date_index_name="Date",
    input_skiprows=0,
    input_skipfooter=1,
    input_date_format="%Y-%m-%d",
    input_column_mapping=dict(
        OPEN="Open", HIGH="High", LOW="Low", FINAL="Close", VOLUME="Volume"
    ),
)


def transfer_sierra_prices_to_arctic(datapath):

    init_arctic_with_csv_futures_contract_prices(
        datapath, csv_config=sierra_csv_config
    )


if __name__ == "__main__":
    input("Will overwrite existing prices are you sure?! CTL-C to abort")
    # modify flags as required
    datapath = 'C:\\Quant\\pysystemtrade\\private\\jlfsb_contract_csvs'

    transfer_sierra_prices_to_arctic(datapath)
    print('Done')