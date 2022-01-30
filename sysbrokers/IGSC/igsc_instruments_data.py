import pandas as pd
from syscore.fileutils import get_filename_for_package
from syscore.genutils import value_or_npnan
from sysobjects.instruments import futuresInstrument
from sysbrokers.broker_instrument_data import brokerFuturesInstrumentData
from syslogdiag.log_to_screen import logtoscreen
from syscore.objects import missing_instrument, missing_file
from sysbrokers.IGSC.igsc_instruments import ScInstrumentConfigData
from sysbrokers.IGSC.igsc_instruments import IgInstrumentConfigData
from sysbrokers.IGSC.igsc_instruments import FsbInstrumentWithIgConfigData

IGSC_CONFIG_FILE = get_filename_for_package(
    "private.IGSC.IGSC_config.csv"
)


class IgScConfig(pd.DataFrame):
    pass


def read_igsc_config_from_file() -> IgScConfig:
    df = pd.read_csv(IGSC_CONFIG_FILE, index_col='Instrument')
    return IgScConfig(df)


class IgscFsbInstrumentData(brokerFuturesInstrumentData):
    def __init__(self, log=logtoscreen("IgscFsbInstrumentData")):
        super().__init__(log=log)

    def __repr__(self):
        return "IG SC Fsb per contract data"

    def get_brokers_instrument_code(self, instrument_code: str) -> str:
        if instrument_code.endswith("_fsb"):
            instrument_code = instrument_code.split('_')[0]
        igsc_fsb_instrument = self.get_igsc_fsb_instrument(instrument_code)
        return igsc_fsb_instrument.bc_symbol

    def get_instrument_code_from_broker_code(self, bc_code: str) -> str:
        config = self._get_igsc_config()
        config_row = config[config.Symbol == bc_code]
        if len(config_row) == 0:
            msg = f"BC symbol {bc_code} not found in config"
            self.log.critical(msg)
            raise Exception(msg)

        if len(config_row) > 1:
            msg = f"BC symbol {bc_code} appears more than once, likely an FSB, picking first one"
            self.log.msg(msg)
            # TODO total hack - to be fixed properly
            return config_row.iloc[0].Instrument

        return config_row.iloc[0].Instrument

    def _get_instrument_data_without_checking(self, instrument_code: str):
        return self.get_igsc_fsbs_instrument(instrument_code)

    def get_igsc_fsb_instrument(self, instr_code: str) -> FsbInstrumentWithIgConfigData:
        new_log = self.log.setup(instrument_code=instr_code)

        try:
            assert instr_code in self.get_list_of_instruments()
        except Exception:
            new_log.warn(f"Instrument {instr_code} is not in BC config")
            return missing_instrument

        config = self._get_igsc_config()
        if config is missing_file:
            new_log.warn(
                f"Can't get config for instrument {instr_code} as BC config file missing"
            )
            return missing_instrument

        instrument_object = get_instrument_object_from_config(instr_code, config=config)

        return instrument_object

    def get_list_of_instruments(self) -> list:
        """
        Get instruments that have price data
        Pulls these in from a config file

        :return: list of str
        """

        config = self._get_igsc_config()
        if config is missing_file:
            self.log.warn(
                "Can't get list of instruments because IGSC config file missing"
            )
            return []

        instrument_list = list(config.index)

        return instrument_list

    # Configuration read in and cache
    def _get_igsc_config(self) -> IgScConfig:
        config = getattr(self, "_config", None)
        if config is None:
            config = self._get_and_set_igsc_config_from_file()

        return config

    def _get_and_set_igsc_config_from_file(self) -> IgScConfig:

        try:
            config_data = read_igsc_config_from_file()
        except BaseException:
            self.log.warn(f"Can't read file IGSC_CONFIG_FILE")
            config_data = missing_file

        self._config = config_data

        return config_data

    def _delete_instrument_data_without_any_warning_be_careful(
        self, instrument_code: str
    ):
        raise NotImplementedError("IGSC instrument config is read only")

    def _add_instrument_data_without_checking_for_existing_entry(
        self, instrument_object
    ):
        raise NotImplementedError("IGSC instrument config is read only")


def get_instrument_object_from_config(
    instr_code: str, config: IgScConfig = None
) -> FsbInstrumentWithIgConfigData:

    if config is None:
        config = read_igsc_config_from_file()

    inst_config = config.loc[instr_code].to_dict()

    instrument = futuresInstrument(instr_code)
    ig_data = IgInstrumentConfigData(**inst_config)
    sc_data = ScInstrumentConfigData(**inst_config)

    fsb_instrument_with_igsc_data = FsbInstrumentWithIgConfigData(instrument, ig_data, sc_data)

    return fsb_instrument_with_igsc_data
