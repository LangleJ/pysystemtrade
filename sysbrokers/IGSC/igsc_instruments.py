from dataclasses import dataclass
from sysobjects.instruments import futuresInstrument


@dataclass
class IgInstrumentConfigData:
    IGName        : str =''
    IGMultiplier  : float = 0.0
    IGPointSize   : float = 0.0
    IGMinBet      : float = 0.0
    IGInverse     : bool = False
    IGSlippage    : float = 0.0
    IGFullSlippage: float = 0.0
    IGShortEpic   : str =''
    IGType        : str =''
    IGMargin      : float = 0.0

    def __init__(self, **kwargs):
        for key in kwargs:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])

@dataclass
class ScInstrumentConfigData:
    FutureExchange: str = ''
    FutureSymbol  : str = ''
    FutureName    : str = ''

    def __init__(self, **kwargs):
        for key in kwargs:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])

@dataclass
class FsbInstrumentWithIgConfigData(object):
    '''
        Bolts together 3 data objects
    '''
    instrument: futuresInstrument
    ig_data: IgInstrumentConfigData
    sc_data: ScInstrumentConfigData

    @property
    def instrument_code(self):
        return self.instrument.instrument_code

    @property
    def epic(self):
        return self.ig_data.IGShortEpic

