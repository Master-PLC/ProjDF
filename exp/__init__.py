from .exp_long_term_forecasting import Exp_Long_Term_Forecast
from .exp_long_term_forecasting_cca import Exp_Long_Term_Forecast_CCA
from .exp_long_term_forecasting_cycle import Exp_Long_Term_Forecast_Cycle
from .exp_long_term_forecasting_meta import Exp_Long_Term_Forecast_META
from .exp_long_term_forecasting_ot import Exp_Long_Term_Forecast_OT
from .exp_short_term_forecasting import Exp_Short_Term_Forecast
from .exp_long_term_forecasting_cca_loss import Exp_Long_Term_Forecast_CCA_Loss
from .exp_long_term_forecasting_cca_cycle_loss import Exp_Long_Term_Forecast_CCA_Cycle_Loss

EXP_DICT = {
    'long_term_forecast': Exp_Long_Term_Forecast,
    'long_term_forecast_cycle': Exp_Long_Term_Forecast_Cycle,
    'short_term_forecast': Exp_Short_Term_Forecast,
    'long_term_forecast_cca': Exp_Long_Term_Forecast_CCA,
    'long_term_forecast_meta': Exp_Long_Term_Forecast_META,
    'long_term_forecast_ot': Exp_Long_Term_Forecast_OT,
    'long_term_forecast_cca_loss': Exp_Long_Term_Forecast_CCA_Loss,
    'long_term_forecast_cca_cycle_loss': Exp_Long_Term_Forecast_CCA_Cycle_Loss,
}