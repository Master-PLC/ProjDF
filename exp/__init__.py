from .exp_long_term_forecasting import Exp_Long_Term_Forecast
from .exp_long_term_forecasting_cca_loss import Exp_Long_Term_Forecast_CCA_Loss
from .exp_long_term_forecasting_meta_reptile_byturn import Exp_Long_Term_Forecast_META_Reptile_Byturn
from .exp_long_term_forecasting_meta_reptile import Exp_Long_Term_Forecast_META_Reptile
from .exp_long_term_forecasting_meta_mamlpp import Exp_Long_Term_Forecast_META_MAMLPP
from .exp_long_term_forecasting_meta import Exp_Long_Term_Forecast_META
from .exp_long_term_forecasting_meta_imaml import Exp_Long_Term_Forecast_META_iMAML
from .exp_long_term_forecasting_meta_ml3 import Exp_Long_Term_Forecast_META_ML3
from .exp_long_term_forecasting_ot import Exp_Long_Term_Forecast_OT
from .exp_short_term_forecasting import Exp_Short_Term_Forecast
from .exp_long_term_forecasting_gan import Exp_Long_Term_Forecast_GAN
from .exp_long_term_forecasting_iter import Exp_Long_Term_Forecast_Iter
from .exp_long_term_forecasting_vali import Exp_Long_Term_Forecast_Vali
from .exp_long_term_forecasting_vali2 import Exp_Long_Term_Forecast_Vali2

EXP_DICT = {
    'long_term_forecast': Exp_Long_Term_Forecast,
    'short_term_forecast': Exp_Short_Term_Forecast,
    'long_term_forecast_meta_reptile_byturn': Exp_Long_Term_Forecast_META_Reptile_Byturn,
    'long_term_forecast_meta_reptile': Exp_Long_Term_Forecast_META_Reptile,
    'long_term_forecast_meta_mamlpp': Exp_Long_Term_Forecast_META_MAMLPP,
    'long_term_forecast_meta_imaml': Exp_Long_Term_Forecast_META_iMAML,
    'long_term_forecast_meta': Exp_Long_Term_Forecast_META,
    'long_term_forecast_meta_ml3': Exp_Long_Term_Forecast_META_ML3,
    'long_term_forecast_ot': Exp_Long_Term_Forecast_OT,
    'long_term_forecast_cca_loss': Exp_Long_Term_Forecast_CCA_Loss,
    'long_term_forecast_gan': Exp_Long_Term_Forecast_GAN,
    'long_term_forecast_iter': Exp_Long_Term_Forecast_Iter,
    'long_term_forecast_vali': Exp_Long_Term_Forecast_Vali,
    'long_term_forecast_vali2': Exp_Long_Term_Forecast_Vali2,
}