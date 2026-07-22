from scipy.constants import codata
lightvel = codata.value('speed of light in vacuum')

def corr_waves_RV(waves, RV):
    # positive RV means redshift (receding from star) -> correction is a blue shift meaning that wavelengths become shorter
    lambda_corr = waves/(RV/lightvel +1.0)
    return lambda_corr