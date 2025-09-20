"""
galaxy_mw_data.py
-----------------
Satellite galaxies stellar masses.

This module contains the DataFrame that contains the stellar masses ( M_sun) of observed satellite mw galaxies.

References:
    - McConnachie et al. 2012, The Astronomical Journal, 144, 4.
"""
import numpy as np
import pandas as pd
def mw_sat():
    data={'Sat_Name':['Canis Major','Sagittarius dSph','Segue (I)','Ursa Major II','Bootes II','Segue II','William I','Coma Berenices','Bootes III',
                      'LMC','SMC','Bootes (I)','Draco','Ursa Minor','Sculptor','Sextans (I)','Ursa Major (I)','Carina','Hercules','Fornax','Leo IV',
                      'Canes Venatici II','Leo V','Pisces II','Canes Venatici (I)','Leo II','Leo I','Phoenix','NGC 6822','Leo T'],
          'Mstar':[49,21,0.00034,0.0041,0.0010,0.00086,0.0010,0.0037,0.017,     #------->(x10⁶)
                 1500,460,0.029,0.29,0.29,2.3,0.44,0.014,0.38,0.037,20,0.019,
                 0.0079,0.011,0.0086,0.23,0.74,5.5,0.77,100,0.14]}
    df=pd.DataFrame(data)
    df['log_Mstar'] = np.log10(df['Mstar']*(10**6))
    df = df.sort_values(by= 'log_Mstar')
    return df
