"""
galaxy_distributions.py
-----------------------
Central and satellite galaxy statistics.

At first, it computes de Stellar Mass Function (SMF) for central and satellite galaxies.

"""
import numpy as np
import pandas as pd
from joblib import Parallel,delayed

#===============================
# Auxiliar Functions
#===============================
def bin_midpoints(bin_array):
    """
    This function obtains the midpoints of the bin array returned by np.histogram function.
    :param bin_array: Array with the bin edges.
    :return: half_point_array: Array with the half points for each bin
    """
    hollow_list = list()
    for i in range(len(bin_array)-1):
        hollow_list.append((bin_array[i]+bin_array[i+1])/2)
    return hollow_list

#==============================
# Stellar Mass Function SMF
#==============================
def compute_smf(logmste_array):
    """
    This function computes the Stellar Mass Function using histograms normalized by the bin width (dex) and the total
    number of galaxies.
    :param logmste_array: Pandas series that contains all the stellar mass of the central or satellite galaxies.
    :return smf_df: DataFrame with the SMF. Columns ---> ['log_Mste','SMF']
    """
    totnum_gal = len(logmste_array)
    n_bins = 15
    max_logmste = logmste_array.max()
    min_logmste = logmste_array.min()
    dex = (max_logmste-min_logmste)/n_bins
    # Computing the Histogram
    count_array, dex_array = np.histogram(logmste_array,bins= n_bins, range= (min_logmste,max_logmste))
    # Getting the midpoints of dex_array
    midpoints_array = bin_midpoints(dex_array)
    # Creating SMF DataFrame
    smf_df = pd.DataFrame({
        'log_Mste':midpoints_array,
        'SMF': count_array/(totnum_gal*dex)
    })
    return smf_df
#=========================================================================
# Cumulative Number of Satellite Galaxies. (Satellite Galaxies CSMF)
#=========================================================================
def mean_gal_csmf(subhalos_df,logmste_threshold):
    """
    This function computes the cumulative number os¿f satellite galaxies as a function of the stellar mass.
    :param subhalos_df: DataFrame of the subhalos and therefor, their satellite galaxies.
    :param logmste_threshold: Stellar mass threshold for completeness.
    :return mean_galaxy_csmf: DataFrame with the mean CSMF. --> Columns: ['log_Mste','mean_galCSMF']
    """
    # The computation of the average is in function on the total number of host halos.
    totnum_hosthalos = len(subhalos_df['Halo_id'].unique())
    logmste_array = np.linspace(logmste_threshold,subhalos_df['Subhalo_logMste'].max(),20)
    hollow_list = list()
    for logmste in logmste_array:
        auxiliar_df = subhalos_df[subhalos_df['Subhalo_logMste'] >= logmste]
        n_gal = len(auxiliar_df['Subhalo_logMste'])
        mean_csmf = n_gal/totnum_hosthalos
        hollow_list.append(mean_csmf)
    mean_galaxy_csmf = pd.DataFrame({
        'log_Mste' : logmste_array,
        'mean_galCSMF': hollow_list
    })
    return mean_galaxy_csmf

def ind_gal_csmf(subhalos_df):
    """
    This function computes for each host halo its CSMF.
    :param subhalos_df: Dataframe with the data of subhalos and their centarl galaxies (for subhalos).
    :return individual_galaxy_csmf: DataFrame with all the individual CSMF from each host halo.
    Columns ---> ['Halo_id','log_Mste','ind_galCSMF']
    """
    def halo_csmf(halo_id):
        hollow_haloid = list()
        hollow_logmste = list()
        hollow_csmf = list()
        auxiliar_df = subhalos_df[subhalos_df['Halo_id'] == halo_id].sort_values(by= 'Subhalo_logMste')
        for logmste in auxiliar_df['Subhalo_logMste']:
            n_gal = len(auxiliar_df[auxiliar_df['Subhalo_logMste'] >= logmste])
            hollow_haloid.append(halo_id)
            hollow_logmste.append(logmste)
            hollow_csmf.append(n_gal)
        partial_df = pd.DataFrame({
            'Halo_id': hollow_haloid,
            'log_Mste': hollow_logmste,
            'ind_galCSMF': hollow_csmf
        })
        return partial_df
    results = Parallel(n_jobs= 8)(delayed(halo_csmf)(halo_id) for halo_id in subhalos_df['Halo_id'].unique())
    # We concat each DataFrame vertically
    individual_galaxy_csmf = pd.concat(results,axis= 0, ignore_index= True)
    return individual_galaxy_csmf

def gal_csmf_std(mean_csmf_array,subhalos_df):
    """
    This function computes the standard deviation for the csmf, creating again and individual csmf
    but with uniform spacing in logmste array.
    :param mean_csmf_array: DataFrame with the mean csmf for all the system
    :param subhalos_df: DataFrame with the satellite galaxies log stellar mass
    :return None: It adds to mean_csmf_array 3 columns ---> ['Sigma','up_std','below_std']
    """
    logmste_array = mean_csmf_array['log_Mste'].values
    # We compute the individual csmf for each galaxy but with logmste_array
    def halo_csmf(halo_id):
        hollow_ngal = list()
        auxiliar_df = subhalos_df[subhalos_df['Halo_id'] == halo_id]
        for logmste in logmste_array:
            n_gal = len(auxiliar_df[auxiliar_df['Subhalo_logMste'] >= logmste])
            hollow_ngal.append(n_gal)
        df_row = pd.DataFrame([hollow_ngal],columns=logmste_array)
        return df_row
    results = Parallel(n_jobs=8)(delayed(halo_csmf)(halo_id) for halo_id in subhalos_df['Halo_id'].unique())
    galaxy_csmf = pd.concat(results,axis= 0,ignore_index= True)
    # Now we can compute the standard deviation for the same value logmste
    hollow_std = list()
    for logmste in logmste_array:
        auxiliar_column = galaxy_csmf[logmste].values
        std = auxiliar_column.std(ddof=0)
        avg = auxiliar_column.mean()
        hollow_std.append(std)
    mean_csmf_array['Sigma'] = hollow_std
    # We use np.maximum to stablish a floor value
    mean_csmf_array['up_std'] = mean_csmf_array['mean_galCSMF'] + mean_csmf_array['Sigma']
    mean_csmf_array['below_std'] = np.maximum(mean_csmf_array['mean_galCSMF'] - mean_csmf_array['Sigma'],10**(-3))

