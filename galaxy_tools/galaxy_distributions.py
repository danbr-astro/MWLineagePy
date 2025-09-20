"""
galaxy_distributions.py
-----------------------
Central and satellite galaxy statistics.

At first, it computes de Stellar Mass Function (SMF) for central and satellite galaxies.

"""
import numpy as np
import pandas as pd

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