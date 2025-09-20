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