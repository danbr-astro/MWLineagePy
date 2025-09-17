"""
halo_distributions.py
=====================
Halo and subhalo statistics.

This module computes the synthetic HMF for our Halo Mock Catalog.

"""
import numpy as np
import pandas as pd


#===============================================
# Compute synthetic HMF
#===============================================
def mass_bin_centers(bin_array):
    """
    This function obtains the midpoints of the bin array returned by np.histogram function.
    :param bin_array: Array with the bin edges.
    :return: half_point_array: Array with the half points for each bin
    """
    hollow_list = list()
    for i in range(len(bin_array)-1):
        hollow_list.append((bin_array[i]+bin_array[i+1])/2)
    return hollow_list

def compute_hmf(halos_df,comoving_volume):
    """
    This function computes the synthetic HMF using histograms normalized by the bin width and comoving volume,
    yielding to a PDF.
    :param halos_df: DataFrame that contains the log10 virial mass of the halos. ['Halo_logMvir']
    :param comoving_volume: Comoving volume of the mock universe.
    :return: hmf_df: DataFrame with the computed HMF with 2 columns; ['Halo_logMvir','HMF']
    """
    n_bins = 15
    min_logmvir = halos_df['Halo_logMvir'].min()
    max_logmvir = halos_df['Halo_logMvir'].max()
    dex = (max_logmvir-min_logmvir)/n_bins
    # Compute histogram
    counts_array, bin_array = np.histogram(halos_df['Halo_logMvir'],bins = n_bins, range = (min_logmvir,max_logmvir))
    # Obtaining the midpoints of bin_array
    midpoints_array = mass_bin_centers(bin_array)
    # Create HMF DataFrame
    hmf_df = pd.DataFrame({
        'Halo_logMvir' : midpoints_array,
        'HMF' : counts_array/(dex*comoving_volume)
    })
    return hmf_df