"""
halo_distributions.py
=====================
Halo and subhalo statistics.

This module computes the synthetic HMF for our Halo Mock Catalog. It also computes the distribution as a function of
the virial mass and its expected value. More over, it computes de Cumulative Number of Subhalos, its mean value and
its standard deviation.

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


#===============================================
# Compute synthetic HMF
#===============================================
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
    midpoints_array = bin_midpoints(bin_array)
    # Create HMF DataFrame
    hmf_df = pd.DataFrame({
        'Halo_logMvir' : midpoints_array,
        'HMF' : counts_array/(dex*comoving_volume)
    })
    return hmf_df
#=========================================================================
# Halo virial mass distribution. (Distribution as a function of halo mass)
#=========================================================================
def compute_halomass_distribution(logmvir_array):
    """
    This function computes the distribution as a function of halo mass.
    :param logmvir_array: pandas series that contains all the virial mass in their logarithmic form.
    :return mean_logmvir, distribution_df: The expected value of the PDF and a DataFame with the PDF of the halo virial mass.
    Columns ---> ['log_Mvir','PDF']
    """
    mean_logmvir = logmvir_array.mean()
    totnum_halos = len(logmvir_array)
    nbins = 20
    max_logmvir = logmvir_array.max()
    min_logmvir = logmvir_array.min()
    bin_width = (max_logmvir-min_logmvir)/nbins
    # Computing the histogram
    count_array, bins_array = np.histogram(logmvir_array,bins= nbins, range= (min_logmvir,max_logmvir))
    # Getting the midpoints of each bin
    midpoints_array = bin_midpoints(bins_array)
    # Creating DataFrame
    distribution_df = pd.DataFrame({
        'log_Mvir':midpoints_array,
        'PDF':count_array/(totnum_halos*bin_width)
    })
    return mean_logmvir, distribution_df
#======================================================================
# Cumulative Number of Subhalos. (Subhalo CSMF)
#======================================================================
def mean_subh_csmf(subhalos_df):
    """
    This function computes the mean subhalo csmf.
    :param subhalos_df: Dataframe with the virial masses of the subhalos.
    :return mean_subhalo_csmf: Dataframe with the subhalo csmf. ---> Columns: ['log_Mvir','mean_subhaloCSMF']
    """
    totnum_halos = len(subhalos_df['Halo_id'].unique())
    logmvir_array = np.linspace(subhalos_df['Subhalo_logMvir'].min(),subhalos_df['Subhalo_logMvir'].max(),20)
    hollow_list = list()
    for logmvir in logmvir_array:
        auxiliar_df = subhalos_df[subhalos_df['Subhalo_logMvir'] >= logmvir]
        n_subhalo = len(auxiliar_df['Subhalo_logMvir'])
        mean_csmf = n_subhalo/totnum_halos
        hollow_list.append(mean_csmf)
    mean_subhalo_csmf = pd.DataFrame({
        'log_Mvir': logmvir_array,
        'mean_subhaloCSMF': hollow_list
    })
    return mean_subhalo_csmf

def ind_subh_csmf(subhalos_df):
    """
    This function computes the individual CSMF that is to say, compute for each host halo
    its cumulative number of satellites.
    :param subhalos_df: DataFrame with subhalos logarithmic virial mass for thc csmf
    :return individual_subhalo_csmf: DataFrame with all the csmf for each halo:
    Columns ---> ['Halo_id','log_Mvir','ind_subhCSMF']
    """
    def halo_csmf(halo_id):
        hollow_haloid = list()
        hollow_logmvir = list()
        hollow_csmf = list()
        auxiliar_df = subhalos_df[subhalos_df['Halo_id'] == halo_id].sort_values(by='Subhalo_logMvir')
        for logmvir in auxiliar_df['Subhalo_logMvir']:
            n_subh = len(auxiliar_df[auxiliar_df['Subhalo_logMvir'] >= logmvir])
            hollow_haloid.append(halo_id)
            hollow_logmvir.append(logmvir)
            hollow_csmf.append(n_subh)
        partial_df = pd.DataFrame({
            'Halo_id': hollow_haloid,
            'log_Mvir': hollow_logmvir,
            'ind_subhCSMF': hollow_csmf
        })
        return partial_df
    results = Parallel(n_jobs=8)(delayed(halo_csmf)(halo_id) for halo_id in subhalos_df['Halo_id'].unique())
    individual_subhalo_csmf = pd.concat(results,axis=0, ignore_index= True)
    return individual_subhalo_csmf

def subh_csmf_std(mean_csmf_df,subhalos_df):
    """
    This function computes the standard deviation for the subhalo CSMF.
    :param mean_csmf_df: DataFrame with the subhalo mean csmf
    :param subhalos_df: DataFrame with the subhalo virial masses
    :return None: It adds 3 columns to mean_csmf_df ---> ['Sigam','up_std','below_std']
    """
    logmvir_array = mean_csmf_df['log_Mvir'].values
    # We compute the individual csmf for each galaxy but with logmste_array
    def halo_csmf(halo_id):
        hollow_nsubh = list()
        auxiliar_df = subhalos_df[subhalos_df['Halo_id'] == halo_id]
        for logmvir in logmvir_array:
            n_subh = len(auxiliar_df[auxiliar_df['Subhalo_logMvir'] >= logmvir])
            hollow_nsubh.append(n_subh)
        df_row = pd.DataFrame([hollow_nsubh], columns=logmvir_array)
        return df_row
    results = Parallel(n_jobs=8)(delayed(halo_csmf)(halo_id) for halo_id in subhalos_df['Halo_id'].unique())
    subhalo_csmf = pd.concat(results, axis=0, ignore_index=True)
    # Now we can compute the standard deviation for the same value logmste
    hollow_std = list()
    for logmste in logmvir_array:
        auxiliar_column = subhalo_csmf[logmste].values
        std = auxiliar_column.std(ddof=0)
        hollow_std.append(std)
    mean_csmf_df['Sigma'] = hollow_std
    # We use np.maximum to stablish a floor value
    mean_csmf_df['up_std'] = mean_csmf_df['mean_subhaloCSMF'] + mean_csmf_df['Sigma']
    mean_csmf_df['below_std'] = np.maximum(mean_csmf_df['mean_subhaloCSMF'] - mean_csmf_df['Sigma'], 10 ** (-3))

def mw_subhcsmf(mwanalog_xis,subhalos_df):
    """
    This function computes the 'real' MW subhalo CSMF using the 5% of the mock satellite galaxies of lower Xi²
    to the observed MW CSMF using McConnachie observed data. For this purpose, it uses the subhalos that host
    that satellite galaxies to computes its subhalo CSMF.
    :param mwanalog_xis: DataFrame with the halo_ids of the host halos with satellite galaxies with lower Xi².
    :param subhalos_df: DataFrame with subhalo virial masses neede to comute the CSMF
    :return mw_subhalo_csmf: DataFrame with the observed MW subhalo CSMF
    """
    ntot_halos = len(mwanalog_xis['Halo_id'])
    # We filter those systems were their Xi² are in the 5%
    auxiliar_df1 = subhalos_df[subhalos_df['Halo_id'].isin(mwanalog_xis['Halo_id'])]
    n_bins=20
    logmvir_array = np.linspace(auxiliar_df1['Subhalo_logMvir'].min(),auxiliar_df1['Subhalo_logMvir'].max(),n_bins)
    hollow_cmsf = list()
    for logmvir in logmvir_array:
        auxiliar_df2 = auxiliar_df1[auxiliar_df1['Subhalo_logMvir'] >= logmvir]
        n_subh = len(auxiliar_df2['Subhalo_logMvir'])
        subh_csmf = n_subh/ntot_halos
        hollow_cmsf.append(subh_csmf)
    mw_subhalo_csmf = pd.DataFrame({
        'log_Mvir':logmvir_array,
        'mw_subhCSMF': hollow_cmsf
    })
    return mw_subhalo_csmf

def csmf_slope(csmf_df,param):
    """
    This function computes the csmf function slope for the MW analog subhalo csmf (if param = 0) or for the
    MW like subhalo csmf (if param != 0)
    :param csmf_df: DataFrame with MW-like o MW-analog CSMF
    :param param: Selection (0 or anything) to select wich slope the function is going to compute
    :return m: Slope value
    """
    if param == 0:
        y = csmf_df['mw_subhCSMF'].values
        x = csmf_df['log_Mvir'].values
        m = (y[y.argmax()] - y[y.argmin()]) / (x[y.argmax()] - x[y.argmin()])
        return m
    else:
        y = csmf_df['mean_subhaloCSMF'].values
        x = csmf_df['log_Mvir'].values
        m = (y[y.argmax()] - y[y.argmin()]) / (x[y.argmax()] - x[y.argmin()])
        return m



