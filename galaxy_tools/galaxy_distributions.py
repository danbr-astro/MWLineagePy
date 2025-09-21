"""
galaxy_distributions.py
-----------------------
Central and satellite galaxy statistics.

At first, it computes de Stellar Mass Function (SMF) for central and satellite galaxies.

"""
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from scipy.interpolate import interp1d

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
    :return None: It adds 3 columns to mean_csmf_array ---> ['Sigma','up_std','below_std']
    """
    logmste_array = mean_csmf_array['log_Mste'].values
    # We compute the individual csmf for each galaxy but with logmste_array
    def halo_csmf(halo_id):
        hollow_ngal = list()
        auxiliar_df = subhalos_df[subhalos_df['Halo_id'] == halo_id]
        # noinspection PyShadowingNames
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
        hollow_std.append(std)
    mean_csmf_array['Sigma'] = hollow_std
    # We use np.maximum to stablish a floor value
    mean_csmf_array['up_std'] = mean_csmf_array['mean_galCSMF'] + mean_csmf_array['Sigma']
    mean_csmf_array['below_std'] = np.maximum(mean_csmf_array['mean_galCSMF'] - mean_csmf_array['Sigma'],10**(-3))

def satgal_data_csmf(data_df):
    """
    This function computes de satellite galaxies csmf of observational data (McConnachie et al. 2012).
    :param data_df: DataFrame with ths observed log stellar mass
    :return galaxy_data_csmf: DataFrame with the data csmf with 2 columns. ---> ['log_Mstar','data_CSMF']
    """
    hollow_csmf = list()
    for logmstar in data_df['log_Mstar'].unique():
        n_gal = len(data_df[data_df['log_Mstar'] >= logmstar])
        hollow_csmf.append(n_gal)
    galaxy_data_csmf = pd.DataFrame({
        'log_Mstar': data_df['log_Mstar'].unique(),
        'data_CSMF':hollow_csmf
    })
    return galaxy_data_csmf

def xis_csmf(mw_csmf,ind_galcsmf,subhalos_df):
    """
    This function computes the Xi² between individual satellite galaxies csmf and the real MW csmf in order to select
    the MW analog systems. Note: logMste and logMstar are both used to name the log stellar mass., they are used to
    dofferentiate between the mock data and the observed data.
    :param mw_csmf: DataFrame with the observed MW CSMF
    :param ind_galcsmf: DataFrame with the individual mock satellite galaxies CSMF
    :param subhalos_df: DataFrame of the individual stellar mass for each of the host halos.
    :return:
    """
    # Interval validation to avoid extrapolation
    min_data_logmstar = mw_csmf['log_Mstar'].min()
    max_data_logmstar = mw_csmf['log_Mstar'].max()
    def paral_halo(halo_id):
        # We compute the Xi² for each individual CSMF with the real MW CSMF
        auxiliar_df = ind_galcsmf[ind_galcsmf['Halo_id'] == halo_id]
        # We need minimum five points
        if len(auxiliar_df['Halo_id']) > 5:
            min_individual_logmste = auxiliar_df['log_Mste'].min()
            max_individual_logmste = auxiliar_df['log_Mste'].max()
            # To constraint the MW-like galaxies of maximum log stellar mass above 8.5 (Small Magellanic Cloud)
            logmste_df = subhalos_df[subhalos_df['Halo_id'] == halo_id]
            interp_function = interp1d(auxiliar_df['log_Mste'],auxiliar_df['ind_galCSMF'],kind='cubic')
            if (min_individual_logmste<=min_data_logmstar) and (max_individual_logmste<=max_data_logmstar):
                interp_interval = mw_csmf[mw_csmf['log_Mstar']<max_individual_logmste]
            elif (min_individual_logmste<=min_data_logmstar) and (max_individual_logmste>=max_data_logmstar):
                interp_interval = mw_csmf
            elif (min_individual_logmste>=min_data_logmstar) and (max_individual_logmste>=max_data_logmstar):
                interp_interval = mw_csmf[mw_csmf['log_Mstar']>min_individual_logmste]
            elif (min_individual_logmste>=min_data_logmstar) and (max_individual_logmste<=max_data_logmstar):
                interp_interval = mw_csmf[(mw_csmf['log_Mstar']>min_individual_logmste) & (mw_csmf['log_Mstar']<max_individual_logmste)]
            else: interp_interval = None # ---> For control
            interp_data = pd.DataFrame({
                'log_Mstar':interp_interval['log_Mstar'],
                'interp_csmf':interp_function(interp_interval['log_Mstar']),
                'mw_csmf': interp_interval['data_CSMF']
            })
            # Compute Xi²
            substraction = interp_data['interp_csmf'] - interp_data['mw_csmf']
            squared = substraction ** 2.0
            xi2 = squared.sum()
            xi2_csmf = pd.DataFrame({
                'Halo_id': [halo_id],
                'xi_2': [xi2],
                'min_logMste': [logmste_df['Subhalo_logMste'].min()],
                'max_logMste': [logmste_df['Subhalo_logMste'].max()]
            })
            return xi2_csmf
        else:
            return None # If neither of the conditions are satisfied
    results = Parallel(n_jobs=8)(delayed(paral_halo)(halo_id) for halo_id in ind_galcsmf['Halo_id'].unique())
    # For the csmf that has less than 5 points, we have to delete their None values
    results = [df for df in results if df is not None]
    xis_df = pd.concat(results,axis=0,ignore_index=True).sort_values(by= 'xi_2')
    return xis_df
