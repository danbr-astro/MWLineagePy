"""
galaxy_assembly.py
------------------
Central Galaxy evolution.

This module computes the central galaxy log stellar mass for z>0 using the concentraion model and
the stellar to halo mass relation.
"""
import pandas as pd

import halo_tools.halo_relations as hrel
def evolve_galaxies(halos_evolution_df):
    """
    This function computes the central galaxy log stellar mass for z>o using the
    halo log virial mass of its host halo and the stellar to halo mass relation.
    :param halos_evolution_df:
    :return None: It adds the column ['Halo_logMste[z]']
    """
    mean_logmstarz = hrel.SHMR_RP17(halos_evolution_df['z+1']-1,halos_evolution_df['Halo_logMvir[z]'])
    halos_evolution_df['Halo_logMste[z]'] = mean_logmstarz + halos_evolution_df['Sigma_Gauss']

def mean_evolve_galaxies(halos_evolution_df):
    """
    This function cmputes the average evolution of galaxies
    :param halos_evolution_df: DataFrame with the log stellar mass for z>0
    :return mean_evolution_df: DataFrame with the average log stellar mass for z>0
    """
    zplus_array = halos_evolution_df['z+1'].unique()
    hollow_list = list()
    for zplus in zplus_array:
        auxiliar_df = halos_evolution_df[halos_evolution_df['z+1'] == zplus]
        mean_logmvir = auxiliar_df['Halo_logMvir[z]'].mean()
        mean_logmstar = auxiliar_df['Halo_logMste[z]'].mean()
        std_logmvir = auxiliar_df['Halo_logMvir[z]'].std(ddof=0)
        std_logmstar = auxiliar_df['Halo_logMste[z]'].std(ddof=0)
        hollow_list.append({
            'z+1':zplus,
            'mean_logMvir[z]':mean_logmvir,
            'mean_logMste[z]': mean_logmstar,
            'std_logMvir[z]': std_logmvir,
            'std_logMste[z]': std_logmstar
        })
    mean_evolution_df = pd.DataFrame(hollow_list)
    mean_evolution_df['up_mean_vir'] = mean_evolution_df['mean_logMvir[z]'] + mean_evolution_df['std_logMvir[z]']
    mean_evolution_df['below_mean_vir'] = mean_evolution_df['mean_logMvir[z]'] - mean_evolution_df['std_logMvir[z]']
    mean_evolution_df['up_mean_ste'] = mean_evolution_df['mean_logMste[z]'] + mean_evolution_df['std_logMste[z]']
    mean_evolution_df['below_mean_ste'] = mean_evolution_df['mean_logMste[z]'] - mean_evolution_df['std_logMste[z]']
    return mean_evolution_df