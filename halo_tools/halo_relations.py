"""
halo_relations.py
=================
Halo determined relations functions for the number of subhalos and stellar mass for their central galaxy.

This module provides the necessary functions to compute the cumulative number of subhalos at a given host halo virial
mass known as the Cumulative SubHalo Mass Function (Cumulative SHMF) with its poisson scatter. It also includes the
functions needed to assign to each halo or subhalo a Central Galaxy Stellar Mass using the Stellar to Halo Mass
Relation (SHMR) with its lognormal scatter. Finally, it also computes the concentrations for halos at z = 0.

References:
    - Rodríguez-Puebla et al. (2013) ApJ, 773:172, p.2
    - Rodríguez-Puebla et al. (2016) MNRAS, 462, 894-896
    - Boylan-Kolchin et al. (2010) MNRAS,406, 896
"""
import numpy as np
from scipy.stats import poisson
import halo_tools.halo_assembly as hass
import halo_tools.halo_constants as hcnst
import cosmology_tools.cosmo_constants as csmlgy

#======================================================
# Cumulative SHMF
#======================================================
def mean_nsub(halo_mvir,subhalo_mvir):
    """
    This function computes the mean Cumulative SHMF using the virial mass of its host halo.
    Parameters
    ----------
    :param halo_mvir: Virial Mass of the Halo Host without log10 (linear form)
    :param subhalo_mvir: Virial Mass for N_sub(>subhalo_mvir|halo_mvir)
    :return: Mean Cumulative Number of Subhalos (Mean SHMF)
    """
    mvir_12 = halo_mvir / ((10 ** 12) / csmlgy.h)  # -> (Rodríguez-Puebla et al. 2016, p.896)
    c = 0.118
    mu_0 = mvir_12 ** c
    mu_1 = 0.042
    mu_cut = 0.199
    mu = subhalo_mvir / halo_mvir
    a = -0.749
    b = 1.088
    nsub_avg = mu_0 * ((mu / mu_1) ** a) * np.exp(-(mu / mu_cut) ** b)
    return nsub_avg

def compute_nsub(halos_df):
    """
    This function computes the Cumulative SHMF supposing a poisson scatter (Rodríguez-Puebla et al. 2013).
    :param halos_df: Mock Halos DataFrame with column ['Halo_logMvir']
    :return: None. In place, it adds a new column to the DataFrame ['Halo_Nsub']
    """
    hmvir_array = 10 ** halos_df['Halo_logMvir'] # It must be without log10
    # We wish to create a Subhalo Population above SUBHALO_LOGMVIR_MIN = 9 M_sun
    halos_df['Halo_mean_Nsub'] = mean_nsub(hmvir_array,10**hcnst.SUBHALO_LOGMVIR_MIN)
    halos_df['Halo_Nsub'] = poisson.rvs(mu=halos_df['Halo_mean_Nsub'] )
    halos_df['Poisson_u'] = 1-poisson.cdf(halos_df['Halo_Nsub'],halos_df['Halo_mean_Nsub']) # ---> 1 - Cumulative PDF
#======================================================
# Stellar to Halo Mass Relation
#======================================================
def SHMR_func(alpha, delta, gamma, log10eps, log10M1, log10Mvir):  # SHMR functional form; Behroozi+2010

    def g(x, a, g, d):
        return (-np.log10(10 ** (-a * x) + 1.) +
                d * (np.log10(1. + np.exp(x))) ** g / (1. + np.exp(10 ** (-x))))

    x = log10Mvir - log10M1

    g1 = g(x, alpha, gamma, delta)

    g0 = g(0, alpha, gamma, delta)

    log10Ms = log10eps + log10M1 + g1 - g0

    return log10Ms


def SHMR_RP17(z, log10Mvir):  # Best fitting model for the SHMR RP17.

    def P(x, y, z):
        return y * z - x * z / (1 + z)

    def Q(z):
        return np.exp(-4 / (1. + z) ** 2)

    al = (1.975, 0.714, 0.042)
    de = (3.390, -0.472, -0.931)
    ga = (0.498, -0.157)
    ep = (-1.758, 0.110, -0.061, -0.023)
    M0 = (11.548, -1.297, -0.026)

    alpha = al[0] + P(al[1], al[2], z) * Q(z)

    delta = de[0] + P(de[1], de[2], z) * Q(z)

    gamma = ga[0] + P(ga[1], 0, z) * Q(z)

    log10eps = ep[0] + P(ep[1], ep[2], z) * Q(z) + P(ep[3], 0, z)

    log10M1 = M0[0] + P(M0[1], M0[2], z) * Q(z)

    log10Ms = SHMR_func(alpha, delta, gamma, log10eps, log10M1, log10Mvir)

    return log10Ms

def compute_log_stellar_mass(halos_df):
    """
    This function computes the central galaxy stellar mass given a host halo virial mass with 0.15 lognormal
    scatter (Rodríguez-Publa et al. 2013).
    :param halos_df: DataFrame with column ['Halo_logMvir'] with host halo log virial mass.
    :return: None. It adds the column ['Halo_logMste'] with the stellar mass of the central galaxy for each halo.
    """
    logmvir_array = halos_df['Halo_logMvir'].values
    mean_logmste = SHMR_RP17(0, logmvir_array)
    # Generate random stellar masses with 0.15 lognormal scatter.
    halos_df['Halo_logMste'] = np.random.normal(loc= mean_logmste,scale= 0.15 )
#=============================================
# Halo Concentrations
#=============================================
def compute_concentration(halos_df):
    """
    TO DO: Explain this function
    :return: None. It adds a column to halos_df. ['Halo_logCvir']
    """
    scatter_cvir = 0.1
    delta_log_cvir = list()
    for u in halos_df['Poisson_u']:
        delta_log_cvir.append(hass.inverse_of_normal_distribution(u, scatter_cvir))
    halos_df['Halo_logCvir'] = np.log10(hass.cvir_hal(halos_df['Halo_logMvir'], 0, hass.h_BP)) + delta_log_cvir
