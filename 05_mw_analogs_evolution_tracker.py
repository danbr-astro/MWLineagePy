"""
05_mw_analogs_evolution_tracker.py
----------------------------------
This 5.° main scripy computes the evolution of host halos with mw analogs central galaxies using
the halo concentrations model to compute it.

Outputs:
    plots/:
    mock_data/:
    data/:
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphic_tools.mycolors as clrs
import halo_tools.halo_assembly as hass
import halo_tools.halo_relations as hrel
import galaxy_tools.galaxy_assembly as gass
import cosmology_tools.cosmo_constants as csmlgy

# Personal set up for plots
plt.rcParams.update(clrs.my_plt_confg)
# Set up for Cosmological Parameters
cosmology=[0,csmlgy.O_m0,csmlgy.O_l0,csmlgy.O_b0,csmlgy.sigma_8,csmlgy.h,csmlgy.delta_c]

#=========================================
# DataFrames Loading
#=========================================
# Mock MW-like halos catalog
mock_mwlike_halos = pd.read_csv('mock_data/03_mock_mwlike_halos.csv')
# Mock MW-like subhalos catalog
mock_mwlike_subhalos = pd.read_csv('mock_data/03_mock_mwlike_subhalos.csv')
# Mock MW-analog halo's ID's
mw_analogs_haloids = pd.read_csv('mock_data/04_mwanalogs_haloids.csv')
# Dimaduro Galaxy Sample Data Interest Columns --> ['ID','z_best','Mste','Reff_5000']
dimaduro_galaxy_sample = pd.read_csv('data/Dimaduro_galaxy_sample.dat',sep=r'\s+')

# Obtaining MW-analog halos catalog
mock_mwanalog_halos = mock_mwlike_halos[mock_mwlike_halos['Halo_id'].isin(mw_analogs_haloids['Halo_id'])].copy()
# Obtaining its stellar mass standard deviation
mock_mwanalog_halos['mean_Halo_logMste'] = hrel.SHMR_RP17(0,mock_mwanalog_halos['Halo_logMvir'])
mock_mwanalog_halos['Sigma_gauss'] = mock_mwanalog_halos['Halo_logMste'] - mock_mwanalog_halos['mean_Halo_logMste']

#=========================================
# Concentration Evolution Model
#=========================================
# Concentration Model for Halos
mwanalogs_evo = hass.evolve_halos(mock_mwanalog_halos,cosmology)

# Concentration Model for Central Galaxies
gass.evolve_galaxies(mwanalogs_evo)

# Delete column ['Sigma_Gauss'], it is not necessary anymore
mwanalogs_evo.drop('Sigma_Gauss',axis=1,inplace=True)


