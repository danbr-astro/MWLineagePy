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
import matplotlib.pyplot as plt
import pandas as pd

import graphic_tools.mycolors as clrs

# Personal set up for plots
plt.rcParams.update(clrs.my_plt_confg)

#=========================================
# DataFrames Loading
#=========================================
# Mock MW-like halos catalog
mock_mwlike_halos = pd.read_csv('mock_data/03_mock_mwlike_halos.csv')
# Mock MW-like subhalos catalog
mock_mwike_subhalos = pd.read_csv('mock_data/03_mock_mwlike_subhalos.csv')
# Mock MW-analog halo's ID's
mw_analogs_haloids = pd.read_csv('mock_data/04_mwanalogs_haloids.csv')
# Dimaduro Galaxy Sample Data Interest Columns --> ['ID','z_best','Mste','Reff_5000']
dimaduro_galaxy_sample = pd.read_csv('data/Dimaduro_galaxy_sample.dat',sep=r'\s+')
