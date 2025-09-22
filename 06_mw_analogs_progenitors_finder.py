"""
06_mw_analogs_progenitors_finder.py
-----------------------------------
This 6.° main script assigns to each gaaxy from Dimaduro & Golden sample, a probability density
of being a mw analog progenitor at z>0. Then using that statistic weight, we compute properties
of the mw_progenitors for some z>0

Outputs:
    data/:
    plots/:
"""
import pandas as pd
import matplotlib.pyplot as plt
import graphic_tools.mycolors as clrs

# Personal set up for plots
plt.rcParams.update(clrs.my_plt_confg)

#======================================
# DataFrames loading
#======================================
# Halo and Galaxy Evolution (Concentration Model).
mwanalogs_evo = pd.read_csv('mock_data/05_mwanalogs_evo.csv')
# Mean Halo and Galaxy Evolution (Concentration Model).
mean_mwanalogs_evo = pd.read_csv('mock_data/05_mean_mwanalogs_evo.csv')
# Dimaduro Galaxy Sample Data Interest Columns --> ['ID','z_best','Mste','Reff_5000']
dimaduro_galaxy_sample = pd.read_csv('data/Dimaduro_galaxy_sample.dat',sep=r'\s+')


