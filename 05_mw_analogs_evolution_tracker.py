"""
05_mw_analogs_evolution_tracker.py
----------------------------------
This 5.° main scripy computes the evolution of host halos with mw analogs central galaxies using
the halo concentrations model to compute it.

Outputs:
    plots/:
        - 05_halo&galaxy_evolution.pdf
        - 05_mean_halo_evolution.pdf
        - 05_mean_galaxy_evolution.pdf
    mock_data/:
        - 05_mwanalogs_evo.csv
        - 05_mean_mwanalogs_evo.csv
        - 05_mock_mwanalog_halos.csv
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

# Mean Concentration Model
mean_mwanalogs_evo = gass.mean_evolve_galaxies(mwanalogs_evo)

#==========================================
# Exporting DataFrames to CSV files
#==========================================
# Concentration Model for Halos
mwanalogs_evo.to_csv('mock_data/05_mwanalogs_evo.csv',index=False)
# Mean Concentration Model
mean_mwanalogs_evo.to_csv('mock_data/05_mean_mwanalogs_evo.csv',index=False)

# Mock MW-analog Halo Mock Catalog
mock_mwanalog_halos.to_csv('mock_data/05_mock_mwanalog_halos.csv',index=False)

#================================================
# Plots
#================================================
# Halo and Central Galaxy Evolution
fig1, axs1 = plt.subplots(1,1,figsize=(7,7))
# 1σ Area for Halo and Central Galaxy Evolution
axs1.fill_between(np.log10(mean_mwanalogs_evo['z+1']),mean_mwanalogs_evo['below_mean_vir'],mean_mwanalogs_evo['up_mean_vir'],
                  color=clrs.FAV_PURPLE,alpha=1,rasterized=True, label=r'Halo Evolution $1\sigma$  Area')
axs1.fill_between(np.log10(mean_mwanalogs_evo['z+1']),mean_mwanalogs_evo['below_mean_ste'],mean_mwanalogs_evo['up_mean_ste'],
                  color=clrs.FAV_ORANGE,alpha=1,rasterized=True, label=r'Central Galaxy Evolution $1\sigma$  Area')

# Individual Halo and Central Galaxy Evolution
first = True
for halo_id in pd.Series(mwanalogs_evo['Halo_id'].unique()).sample(25,random_state=7):
    auxiliar_df = mwanalogs_evo[mwanalogs_evo['Halo_id'] == halo_id]
    if first:
        axs1.plot(np.log10(auxiliar_df['z+1']),auxiliar_df['Halo_logMvir[z]'],color=clrs.MID_GREY,linewidth=1,
                  rasterized=True, alpha=0.6,label=r'Individual Halo Evolution')
        first = False
    else:
        axs1.plot(np.log10(auxiliar_df['z+1']), auxiliar_df['Halo_logMvir[z]'], color=clrs.MID_GREY, linewidth=1,
                  rasterized=True, alpha=0.6)

first = True
for halo_id in pd.Series(mwanalogs_evo['Halo_id'].unique()).sample(25,random_state=7):
    auxiliar_df = mwanalogs_evo[mwanalogs_evo['Halo_id'] == halo_id]
    if first:
        axs1.plot(np.log10(auxiliar_df['z+1']),auxiliar_df['Halo_logMste[z]'],color=clrs.MID_GREY,linewidth=1,
                  rasterized=True,alpha=0.6,label=r'Individual Central Galaxy Evolution')
        first = False
    else:
        axs1.plot(np.log10(auxiliar_df['z+1']), auxiliar_df['Halo_logMste[z]'], color=clrs.MID_GREY, linewidth=1,
                  rasterized=True, alpha=0.6)

# Mean Halo and Central Galaxy Evolution
axs1.plot(np.log10(mean_mwanalogs_evo['z+1']), mean_mwanalogs_evo['mean_logMvir[z]'],color=clrs.PEARL_BLACK,linewidth=3,
          rasterized=True,label=r'Mean Halo Evolution')
axs1.plot(np.log10(mean_mwanalogs_evo['z+1']),mean_mwanalogs_evo['mean_logMste[z]'],color=clrs.PEARL_BLACK,linewidth=3,
          rasterized=True,label=r'Mean Central Galaxy Evolution')
axs1.set_xlabel(r'$\log (z+1)$',fontsize=15)
axs1.set_ylabel(r'$\log M_\ast$,  $\log M_{vir}$',fontsize=15)
plt.legend()
plt.savefig('plots/05_halo&galaxy_evolution.pdf')
plt.show()

# Mean Halo Evolution
fig2, axs2 = plt.subplots(1,1,figsize=(7,7))
axs2.fill_between(np.log10(mean_mwanalogs_evo['z+1']),mean_mwanalogs_evo['below_mean_vir'],mean_mwanalogs_evo['up_mean_vir'],
                  color=clrs.FAV_PURPLE,alpha=1,rasterized=True, label=r'Halo Evolution $1\sigma$  Area')
axs2.plot(np.log10(mean_mwanalogs_evo['z+1']), mean_mwanalogs_evo['mean_logMvir[z]'],color=clrs.PEARL_BLACK,linewidth=3,
          rasterized=True,label=r'Mean Halo Evolution')
axs2.set_xlabel(r'$\log (z+1)$',fontsize=15)
axs2.set_ylabel(r'$\log M_{vir}$   $[M_\odot]$',fontsize=15)
plt.savefig('plots/05_mean_halo_evolution.pdf')
plt.show()

# Mean Central Galaxy Evolution
fig3, axs3 = plt.subplots(1,1,figsize=(7,7))
axs3.fill_between(np.log10(mean_mwanalogs_evo['z+1']),mean_mwanalogs_evo['below_mean_ste'],mean_mwanalogs_evo['up_mean_ste'],
                  color=clrs.FAV_ORANGE,alpha=1,rasterized=True, label=r'Central Galaxy Evolution $1\sigma$  Area')
axs3.plot(np.log10(mean_mwanalogs_evo['z+1']), mean_mwanalogs_evo['mean_logMste[z]'],color=clrs.PEARL_BLACK,linewidth=3,
          rasterized=True,label=r'Mean Central Galaxy Evolution')
axs3.set_xlabel(r'$\log (z+1)$',fontsize=15)
axs3.set_ylabel(r'$\log M_{\ast}$   $[M_\odot]$',fontsize=15)
plt.savefig('plots/05_mean_galaxy_evolution.pdf')
plt.show()