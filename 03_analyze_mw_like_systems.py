"""
03_analyze_mw_like_systems
--------------------------
Outputs
        - mockdata/
            - 03_mock_mwlike_halos.csv
            - 03_mock_mwlike_subhalos.csv
            - 03_logmvir_pdf.csv

        - plots/
            - 03_halo_shmr_&_logmvir_pdf.pdf
            - 03_smf.pdf
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphic_tools.mycolors as clrs
import halo_tools.halo_relations as hrel
import halo_tools.halo_constants as hcnst
import halo_tools.halo_distributions as hdst
import galaxy_tools.galaxy_distributions as gdst

# Personal set up for plots
plt.rcParams.update(clrs.my_plt_confg)

#======================================
#DataFrames loading
#======================================
# Halos DataFrame
mock_halos = pd.read_csv('mock_data/02_mock_halos.csv')
# Subhalos DataFrame
mock_subhalos = pd.read_csv('mock_data/02_mock_subhalos.csv')

#=======================================
# MW-like systems filtering
#=======================================
# MW-like central galaxy filter by logMste ∈ [10.64,10.84] (Rodríguez-Puebla et al. 2013)
mock_mwlike_halos = mock_halos[(mock_halos['Halo_logMste'] >= 10.64) & (mock_halos['Halo_logMste'] <= 10.84)] # ~ 26k halos
mock_mwlike_subhalos = mock_subhalos[mock_subhalos['Halo_id'].isin(mock_mwlike_halos['Halo_id'])] # ~ 1.02 M
# Data completeness filter for subhalos
min_meanlogmste = hrel.SHMR_RP17(0,hcnst.SUBHALO_LOGMVIR_MIN)
logmste_threshold = min_meanlogmste + (3*0.15) # +3σ (0.15 lognormal)
mock_mwlike_subhalos = mock_mwlike_subhalos[mock_mwlike_subhalos['Subhalo_logMste'] >= logmste_threshold] # ~695 k

#=======================================
# Compute Halo Virial Mass Distribution
#=======================================
mean_logmvir, logmvir_pdf = hdst.compute_halomass_distribution(mock_mwlike_halos['Halo_logMvir'])
print(f'The logarithmic expected value for the halo virial mass distribution is: {mean_logmvir:5.2f}')

#================================================================================
# Stellar Mass Function (SMF) for Central Galaxies and Satellite Galaxies
#================================================================================
# SMF for central galaxies
central_smf = gdst.compute_smf(mock_mwlike_halos['Halo_logMste'])
#SMF for satellite galaxies
satellite_smf = gdst.compute_smf(mock_mwlike_subhalos['Subhalo_logMste'])

#===================================================================================================
# Cumulative Number of Subhalos/Satellite Galaxies or Cumulative Satellite Mass Function (CSMF)
#===================================================================================================
# Mean Galaxy CSMF
mean_galcsmf = gdst.mean_gal_csmf(mock_mwlike_subhalos,logmste_threshold)
# Mean Subhalo CSMF
mean_subhcsmf = hdst.mean_subh_csmf(mock_mwlike_subhalos)

#=======================================
# Plots
#=======================================
# Halo Virial Mass Distribution
fig1, axs1 = plt.subplots(1,1,figsize= (7,7))
axs1.plot(logmvir_pdf['log_Mvir'],logmvir_pdf['PDF'],color= clrs.PEARL_BLACK,rasterized= True)
#plt.show()

# Stellar Mass Function for Central Galaxies and Satellite Galaxies
fig2, axs2 = plt.subplots(1,1,figsize= (7,7))
axs2.plot(central_smf['log_Mste'],np.log10(central_smf['SMF']),color= clrs.SAKURA,label= r'SMF for MW Central Galaxies')
axs2.plot(satellite_smf['log_Mste'],np.log10(satellite_smf['SMF']),color= clrs.FAV_PURPLE,label= r'SMF for MW Satellite Galaxies')
axs2.set_xlabel(r'$\log{M_{star}}$  $[M_\odot]$',fontsize=15)
axs2.set_ylabel(r'$\phi_{star}(M_{star,sat}|M_{star,cen})$  $[{dex}^{-1}]$',fontsize=15)
axs2.legend()
#plt.show()

# Cumulative Mass Function
fig3, axs3 = plt.subplots(1,1,figsize=(7,7))
# Galaxies
    # Mean Galaxy CSMF
axs3.plot(mean_galcsmf['log_Mste'],np.log10(mean_galcsmf['mean_galCSMF']),color='#404040',linewidth=3)

# Subhalos
    # Mean Subhalo CSMF
axs3.plot(mean_subhcsmf['log_Mvir'],np.log10(mean_subhcsmf['mean_subhaloCSMF']),color= '#756bb1', linewidth= 3)
axs3.axis((logmste_threshold,12.3,0,3))
#plt.show()