"""
03_analyze_mw_like_systems
--------------------------
This 3.° main script focus our mock data into halos that host MW-like galaxies, defined by the ones
that host a central galaxy stellar mass log(Mste) that satifies log(Mste) ∈ [10.64,10.84], then it studies and computes
their properties, specifically its Stellar Mass Function SMF and its Cumulative Number of Satellite Galaxies/Subhalos
CSMF.

Outputs
        - mockdata/
            - 03_mock_mwlike_halos.csv
            - 03_mock_mwlike_subhalos.csv
            - 03_mwlogmvir_pdf.csv
            - 03_mean_galcsmf.csv
            - 03_mean_subhcsmf.csv
            - 03_ind_galcsmf.csv
            - 03_ind_subhcsmf.csv
        - data/
            - McConnachie_2012.csv
        - plots/
            - 03_halo_shmr_&_logmvir_pdf.pdf
            - 03_smf.pdf
            - 03_galaxy_&_subhalo_csmf.pdf
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphic_tools.mycolors as clrs
import halo_tools.halo_relations as hrel
import halo_tools.halo_constants as hcnst
import halo_tools.halo_distributions as hdst
import galaxy_tools.galaxy_mw_data as mwstdata
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
# Compute MW Halo Virial Mass Distribution
#=======================================
mean_logmvir, logmvir_pdf = hdst.compute_halomass_distribution(mock_mwlike_halos['Halo_logMvir'])
mwlike_std = mock_mwlike_halos['Halo_logMvir'].std()
print(f'''
The logarithmic expected value for the halo virial mass distribution is: {mean_logmvir:5.2f}
The expected value has a standard deviation of: {mwlike_std:5.2f}
''')

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

# Individual Galaxy CSMF
ind_galcsmf = gdst.ind_gal_csmf(mock_mwlike_subhalos)
# Individual Subhalo CSMF
ind_subhcsmf = hdst.ind_subh_csmf(mock_mwlike_subhalos)

# Compute Galaxy CSMF Standard Deviation
gdst.gal_csmf_std(mean_galcsmf,mock_mwlike_subhalos)
# Compute Subhalo CSMF Standard Deviation
hdst.subh_csmf_std(mean_subhcsmf,mock_mwlike_subhalos)

# MW satellite galaxies data
mwsat_data = mwstdata.mw_sat()
# MW satellite galaxies data csmf
mw_data_csmf = gdst.satgal_data_csmf(mwsat_data)

#==========================================
# Exporting DataFrames to CSV files
#==========================================
# MW-like mock halos catalog
mock_mwlike_halos.to_csv('mock_data/03_mock_mwlike_halos.csv',index=False)
# MW-like mock subhalos catalog
mock_mwlike_subhalos.to_csv('mock_data/03_mock_mwlike_subhalos.csv',index=False)
# MW-like halo virial mass distribution
logmvir_pdf.to_csv('mock_data/03_mwlogmvir_pdf.csv',index=False)
# MW-like mean satellite galaxies CSMF
mean_galcsmf.to_csv('mock_data/03_mean_galcsmf.csv',index=False)
# MW-like mean subhalo CSMF
mean_subhcsmf.to_csv('mock_data/03_mean_subhcsmf.csv',index=False)
# MW-like individual satellite galaxies CSMF
ind_galcsmf.to_csv('mock_data/03_ind_galcsmf.csv',index=False)
# MW-like individual subhalo CSMF
ind_subhcsmf.to_csv('mock_data/03_ind_subhcsmf.csv',index=False)
# McConnachie MW satellite galaxies data
mwsat_data.to_csv('data/McConnachie_2012.csv',index=False)
# McConnachie MW satellite galaxies CSMF
mw_data_csmf.to_csv('mock_data/03_mw_data_csmf.csv',index=False)

#=======================================
# Plots
#=======================================
# Stellar to Halo Mass Relation for Halos and MW-like Halo Virial Mass Distribution
fig1, axs1 = plt.subplots(1,1,figsize= (7,7))
    #Stellar to Halo Mass Relation
logmvir_array = np.linspace(hcnst.HALO_LOGMVIR_MIN,hcnst.HALO_LOGMVIR_MAX,1000)
axs1.scatter(mock_halos['Halo_logMvir'],mock_halos['Halo_logMste'],s = 1, rasterized = True,color = clrs.FAV_BLUE)
axs1.plot(logmvir_array,hrel.SHMR_RP17(0,logmvir_array),linewidth = 2, color = clrs.PEARL_BLACK, label = r'Mean SHMR',rasterized=True)
axs1.set_xlabel(r'$\log{M_{vir}}$  $[M_\odot]$',fontsize=15)
axs1.set_ylabel(r'$\log{M_\ast}$  $[M_\odot]$',fontsize=15)
    # MW-like Halo Virial Mass Distribution
# Plot with an inset
inset_axs = axs1.inset_axes((0.27,0.03,0.5,0.4))
inset_axs.plot(logmvir_pdf['log_Mvir'],logmvir_pdf['PDF'],color= clrs.PEARL_BLACK,rasterized= True,label=r'Halo $\log{M_{vir}}$ PDF')
inset_axs.set_xlim(11.76896191680672, 13.660358985495744)
inset_axs.set_xticks([])
inset_axs.set_yticks([])
for spine in inset_axs.spines.values():
    spine.set_visible(False)
plt.savefig('plots/03_halo_shmr_&_logmvir_pdf.pdf')
plt.show()

# Stellar Mass Function for Central Galaxies and Satellite Galaxies
fig2, axs2 = plt.subplots(1,1,figsize= (7,7))
axs2.plot(central_smf['log_Mste'],np.log10(central_smf['SMF']),color= clrs.SAKURA,label= r'SMF for MW Central Galaxies')
axs2.plot(satellite_smf['log_Mste'],np.log10(satellite_smf['SMF']),color= clrs.FAV_PURPLE,label= r'SMF for MW Satellite Galaxies')
axs2.set_xlabel(r'$\log{M_{star}}$  $[M_\odot]$',fontsize=15)
axs2.set_ylabel(r'$\phi_{star}(M_{star,sat}|M_{star,cen})$  $[{dex}^{-1}]$',fontsize=15)
axs2.legend()
plt.savefig('plots/03_smf.pdf')
plt.show()

# Cumulative Number of Subhalos/Satellite Galaxies CSMF
fig3, axs3 = plt.subplots(1,1,figsize=(7,7))
# Galaxies
    # 1σ Area
axs3.fill_between(mean_galcsmf['log_Mste'],np.log10(mean_galcsmf['below_std']),np.log10(mean_galcsmf['up_std']),color='#dadaeb',alpha=1,rasterized= True,label=r'Galaxy CSMF $1\sigma$  Area')
    # Individual Satellite Galaxy CMSF
first = True
for halo_id in mock_mwlike_halos['Halo_id'].sample(150,random_state=7):
    auxiliar_df = ind_galcsmf[ind_galcsmf['Halo_id'] == halo_id]
    if first:
        plt.plot(auxiliar_df['log_Mste'], np.log10(auxiliar_df['ind_galCSMF']), color='#9e9ac8', alpha=0.3, linewidth=1,
                 rasterized=True,label=r'MW-like galaxies CSMF')
        first = False
    else:
        plt.plot(auxiliar_df['log_Mste'],np.log10(auxiliar_df['ind_galCSMF']),color='#9e9ac8',alpha=0.3,linewidth=1,rasterized= True)
    # Mean Galaxy CSMF
axs3.plot(mean_galcsmf['log_Mste'],np.log10(mean_galcsmf['mean_galCSMF']),color='#756bb1',linewidth=3, rasterized= True,label=r'Mean Galaxy CSMF')
    # Galaxy Data CSMF
axs3.plot(mw_data_csmf['log_Mstar'],np.log10(mw_data_csmf['data_CSMF']),color=clrs.FAV_ORANGE,ls='--',linewidth=3, rasterized=True, label=r'Milky Way CSMF')

# Subhalos
# 1σ Area
axs3.fill_between(mean_subhcsmf['log_Mvir'],np.log10(mean_subhcsmf['below_std']),np.log10(mean_subhcsmf['up_std']),color='#cccccc',alpha=1,rasterized= True,label=r'Subhalo CSMF $1\sigma$  Area')
    # Individual Subhalo CSMF
first = True
for halo_id in mock_mwlike_halos['Halo_id'].sample(150,random_state=7):
    auxiliar_df = ind_subhcsmf[ind_subhcsmf['Halo_id'] == halo_id]
    if first:
        plt.plot(auxiliar_df['log_Mvir'], np.log10(auxiliar_df['ind_subhCSMF']), color='#666666', alpha=0.2,
                 linewidth=1, rasterized=True,label=r'MW-like subhalos CSMF')
        first = False
    else:
        plt.plot(auxiliar_df['log_Mvir'],np.log10(auxiliar_df['ind_subhCSMF']),color= '#666666',alpha=0.2, linewidth=1,rasterized=True)
    # Mean Subhalo CSMF
axs3.plot(mean_subhcsmf['log_Mvir'],np.log10(mean_subhcsmf['mean_subhaloCSMF']),color= '#404040', linewidth= 3,label=r'Mean Subhalo CSMF')
axs3.axis((logmste_threshold,12.3,0,2))
axs3.set_xlabel(r'$\log M_\ast, \; \log M_\mathrm{sub-peak}$.',fontsize=13)
axs3.set_ylabel(r'$N_\mathrm{sat}(\geq M_\ast|M_{\ast,\mathrm{MW}}), \; N_\mathrm{sub}(\geq M_\mathrm{sub-peak}|M_{\ast,\mathrm{MW}})$',fontsize=13)
plt.legend()
plt.savefig('plots/03_galaxy_&_subhalo_csmf.pdf')
plt.show()