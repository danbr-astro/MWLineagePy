"""
04_analyze_mw_analog_systems.py
-------------------------------
This 4.° mains script contrainst in a more robust way our mw-like systems, selecting the 5% of the mw-like mock catalog
with the lower Xi² and denominate them as mw-analogs. More over, it computes its statistics and its Subhalo CSMF.

Outputs:
    mock_data/:
        - 04_mwanalogs_haloids.csv
    plots/:
        - 04_Xi_pdf.pdf
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphic_tools.mycolors as clrs
import halo_tools.halo_distributions as hdst
import galaxy_tools.galaxy_distributions as gdst

# Personal set up for plots
plt.rcParams.update(clrs.my_plt_confg)

#=====================================
# DataFrame Loading
#=====================================
# MW-like halos & subhalos mock data
mock_mwlike_halos = pd.read_csv('mock_data/03_mock_mwlike_halos.csv')
mock_mwlike_subhalos = pd.read_csv('mock_data/03_mock_mwlike_subhalos.csv')
# McConnachie MW satellite galaxies data
mw_dat = pd.read_csv('data/McConnachie_2012.csv')
    # Satellite Galaxies CSMF DataFrames (Mean, Individual, Observed)
# Mean Satellite Galaxy CSMF
mean_galcsmf = pd.read_csv('mock_data/03_mean_galcsmf.csv')
# Individual Satellite Galaxy CSMF
ind_galcsmf = pd.read_csv('mock_data/03_ind_galcsmf.csv')
# McConnachie MW satellite galaxies CSMF
mw_galaxy_csmf = pd.read_csv('mock_data/03_mw_data_csmf.csv')
    # Subhalo CSFM DataFrames (Mean, Individual)
# Mean Subhalo CSMF
mean_subhcsmf = pd.read_csv('mock_data/03_mean_subhcsmf.csv')
# Individual Subhalo CSMF
ind_subhcsmf = pd.read_csv('mock_data/03_ind_subhcsmf.csv')
# MW-like halo virial mass distribution
mwlike_logmvir_pdf = pd.read_csv('mock_data/03_mwlogmvir_pdf.csv')

#============================================
# Xi²
#============================================
# Compute Xi² for each individual csmf for the mw csmf
xis_df = gdst.xis_csmf(mw_galaxy_csmf,ind_galcsmf,mock_mwlike_subhalos)
# Compute Xi² PDF
xis_pdf = gdst.xis_pdf(xis_df)

#================================================
# MW Analogs
#================================================
# We add another constraint. That the logarithmic maximum stellar mass for a satellite galaxy has to be above 8.5. (SMC)
smc_xis_df = xis_df[xis_df['max_logMste'] >= 8.5] # The Small Magellanic Cloud has aproximately logMste ~ 8.5
# Compute SMC Xi² PDF}
smc_xis_pdf = gdst.xis_pdf(smc_xis_df)
# 5% selection of lower Compute Xi²'s
select_5 = int(len(smc_xis_df['xi_2'])*0.05)
mwanalogs_xis = smc_xis_df.iloc[:select_5] # Selection based on SMC and Xi² constraints.
mwanalogs_haloid = mwanalogs_xis['Halo_id']
# Compute Observed MW Subhalo CSMF
mw_subhalo_csmf = hdst.mw_subhcsmf(mwanalogs_xis,mock_mwlike_subhalos)

#=================================================
# Plots
#==================================================
# Xi² PDF
fig1, axs1= plt.subplots(1,1,figsize=(7,7))
axs1.plot(xis_pdf['log_xi'],xis_pdf['xi_PDF'],color = clrs.SAKURA,label=r'Without SMC constraint')
axs1.plot(smc_xis_pdf['log_xi'],smc_xis_pdf['xi_PDF'],color=clrs.FAV_PURPLE,label=r'With SMC constraint')
axs1.set_xlabel(r'$\log{\mathcal{X}i^2}$',fontsize=15)
axs1.set_ylabel(r'$P(\log{\mathcal{X}i^2})$',fontsize=15)
plt.legend()
#plt.show()

# Cumulative Number of Subhalos/Satellite Galaxies CSMF
fig2, axs2 = plt.subplots(1,1,figsize=(7,7))
    # Subhalos
# MW Subhalo CSMF
axs2.plot(mw_subhalo_csmf['log_Mvir'],np.log10(mw_subhalo_csmf['mw_subhCSMF']),color=clrs.FAV_BLUE,ls='--',linewidth=3,rasterized=True, label=r'MW subhalo CSMF')

    # Satellite Galaxies
axs2.plot(mw_galaxy_csmf['log_Mstar'],np.log10(mw_galaxy_csmf['data_CSMF']),color=clrs.FAV_ORANGE,ls='--',linewidth=3,rasterized=True,label=r'MW galaxies CSMF')

axs2.axis((gdst.logmste_threshold,12.3,0,2))
plt.show()
