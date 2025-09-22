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
        - 04_mwanalog&mwlike_logmvir_pdf.pdf
        - 04_galaxy_&_subhalo_csmf.pdf
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

# Compute slope for MW analog subhalo CSMF and MW like subhalo CSMF
mwanalog_slope = hdst.csmf_slope(mw_subhalo_csmf,0)
mwlike_slope = hdst.csmf_slope(mean_subhcsmf,2)
print(f'The slope for the MW analog CSMF is: {mwanalog_slope}')
print(f'The slope for the MW like CSMF is : {mwlike_slope}')

#==================================================
# MW Analogs Distributions
#==================================================
mwanalog_halos = mock_mwlike_halos[mock_mwlike_halos['Halo_id'].isin(mwanalogs_haloid)]
mean_mwanalog_logmvir, mwanalog_logmvir_pdf = hdst.compute_halomass_distribution(mwanalog_halos['Halo_logMvir'])
mw_std= mwanalog_halos['Halo_logMvir'].std()

#==========================================
# Exporting DataFrames to CSV files
#==========================================
mwanalogs_haloid.to_csv('mock_data/04_mwanalogs_haloids.csv',index=False)

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
plt.savefig('plots/04_Xi_pdf.pdf')
plt.show()

# MW-analogs & MW-like logMvir PDF
fig2, axs2 = plt.subplots(1,1,figsize=(7,7))
    # MW Analogs
y_analog = np.linspace(0,mwanalog_logmvir_pdf['PDF'].max(),50)
analog_mean = mean_mwanalog_logmvir * np.ones_like(y_analog)
below_x = mean_mwanalog_logmvir-mw_std
up_x = mean_mwanalog_logmvir+mw_std
auxiliar_df1 = mwanalog_logmvir_pdf[(below_x<=mwanalog_logmvir_pdf['log_Mvir'])&(mwanalog_logmvir_pdf['log_Mvir']<=up_x)]
y0_analog = np.zeros_like(auxiliar_df1['log_Mvir'])
axs2.fill_between(auxiliar_df1['log_Mvir'],y0_analog,auxiliar_df1['PDF'],color=clrs.FAV_PURPLE,rasterized=True,alpha=0.2,hatch='////')
axs2.plot(mwanalog_logmvir_pdf['log_Mvir'],mwanalog_logmvir_pdf['PDF'],color=clrs.FAV_PURPLE,rasterized=True,linewidth = 2,label=r'Distribution for halos hosting MW-analogs galaxies')
axs2.plot(analog_mean,y_analog,ls='--',color=clrs.FAV_PURPLE)
    # MW-like
y_like = np.linspace(0,mwlike_logmvir_pdf['PDF'].max(),50)
like_mean = 12.37 * np.ones_like(y_like)
below_xl = 12.37-0.25
up_xl = 12.37+0.25
auxiliar_df2 = mwlike_logmvir_pdf[(mwlike_logmvir_pdf['log_Mvir']>=below_xl)&(mwlike_logmvir_pdf['log_Mvir']<=up_xl)]
y0_like = np.zeros_like(auxiliar_df2['log_Mvir'])
axs2.fill_between(auxiliar_df2['log_Mvir'],y0_like,auxiliar_df2['PDF'],color=clrs.FAV_ORANGE,rasterized=True,alpha=0.2,hatch='////')
axs2.plot(mwlike_logmvir_pdf['log_Mvir'],mwlike_logmvir_pdf['PDF'],color=clrs.FAV_ORANGE,rasterized=True,linewidth=2,label=r'Distribution for halos hosting MW-like galaxies')
axs2.plot(like_mean,y_like,ls='--',color=clrs.FAV_ORANGE)
axs2.set_xlabel(r'$\log M_{vir}$   $[M_{\odot}]$',fontsize=15)
axs2.set_ylabel(r'$PDF(\log {M_{vir}})$',fontsize=15)
plt.legend()
plt.savefig('plots/04_mwanalog&mwlike_logmvir_pdf.pdf')
plt.show()

# Cumulative Number of Subhalos/Satellite Galaxies CSMF
# We filter between MW-like systems and MW-analog systems
    #Subhalos
ind_mwlike_subhcsmf = ind_subhcsmf[~ind_subhcsmf['Halo_id'].isin(mwanalogs_haloid)]
ind_mwanalog_subhcsmf = ind_subhcsmf[ind_subhcsmf['Halo_id'].isin(mwanalogs_haloid)]

    #Satellite Galaxies
ind_mwlike_galcsmf = ind_galcsmf[~ind_galcsmf['Halo_id'].isin(mwanalogs_haloid)]
ind_mwanalog_galcsmf = ind_galcsmf[ind_galcsmf['Halo_id'].isin(mwanalogs_haloid)]

fig3, axs3 = plt.subplots(1,1,figsize=(7,7))
    # Subhalos
# 1σ Area
axs3.fill_between(mean_subhcsmf['log_Mvir'],np.log10(mean_subhcsmf['below_std']),np.log10(mean_subhcsmf['up_std']),color=clrs.LIGHT_GREY,alpha=1,rasterized= True,label=r'Subhalo CSMF $1\sigma$  Area')
# Individual MW-like subhalo CSMF
first = True
for halo_id in pd.Series(ind_mwlike_subhcsmf['Halo_id'].unique()).sample(80,random_state=7):
    auxiliar_df = ind_mwlike_subhcsmf[ind_mwlike_subhcsmf['Halo_id'] == halo_id]
    if first:
        plt.plot(auxiliar_df['log_Mvir'], np.log10(auxiliar_df['ind_subhCSMF']), color=clrs.MID_GREY, alpha=0.3,
                 linewidth=1, rasterized=True,label=r'MW-like subhalos CSMF')
        first = False
    else:
        plt.plot(auxiliar_df['log_Mvir'],np.log10(auxiliar_df['ind_subhCSMF']),color= clrs.MID_GREY,alpha=0.3, linewidth=1,rasterized=True)
# Individual MW-analog subhalo CSMF
first = True
for halo_id in pd.Series(ind_mwanalog_subhcsmf['Halo_id'].unique()).sample(80,random_state=7):
    auxiliar_df = ind_mwanalog_subhcsmf[ind_mwanalog_subhcsmf['Halo_id']==halo_id]
    if first:
        plt.plot(auxiliar_df['log_Mvir'],np.log10(auxiliar_df['ind_subhCSMF']),color=clrs.FAV_ORANGE,alpha=0.3,
                 linewidth=1,rasterized=True,label=r'MW-analog subhalos CSMF')
        first = False
    else:
        plt.plot(auxiliar_df['log_Mvir'],np.log10(auxiliar_df['ind_subhCSMF']),color=clrs.FAV_ORANGE,alpha=0.3,
                 linewidth=1,rasterized=True)
# Mean Subhalo CSMF
axs3.plot(mean_subhcsmf['log_Mvir'],np.log10(mean_subhcsmf['mean_subhaloCSMF']),color= clrs.DARK_GREY, linewidth= 3,label=r'Mean Subhalo CSMF')
# MW Subhalo CSMF
axs3.plot(mw_subhalo_csmf['log_Mvir'],np.log10(mw_subhalo_csmf['mw_subhCSMF']),color=clrs.FAV_BLUE,ls='--',linewidth=3,rasterized=True, label=r'MW subhalo CSMF')

    # Satellite Galaxies
# 1σ Area
axs3.fill_between(mean_galcsmf['log_Mste'],np.log10(mean_galcsmf['below_std']),np.log10(mean_galcsmf['up_std']),color=clrs.LIGHT_LAVENDER,alpha=1,rasterized= True,label=r'Galaxy CSMF $1\sigma$  Area')
# Individual MW-like satellite galaxies CSMF
first = True
for halo_id in pd.Series(ind_mwlike_galcsmf['Halo_id'].unique()).sample(80,random_state=7):
    auxiliar_df = ind_mwlike_galcsmf[ind_mwlike_galcsmf['Halo_id'] == halo_id]
    if first:
        plt.plot(auxiliar_df['log_Mste'], np.log10(auxiliar_df['ind_galCSMF']), color=clrs.MID_LAVENDER, alpha=0.3, linewidth=1,
                 rasterized=True,label=r'MW-like galaxies CSMF')
        first = False
    else:
        plt.plot(auxiliar_df['log_Mste'],np.log10(auxiliar_df['ind_galCSMF']),color=clrs.MID_LAVENDER,alpha=0.3,linewidth=1,rasterized= True)
# Individual MW-analog satellite galaxies CSMF
first = True
for halo_id in pd.Series(ind_mwanalog_galcsmf['Halo_id'].unique()).sample(80,random_state=7):
    auxiliar_df = ind_mwanalog_galcsmf[ind_mwanalog_galcsmf['Halo_id'] == halo_id]
    if first:
        plt.plot(auxiliar_df['log_Mste'],np.log10(auxiliar_df['ind_galCSMF']),color=clrs.FAV_BLUE,alpha=0.3,linewidth=1,
                 rasterized=True,label=r'MW-analog galaxies CSMF')
        first = False
    else:
        plt.plot(auxiliar_df['log_Mste'], np.log10(auxiliar_df['ind_galCSMF']), color=clrs.FAV_BLUE, alpha=0.3,
                 linewidth=1,rasterized=True)
# Mean Satellite Galaxies CSMF
axs3.plot(mean_galcsmf['log_Mste'],np.log10(mean_galcsmf['mean_galCSMF']),color=clrs.DEEP_LAVENDER,linewidth=3, rasterized= True,label=r'Mean Galaxy CSMF')
# MW Satellite Galaxies CSMF
axs3.plot(mw_galaxy_csmf['log_Mstar'],np.log10(mw_galaxy_csmf['data_CSMF']),color=clrs.FAV_ORANGE,ls='--',linewidth=3,rasterized=True,label=r'MW galaxies CSMF')
axs3.set_xlabel(r'$\log M_\ast, \; \log M_\mathrm{sub-peak}$.',fontsize=13)
axs3.set_ylabel(r'$\log{N_\mathrm{sat}(\geq M_\ast|M_{\ast,\mathrm{MW}})}, \; \log{N_\mathrm{sub}(\geq M_\mathrm{sub-peak}|M_{\ast,\mathrm{MW}})}$',fontsize=13)
axs3.axis((gdst.logmste_threshold,12.3,0,2.25))
plt.legend(loc='upper right')
plt.savefig('plots/04_galaxy_&_subhalo_csmf.pdf')
plt.show()