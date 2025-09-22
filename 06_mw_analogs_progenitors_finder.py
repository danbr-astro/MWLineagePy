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
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import graphic_tools.mycolors as clrs
from scipy.interpolate import interp1d
import galaxy_tools.galaxy_distributions as gdst

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
# Adding an identifier to the Galaxy Sample (because ID has duplicates)
dimaduro_galaxy_sample['GS_id'] = np.arange(1,len(dimaduro_galaxy_sample['z_best'])+1)

#=========================================
# Getting Interest Data
#=========================================
galaxy_data = pd.DataFrame({
    'GS_id':dimaduro_galaxy_sample['GS_id'],
    'z+1': (dimaduro_galaxy_sample['z_best']+1),
    'log_Mste': np.log10(dimaduro_galaxy_sample['Mste']),
    'R_eff': dimaduro_galaxy_sample['Reff_5000']
})

#=========================================
# Compute Probability Density
#=========================================
# Interpolation of Galaxy Data for log Stellar Mass
interp_f = interp1d(mean_mwanalogs_evo['z+1'],mean_mwanalogs_evo['mean_logMste[z]'],kind='cubic')
galaxy_data['mean_logMste'] = interp_f(galaxy_data['z+1'])

# Interpolation of Galaxy Data for Concentration Model Sigma
interp_f2 = interp1d(mean_mwanalogs_evo['z+1'],mean_mwanalogs_evo['std_logMste[z]'],kind='cubic')
galaxy_data['sigma_cvir'] = interp_f2(galaxy_data['z+1'])

# Compute the Redshift Sigma
galaxy_data['sigma_z'] = gdst.sigma_z(galaxy_data['z+1']-1)

# Compute the Total Sigma
galaxy_data['sigma'] = ((galaxy_data['sigma_cvir']**2)+(galaxy_data['sigma_z']**2))**0.5

# Compute the PDF
galaxy_data['pdf'] = norm.pdf(galaxy_data['log_Mste'],loc=galaxy_data['mean_logMste'],scale=galaxy_data['sigma'])
# Normalization
norm_const = galaxy_data['pdf'].max()
galaxy_data['norm_pdf'] = galaxy_data['pdf']/norm_const


#==========================================
# Plots
#==========================================
# Test plot for MW progenitors Finder
fig1, axs1 = plt.subplots(1,1,figsize=(7,7))
axs1.scatter(np.log10(galaxy_data['z+1']),galaxy_data['log_Mste'],s=1,color=clrs.FAV_ORANGE,rasterized=True)
axs1.plot(np.log10(mean_mwanalogs_evo['z+1']),mean_mwanalogs_evo['mean_logMste[z]'],color=clrs.PEARL_BLACK,rasterized=True,linewidth=2)
plt.show()

# Probability Density for MW Progenitors finder
fig2, axs2 = plt.subplots(1,1,figsize=(7,7))
scatter = axs2.scatter(np.log10(galaxy_data['z+1']),galaxy_data['log_Mste'],s=1,c=galaxy_data['norm_pdf'],cmap='magma',rasterized=True)
axs2.plot(np.log10(mean_mwanalogs_evo['z+1']),mean_mwanalogs_evo['mean_logMste[z]'],color=clrs.PEARL_BLACK,rasterized=True,linewidth=2)
cbar = plt.colorbar(scatter,ax=axs2)
cbar.set_label('Normalized PDF', rotation=270, labelpad=20, fontsize=12)
cbar.ax.tick_params(labelsize=10)
axs2.set_xlabel(r'$\log (z+1)$',fontsize=15)
axs2.set_ylabel(r'$\log M_\ast$   $[M_{\odot}]$',fontsize=15)
plt.show()
