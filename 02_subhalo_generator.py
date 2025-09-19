"""
02_subhalo_generator.py
-----------------------
Second main script for generating a dark matter subhalo mock catalog.

To generate the dark matter subhalo mock catalog, this code uses at first a defined relationship between the virial
mass of a host halos and its number of subhalos named the Cumulative SHMF supposing a poisson scatter and using this
Cumulative SHMF, it computes the halo Concentration. Then, using the inverse tranform sampling, computes the virial mass
for each subhalo, finally it also assigns to each subhalo a central galaxy stellar mass using again the SHMR.

Outputs:
    - mock_data/
        - 02_mock_halos.csv ---> Columns: Halo_id, Halo_logMvir, Halo_logMste, Halo_Nsub, Poisson_u, Halo_logCvir
        - 02_mock_subhalos.csv ---> Columns: Halo_id, Halo_logMvir, Subhalo_logMvir, Subhalo_logMste
    - plots/
        - 02_mvir_vs_nsub.pdf ---> log10(Mvir) vs log10(Cumulative SHMF)
        - 02_subhalo_shmr.pdf ---> Stellar to Halo Mass Relation for Subhalos
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphic_tools.mycolors as clrs
import halo_tools.halo_relations as hrel

# Personal set up for plots
plt.rcParams.update(clrs.my_plt_confg)

#==========================================
# DataFrame loading
#==========================================
mock_halos = pd.read_csv('mock_data/01_mock_halos.csv') # ---> ['Halo_id','Halo_logMvir','Halo_logMste']

#==========================================
# Compute the Cumulative SHMF
#==========================================
hrel.compute_nsub(mock_halos)

#==========================================
# Compute Concentrations
#==========================================
hrel.compute_concentration(mock_halos)

#==========================================
# Creating Subhalo Mock Catalog
#==========================================
# Compute Subhalo Virial Masses
mock_subhalos = hrel.compute_subhalo_mvir(mock_halos)
# Assignation of Stellar Mass
hrel.compute_log_stellar_mass(mock_subhalos,1)

#==========================================
# Exporting DataFrames to CSV files
#==========================================
# Exporting Halo Mock Catalog
mock_halos = mock_halos.drop(columns= ['Halo_mean_Nsub'])
mock_halos.to_csv('mock_data/02_mock_halos.csv',index=False)
# Exporting Subhalo Mock Catalog
mock_subhalos.to_csv('mock_data/02_mock_subhalos.csv',index=False)
print(mock_halos)
print(mock_subhalos)

#==========================================
# Plots
#==========================================
# LogMvir vs lognNsub
mock_halos_aux = mock_halos[mock_halos['Halo_Nsub'] != 0]  # --> To avoid log10(0)
fig1, axs1 = plt.subplots(1,1,figsize=(8,8))
sc = axs1.scatter(mock_halos_aux['Halo_logMvir'],np.log10(mock_halos_aux['Halo_Nsub']),s=1,c=mock_halos_aux['Halo_logCvir'],cmap='viridis',rasterized = True)
axs1.set_xlabel(r'$\log{M_{vir}}$  $[M_{\odot}h^{-1}]$',fontsize=15)
axs1.set_ylabel(r'$\log{\mathcal{N}_{sub}}$',fontsize=15)
cbar = plt.colorbar(sc)
cbar.set_label("Concentraciones")
plt.savefig('plots/02_mvir_vs_nsub.pdf')
plt.show()

# Stellar to Halo Mass Relation for Subhalos
# Array to plot the Mean SHMR
subh_logmvir_array = np.linspace(9,15,500)
fig2 ,axs2 = plt.subplots(1,1,figsize= (7,7))
axs2.scatter(mock_subhalos['Subhalo_logMvir'],mock_subhalos['Subhalo_logMste'],color = clrs.FAV_PURPLE,s=1,rasterized=True)
axs2.plot(subh_logmvir_array,hrel.SHMR_RP17(0,subh_logmvir_array),linewidth=2,color=clrs.PEARL_BLACK,label=r'Mean SHMR',rasterized=True)
axs2.set_title(r' Stellar to Halo Mass Relation for Subhalos')
axs2.set_xlabel(r'$\log{M_{vir}}$  $[M_\odot]$',fontsize=15)
axs2.set_ylabel(r'$\log{M_\ast}$  $[M_\odot]$',fontsize=15)
plt.legend(loc='center right')
plt.savefig('plots/02_subhalo_shmr.pdf')
plt.show()