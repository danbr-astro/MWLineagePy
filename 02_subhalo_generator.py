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
# Loading mock_halos.csv
mock_halos = pd.read_csv('mock_data/01_mock_halos.csv') # ---> ['Halo_id','Halo_logMvir','Halo_logMste']

#==========================================
# Compute the Cumulative SHMF
#==========================================
hrel.compute_nsub(mock_halos)

#==========================================
# Plots
#==========================================
# LogMvir vs lognNsub
mock_halos_aux = mock_halos[mock_halos['Halo_Nsub'] != 0]  # --> To avoid log10(0)
fig1, axs1 = plt.subplots(1,1,figsize=(8,8))
axs1.scatter(mock_halos_aux['Halo_logMvir'],np.log10(mock_halos_aux['Halo_Nsub']),s=1,color = clrs.FAV_PURPLE,rasterized = True)
axs1.set_xlabel(r'$\log{M_{vir}}$  $[M_{\odot}h^{-1}]$',fontsize=15)
axs1.set_ylabel(r'$\log{\mathcal{N}_{sub}}$',fontsize=15)
plt.show()