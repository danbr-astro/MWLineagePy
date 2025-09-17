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
import pandas as pd
import matplotlib.pyplot as plt
import graphic_tools.mycolors as clrs

# Personal set up for plots
plt.rcParams.update(clrs.my_plt_confg)

#==========================================
# DataFrame loading
#==========================================
# Loading mock_halos.csv
mock_halos = pd.read_csv('mock_data/01_mock_halos.csv') # ---> ['Halo_id','Halo_logMvir','Halo_logMste']
