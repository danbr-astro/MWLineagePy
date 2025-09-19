"""
03_analyze_mw_like_systems
--------------------------
References:
        -
"""
import pandas as pd
import matplotlib.pyplot as plt
import graphic_tools.mycolors as clrs
import halo_tools.halo_relations as hrel
import halo_tools.halo_constants as hcnst

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