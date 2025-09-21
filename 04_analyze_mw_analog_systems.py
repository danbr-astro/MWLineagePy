"""
04_analyze_mw_analog_systems.py
-------------------------------
This 4.° mains script contrainst in a more robust way our mw-like systems, selecting the 5% of the mw-like mock catalog
with the lower Xi² and denominate them as mw-analogs. More over, it computes its statistics and its Subhalo CSMF.

Outputs:
    mock_data/:
    plots/:
"""
import pandas as pd
import matplotlib.pyplot as plt
import graphic_tools.mycolors as clrs
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
mw_csmf = pd.read_csv('mock_data/03_mw_data_csmf.csv')
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
xis_df = gdst.xis_csmf(mw_csmf,ind_galcsmf,mock_mwlike_subhalos)