"""
03_analyze_mw_like_systems
--------------------------
References:
        -
"""
import pandas as pd
import matplotlib.pyplot as plt
import graphic_tools.mycolors as clrs

# Personal set up for plots
plt.rcParams.update(clrs.my_plt_confg)

#======================================
#DataFrames loading
#======================================
# Halos DataFrame
mock_halos = pd.read_csv('mock_data/02_mock_halos.csv')
# Subhalos DataFrame
mock_subhalos = pd.read_csv('mock_data/02_mock_subhalos.csv')
