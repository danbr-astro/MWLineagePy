"""
galaxy_assembly.py
------------------
Central Galaxy evolution.

This module computes the central galaxy log stellar mass for z>0 using the concentraion model and
the stellar to halo mass relation.
"""
import halo_tools.halo_relations as hrel
def evolve_galaxies(halos_evolution_df):
    """
    This function computes the central galaxy log stellar mass for z>o using the
    halo log virial mass of its host halo and the stellar to halo mass relation.
    :param halos_evolution_df:
    :return None: It adds the column ['Halo_logMste[z]']
    """
    mean_logmstarz = hrel.SHMR_RP17(halos_evolution_df['z+1']-1,halos_evolution_df['Halo_logMvir[z]'])
    halos_evolution_df['Halo_logMste[z]'] = mean_logmstarz + halos_evolution_df['Sigma_Gauss']