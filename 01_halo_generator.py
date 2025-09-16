"""
01_halo_generator.py
--------------------
First main script for generating a dark matter halo mock catalog.

Uses inverse transform sampling to compute halo virial masses from an analytic halo mass function from hmf library.
Outputs halo mock catalog and plots.

Inputs (user):
    - Cumulative number of halos N(>Mvir)
    OR
    - Comoving volume V [h⁻³Mpc³]
Outputs:
    mock_data/
        - 01_mock_halos.csv     ---> Columns: Halo_id, Halo_logMvir, Halo_logMste
    plots/
        - 01_analytic_vs_mock_hmf.pdf
        - 01_halo_shmr.pdf      ---> Stellar to Halo Mass Relation for Halos plot
"""
from hmf import MassFunction
import matplotlib.pyplot as plt
import graphic_tools.mycolors as clrs
import cosmology_tools.cosmo_constants as csmlgy

# Set personal configuration for plots
plt.rcParams.update(clrs.my_plt_confg)

#=====================================================
# hmf MassFunction Parameters
#=====================================================
mf = MassFunction(
    z=0,
    cosmo_params={'Om0':csmlgy.O_m0, 'Ob0':csmlgy.O_b0, 'Tcmb0':2.725, 'Neff':3.05, 'H0':csmlgy.h},
    n=csmlgy.n_s,
    sigma_8=csmlgy.sigma_8,

)