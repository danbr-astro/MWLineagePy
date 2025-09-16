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
import numpy as np
from hmf import MassFunction
import matplotlib.pyplot as plt
import graphic_tools.mycolors as clrs
import halo_tools.halo_constants as hcnst
import cosmology_tools.cosmo_constants as csmlgy

# Set personal configuration for plots
plt.rcParams.update(clrs.my_plt_confg)

#=====================================================
# hmf MassFunction Parameters
#=====================================================
mf = MassFunction(
    z=0,    # Redshift
    cosmo_params={'Om0':csmlgy.O_m0, 'Ob0':csmlgy.O_b0, 'Tcmb0':2.725, 'Neff':3.05, 'H0':csmlgy.h},
    n=csmlgy.n_s,
    sigma_8=csmlgy.sigma_8,
    Mmin=hcnst.HALO_LOGMVIR_MIN, # Minimum halo logMvir
    Mmax=hcnst.HALO_LOGMVIR_MAX, # Maximum halo logmvir
    dlog10m=hcnst.DLOG10MVIR,
    transfer_model='EH',
    mdef_model='SOVirial',
    hmf_model='Behroozi'
)

#===================================
# Creating Halo Mock Catalog
#===================================
print(f'''
Halo Mock Catalog - Creation Mode
=================================================
1. By the Number of Halos N( >{hcnst.HALO_LOGMVIR_MIN} )
2. By the Comoving Volumen V
=================================================''')

# Get user selection with validation and error handling
while True:
    try:
        selection = int(input('Enter your selection (1 or 2): '))
    except ValueError:
        print('Please, enter a integer value (1 or 2).')
    else:
        if (selection == 1) or (selection == 2):
            break
        else:
            print('Please, enter a valid integer (1 or 2).')

# Getting data from hmf.MassFunction
analytic_hmf = mf.dndlog10m         # Halo Mass Function (HMF) [column data]
analytic_logmvir = np.log10(mf.m)   # log10(Mvir) [column data]
nvir_min = mf.ngtm[0]               # Minimum cumulative number density (CHMF) n(>Mvir_min) [float]

# Compute PDF and CPDF associated to HMF
analytic_pdf = analytic_hmf/nvir_min    # P(Mvir) dlogMvir
analytic_cpdf = mf.ngtm/nvir_min        # P(>Mvir)

# Get user desire for the number of Halos N(>Mvir) or comoving volume V
while True:
    if selection == 1:
        try:
            ntot_halos = int(input(f"Enter the number of halos N( >{hcnst.HALO_LOGMVIR_MIN} ) to sample: "))
        except ValueError:
            print('Please, enter a valid integer value.')
        else:
            if ntot_halos <= 0:
                print('Please, enter a valid value.')
                continue
            else:
                # Compute the comoving volume for ntot_halos
                comoving_v = ntot_halos/nvir_min
                break
    elif selection == 2:
        try:
            comoving_v = float(input(f'Enter the Comoving Volume V to sample: '))
        except ValueError:
            print('Please, enter a valid numeric value.')
        else:
            if comoving_v <= 0:
                print('Please, enter a valid value.')
                continue
            else:
                # Compute the number of halos for cmving_v (must be an integer value)
                ntot_halos = int(comoving_v*nvir_min)
                break
