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
        - 01_analytic_vs_mock_hmf.pdf   ---> Validation plot
        - 01_halo_shmr.pdf      ---> Stellar to Halo Mass Relation for Halos plot
"""
import numpy as np
import pandas as pd
from hmf import MassFunction
import matplotlib.pyplot as plt
import graphic_tools.mycolors as clrs
from scipy.interpolate import interp1d
import halo_tools.halo_relations as hrel
import halo_tools.halo_constants as hcnst
import halo_tools.halo_distributions as hdst
import cosmology_tools.cosmo_constants as csmlgy

# Set personal configuration for plots
plt.rcParams.update(clrs.my_plt_confg)

#=====================================================
# hmf MassFunction Parameters
#=====================================================
mf = MassFunction(
    z=0,    # Redshift
    cosmo_params={'Om0':csmlgy.O_m0, 'Ob0':csmlgy.O_b0, 'Tcmb0':2.725, 'Neff':3.05, 'H0':100*csmlgy.h},
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
            continue

# Getting data from hmf.MassFunction
analytic_hmf = mf.dndlog10m         # Halo Mass Function (HMF) [column data]
analytic_logmvir = np.log10(mf.m)   # log10(Mvir) [column data]
nvir_min = mf.ngtm[0]               # Minimum cumulative number density (CHMF) n(>Mvir_min) [float]

# Compute PDF and CPDF associated to HMF
analytic_pdf = analytic_hmf/nvir_min    # P(Mvir) dlogMvir
analytic_cpdf = mf.ngtm/nvir_min        # P(>Mvir)

# Get user desire for the number of Halos N(>Mvir) or comoving volume V
# To avoid undefined variables
ntot_halos = None
comoving_v = None
while True:
    if selection == 1:
        try:
            ntot_halos = int(input(f"Enter the number of halos N( >{hcnst.HALO_LOGMVIR_MIN} ) to sample: "))
        except ValueError:
            print('Please, enter a valid integer value.')
        else:
            if ntot_halos <= 0:
                print('Please, enter a value greater than zero.')
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
                print('Please, enter a value greater than zero.')
                continue
            else:
                # Compute the number of halos for comoving_v (must be an integer value)
                ntot_halos = int(comoving_v * nvir_min)
                break

# Create dictionary with the Mock Sample properties: comoving volume and N(>Mvir)
mock_properties = {'N(>Mvir)' : ntot_halos, 'V': comoving_v}
print(f'''
Halo Mock Catalog Properties
=====================================================
N(>{hcnst.HALO_LOGMVIR_MIN}): {ntot_halos} Halos
V: {comoving_v:13.4f} h⁻³Mpc³
=====================================================
''')

# Create interpolation nodes Data Frame
interp_nodes = pd.DataFrame({
    'Halo_logMvir': analytic_logmvir,   # Analytic logMvir values from hmf
    'P(>Mvir)': analytic_cpdf,          # Analytic P(>Mvir) derived above
})

# We sort the DataFrame for interpolation.
# To interpolate, the 'x' data (from 'x' and 'f(x)') must be in ascending order.
sorted_nodes = interp_nodes.sort_values(by = 'P(>Mvir)')

# Create cubic interpolation function
interp_function = interp1d(sorted_nodes['P(>Mvir)'],sorted_nodes['Halo_logMvir'],kind= 'cubic')

# Generate n_tot_halos uniform random values.
# To avoid extrpolation, random_u must satisfy:   random_u ∈ [P(>Mvir)_min, P(>Mvir)_max]
random_u = np.random.uniform(
    interp_nodes['P(>Mvir)'].min(),     # Minimum analytic P(>Mvir)_min
    interp_nodes['P(>Mvir)'].max(),     # Maximum analytic P(>Mvir)_max
    ntot_halos                          # Total number of halos to sample N(>Mvir)
)

# Generate Halo Mock Catalog
mock_halos = pd.DataFrame({
    'Halo_id' : np.arange(1,ntot_halos+1),      # Unique mock halo identifier
    'Halo_logMvir': interp_function(random_u)   # logMvir interpolated
})

#=========================================
# Compute synthetic HMF
#=========================================
hmf = hdst.compute_hmf(mock_halos,comoving_v)

#=========================================
# Compute Central Galaxy Stellar Masses
#=========================================
hrel.compute_log_stellar_mass(mock_halos)

# =======================================
# Save Mock Halo Catalog to CSV
# =======================================
mock_halos.to_csv('data/01_mock_halos.csv',index = False) # Columns ---> ['Halo_id','Halo_logMvir','Halo_logMste']

#=========================================
# Plots
#=========================================
# Analytic & Synthetic Mass Function
fig1, axs1 = plt.subplots(1,1,figsize=(7,7))
axs1.plot(analytic_logmvir,np.log10(analytic_hmf),color = clrs.FAV_RED, linewidth = 5, label = r'Analytic $\phi_{vir}$')
axs1.plot(hmf['Halo_logMvir'],np.log10(hmf['HMF']),color = clrs.FAV_BLUE, ls ='--', linewidth = 4, label = r'Synthetic $\phi_{vir}$')
axs1.set_title(r'Analytic \& Synthetic Mass Function')
axs1.set_xlabel(r'$\log{M_{vir}}$  $[M_\odot]$',fontsize=15)
axs1.set_ylabel(r'$\phi_{vir}$  $[{Mpc}^{-3}{dex}^{-1}]$',fontsize=15)
plt.legend()
plt.savefig('plots/01_analytic_vs_mock_hmf.pdf')
plt.show()

# Stellar to Halo Mass Relation for Halos
fig2, axs2 = plt.subplots(1,1,figsize=(7,7))
# logMvir array for mean Stellar to Halo Mass Relation
logmvir_array = np.linspace(hcnst.HALO_LOGMVIR_MIN,hcnst.HALO_LOGMVIR_MAX,1000)
axs2.scatter(mock_halos['Halo_logMvir'],mock_halos['Halo_logMste'],s = 1, rasterized = True,color = clrs.FAV_BLUE)
axs2.plot(logmvir_array,hrel.SHMR_RP17(0,logmvir_array),linewidth = 2, color = clrs.PEARL_BLACK, label = r'Mean SHMR')
axs2.set_title(r' Stellar to Halo Mass Relation for Halos')
axs2.set_xlabel(r'$\log{M_{vir}}$  $[M_\odot]$',fontsize=15)
axs2.set_ylabel(r'$\log{M_\ast}$  $[M_\odot]$',fontsize=15)
plt.legend(loc='center right')
plt.savefig('plots/01_halo_shmr.pdf')
plt.show()