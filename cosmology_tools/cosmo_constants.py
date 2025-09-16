"""
cosmo_constants.py
------------------
Cosmological parameter values for the Bolshoi–Planck and MultiDark–Planck simulations.

References:
    - Rodríguez-Puebla et al. (2016) MNRAS, 462, 894
"""
# Reduced Hubble parameter
h = 0.678           # H₀ / (100 km s⁻¹ Mpc⁻¹)

# Density parameteres at z=0
O_m0 = 0.307115     # Total matter density parameter
O_l0 = 1 - O_m0     # Dark energy density parameter
O_b0 = 0.048        # Baryon density parameter

# Primordial power spectrum parameters
n_s = 0.96          # Scalar spectral index
sigma_8 = 0.823     # Matter fluctuation amplitude

# Collapse threshold
delta_c = 1.686