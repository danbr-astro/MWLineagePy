"""
halo_constants.py
-----------------
Halo and Subhalo virial mass limits for synthetic simulation.

This module constains the minimum and maximum viral mass for host halos, the minimum viral mass for the subhalos in
logarithmic scale.

References:
    -Rodríguez-Puebla et al. (2013) ApJ, 773:172, p.2
"""
# Halo virial mass interval
HALO_LOGMVIR_MIN = 10.5     # Maximum halo virial mass [M_sun]
HALO_LOGMVIR_MAX = 15       # Minimum halo virial mass[M_sun]

# Minimum Subhalo virial mass [M_sun]
SUBHALO_LOGMVIR_MIN = 9

# hmf MassFunction paramater
DLOG10MVIR = 0.1
