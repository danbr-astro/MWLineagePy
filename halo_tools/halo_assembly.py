import numpy as np
from scipy.special import erfinv, erf
import math

############################Constants##########################
G = 4.299E-9  # Gravitational constant Mpc Msol**-1 (km/s)**2

H0 = 100  # Today's Hubble constant km/s/Mpc

h_BP = 0.678  # BP h value

logM13 = 13


#####################Cosmology##########################
def scale_factor(z):
    zplus_one = 1. + z

    scale = 1. / zplus_one

    return scale


def redshift(a):
    z = (1 / a) - 1
    return z


def Om_m(Om_mat, Om_lambda, z):  # Omega Matter
    return Om_mat * (1. + z) ** 3 / (Om_lambda + Om_mat * (1. + z) ** 3)


def Om_l(Om_mat, Om_lambda, z):  # Omega lambda
    return Om_lambda / (Om_lambda + Om_mat * (1. + z) ** 3)


def H(Om_mat, Om_lambda, z):
    return H0 * np.sqrt(Om_lambda + Om_mat * (1. + z) ** 3)  # Hubble constant km/s/Mpc


def rho_crit(Om_mat, Om_lambda, z):
    return 3 * H(Om_mat, Om_lambda, z) ** 2 / 8 / math.pi / G  # Critical density M_sun / Mpc**3


def rho_m(Om_mat, Om_lambda, z):
    return Om_m(Om_mat, Om_lambda, z) * rho_crit(Om_mat, Om_lambda, z)  # mean matter density M_sun / Mpc**3


def g_factor(Om_mat, Om_lambda, z):
    Amplitude = Om_mat / (1. + z)

    return Amplitude / (Om_mat ** 0.571428571 - Om_lambda + (1. + Om_mat * 0.5) * (1. + Om_lambda * 0.014285714))


def D_gfactor(Om_mat, Om_lambda, z):
    Om_m_z = Om_m(Om_mat, Om_lambda, z)

    Om_l_z = Om_l(Om_mat, Om_lambda, z)

    return g_factor(Om_m_z, Om_l_z, z) / g_factor(Om_mat, Om_lambda, 0);


def Delta_vir(Om_mat_cero, Om_lambda_cero, z):
    x = Om_mat_cero * (1 + z) ** 3 / (Om_mat_cero * (1 + z) ** 3 + Om_lambda_cero) - 1.

    return (18 * math.pi * math.pi + 82 * x - 39 * x * x) / (
                1 + x)  # Bryan & Norman 98 characteristic virial overdensity delta=334, z=0


def Rvir(log10Mvir, z, Cosmology):
    rv = log10Mvir - np.log10(
        4. * math.pi * rho_m(Cosmology[1], Cosmology[2], z) * Delta_vir(Cosmology[1], Cosmology[2], z) / 3.);

    rv = rv / 3. + 3;

    return 10 ** rv / Cosmology[5];


def dbl_pwlaw_func(x, alpha, beta):
    func_x = x ** (alpha) * (1 + x) ** (beta - alpha)

    return func_x


def sigma_Pl15(Mvir):
    y = 1E12 / Mvir
    T1 = 1.97305
    T1 = 10 ** T1 * y ** 0.539703
    T2 = 1.33525
    T2 = 10 ** T2 * y ** 0.314668
    T3 = 1.34084
    T3 = 10 ** T3 * y ** 0.488091

    return T1 / (1. + T2 + T3)


def effect_S8(sigma8):
    sigma8P15 = 0.8149
    s = sigma8 / sigma8P15

    return s


def effect_Om(logMvir, Om0):
    Om0Pl15 = 0.308

    o_m = Om0 / Om0Pl15

    x_char = 15.9428 + 0.518617 * np.log(o_m)

    y = 10 ** (logMvir - x_char)

    alpha = -0.0264382 * np.log(o_m)

    beta = -0.214687 * np.log(o_m)

    return dbl_pwlaw_func(y, alpha, beta) / dbl_pwlaw_func(1., alpha, beta)


def effect_ns(logMvir, ns):
    nsP15 = 0.9667

    n = ns / nsP15

    x_char = 14.2616

    y = 10 ** (logMvir - x_char)

    alpha = -0.119908 * np.log(n)

    return y ** alpha


def effect_Ob(logMvir, Ob):
    ObP15 = 0.0484

    o_b = Ob / ObP15

    x_char = 14.3605 - (1.25017) * np.log(o_b)

    y = 10 ** (logMvir - x_char)

    alpha = -(-0.00683278) * np.log(o_b)

    beta = -(-0.0365088) * np.log(o_b)

    return dbl_pwlaw_func(y, alpha, beta) / dbl_pwlaw_func(1., alpha, beta)


def Sigma_cosmo(Mvir, sigma8, Om0, Ob0, ns):
    sigma = sigma_Pl15(Mvir) * effect_Om(np.log10(Mvir), Om0) * effect_Ob(np.log10(Mvir), Ob0) * effect_ns(
        np.log10(Mvir), ns) * effect_S8(sigma8)

    return sigma


def f_norm(x, logMvir0, dw):
    alpha = x[1]

    beta = x[2]

    gamma = x[3]

    return pow(1. + dw, alpha) * (1. + 0.5 * dw) ** beta * np.exp(gamma * dw)


def a0_func(x, logMvir0, dw):
    ratio = x[5] - logMvir0

    return x[4] - np.log10(10 ** (x[6] * ratio) + 1.)


def g_func(x, logMvir0, dw):
    scale = 1. / (1 + dw)

    d_scale = scale - a0_func(x, logMvir0, scale)

    return 1 + np.exp(-x[7] * d_scale)


def f_func(x, logMvir0, dw):
    return (logMvir0 - logM13) * g_func(x, logMvir0, 0.) / g_func(x, logMvir0, dw)


def median_log10Mvir_progenitors(log10Mh0, z0, z, Cosmology):
    x = np.array([0., 1.52947, -3.4087, -0.404274, 0.285509, 11.9943, 0.143375, 4.07574])

    h = Cosmology[5]

    delta_c = Cosmology[6]

    dw = delta_c / D_gfactor(Cosmology[1], Cosmology[2], z) - delta_c / D_gfactor(Cosmology[1], Cosmology[2], z0)

    log10Mh0 = log10Mh0 + np.log10(h / h_BP)

    Mvirz = logM13 + np.log10(f_norm(x, log10Mh0, dw)) + f_func(x, log10Mh0, dw)

    return Mvirz - np.log10(h / h_BP)


def subhalos_correction_factor(logMpeak, z, h):
    logMpeak = logMpeak + np.log10(h)

    z2 = z * z

    Csub_z_over_Csub_0 = 0.008670 * z - 0.011330 * z2 - 0.003892 * z2 * z + 0.000370 * z2 * z2

    Normalization = 1.78 * 10 ** Csub_z_over_Csub_0

    logMcut_off = 11.904572 - 0.636422 * z - 0.020686 * z2 + 0.022034 * z * z2 - 0.001151 * z2 * z2

    ratio = logMpeak - logMcut_off

    return Normalization * np.exp(-10 ** (0.220586 * ratio))


def Total_cumulative_halo_function(logMpeak, n_vir, z, h):
    return n_vir * (1. + subhalos_correction_factor(logMpeak, z, h))


def cvir_hal(log10Mh, z, h):
    log10Mh = log10Mh + np.log10(h)

    a = -0.097 + 0.024 * z
    b = 0.537 + (1.025 - 0.537) * np.exp(-0.718 * z ** 1.08)

    C = b + a * (log10Mh - 12)

    return 10 ** C


def a_form(cvir):
    return 2.26 / cvir


def log10Mvir_progenitors_W02(log10Mh0, z0, z, h, cvir):
    return log10Mh0 - a_form(cvir) * (scale_factor(z0) / scale_factor(z) - 1.) * np.log10(np.exp(1.0))


def A_function(log10Mh0, z0, z, h, cvir):
    cvir_med = cvir_hal(log10Mh0, z0, h)

    return log10Mvir_progenitors_W02(log10Mh0, z0, z, h, cvir) - log10Mvir_progenitors_W02(log10Mh0, z0, z, h, cvir_med)


def generalized_log10Mvir_progenitors(log10Mh0, z0, z, cvir, Cosmology):
    h = Cosmology[5]

    return median_log10Mvir_progenitors(log10Mh0, z0, z, Cosmology) + A_function(log10Mh0, z0, z, h, cvir)


def inverse_of_normal_distribution(Px, std):
    if 0 < Px < 1:
        return np.sqrt(2) * std * erfinv(2 * Px - 1)
    elif Px >= 1:
        return 7 * std
    else:
        return -7 * std


def Mhalf_Correa_Schaye17(z):
    z = np.asarray(z)

    Mhalf = np.where(
        z < 2,
        -0.15 + 0.22 * z + 0.07 * z * z,
        np.where(
            z < 4,
            -0.25 + 0.53 * z - 0.07 * z * z,
            0.72 + 0.01 * z
        )
    )

    Mhalf = 1E12 * pow(10, Mhalf)

    return Mhalf


def f_acc_hot_CS17(Mvir, z):
    z = np.asarray(z)

    ztilde = np.log10(1. + z)

    az = np.where(
        z < 2,
        -1.86 * 10 ** (-1.26 * ztilde + 1.29 * ztilde * ztilde),
        np.where(
            z < 4,
            -0.46 * 10 ** (0.81 * ztilde - 0.42 * ztilde * ztilde),
            -1.07
        )
    )

    x = Mvir / Mhalf_Correa_Schaye17(z)

    func_x = 1. / (1. + x ** az)

    return np.where(func_x < 1, func_x, 1.)


def fcooling_CS17(Mvir, z, x_cool):
    return f_acc_hot_CS17(Mvir, z) * x_cool


def f_cool_bar_Mo24(Mvir):
    x = Mvir / 1E12

    return 1 / (1 + x)
