import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from matplotlib.colors import Normalize, to_rgb
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import lambertw
from scipy.special import gamma
from PIL import Image
from functools import reduce

#region Superpositions
def Superpos_1D(func, x, t, coeff, qn, param=None):
    param = {} if param is None else param
    norm = np.linalg.norm(coeff)
    return np.sum([c * func(x, t, n, **param) for c,n in zip(coeff, qn)], axis=0) / norm

def Superpos_2D(func, x, y, t, coeff, qn, param=None):
    param = {} if param is None else param
    norm = np.linalg.norm(coeff)
    return np.sum([c * func(x, y, t, qn, **param) for c, qn in zip(coeff, qn)], axis=0) / norm
#endregion

#region Infinite Square Well
# 1D Energy Eigenstate
def ISW_Eigenstate_1D_Energy(n, L=1):
    return np.pi * np.pi * n * n / (2 * L * L)
def ISW_Eigenstate_1D(x, t, n, L=1):
    x = np.array(x, dtype=complex)
    inside = lambda z: np.sqrt(2/L)*np.sin(n * np.pi * z / L)*np.exp(-1j * t * ISW_Eigenstate_1D_Energy(n))
    outside = lambda z: 0
    return np.piecewise(x, [(np.real(x) >= 0) * (np.real(x) <= L), (np.real(x) < 0) + (np.real(x) > L)], [inside, outside])
# 2D Energy Eigenstate
def ISW_Eigenstate_2D_Energy(n, m, Ls=(1,1)):
    return ISW_Eigenstate_1D_Energy(n, Ls[0]) + ISW_Eigenstate_1D_Energy(m, Ls[1])
def ISW_Eigenstate_2D(x, y, t, ns, Ls=(1,1)):
    n_x, n_y = ns
    L_x, L_y = Ls
    return ISW_Eigenstate_1D(x, t, n_x, L_x) * ISW_Eigenstate_1D(y, t, n_y, L_y)
# Superposition periods
def ISW_Superpos_1D_Period(qn, L=1):
    return 4 * L*L / (np.pi * np.gcd.reduce([n*n for n in qn]))
def ISW_Superpos_2D_Period(quantum_numbers, L_x=1, L_y=1):
    return 4 * L_x*L_x * L_y*L_y / (np.pi * np.gcd.reduce([n*n*L_y*L_y + m*m*L_x*L_x for n,m in quantum_numbers]))
#endregion

#region Harmonic Oscillator
# store Hermite polynomial coefficients for re-use
Hermite_Coeff_dict = {0:[1]}
# 1D Energy Eigenstate
def HO_Eigenstate_1D(x, t, n, omega=1):
    def diff_polynomial(coeff):
        return [0] if len(coeff) == 1 else [i*coeff[i] for i in range(1, len(coeff))]
    def mult_by_minus2x(coeff):
        return [-2*x for x in [0] + coeff]
    def add_polynomials(coeff1, coeff2):
        if len(coeff2) > len(coeff1):
            coeff1, coeff2 = coeff2, coeff1
        for i in range(len(coeff2)):
            coeff1[i] += coeff2[i]
        return coeff1
    def hermite_coeff(n):
        if n not in Hermite_Coeff_dict:
            prev = hermite_coeff(n-1)
            Hermite_Coeff_dict[n] = add_polynomials(diff_polynomial(prev), mult_by_minus2x(prev))
        return Hermite_Coeff_dict[n]
    def evaluate_poly(x, coeff):
        return sum([coeff[i]*(x**i) for i in range(len(coeff))])
    coeff = 1 / np.sqrt((2**n) * gamma(n+1) * np.sqrt(np.pi / omega))
    gauss = np.exp(-1 * omega * x**2 / 2)
    phase = np.exp(-1j * omega * (n + 0.5) * t)
    herm = evaluate_poly(x * np.sqrt(omega), hermite_coeff(n))
    return coeff * gauss * phase * herm
# 2D Energy Eigenstate
def HO_Eigenstate_2D_Energy(n, m, omegas=(1,1)):
    return omegas[0] * (n + 0.5) + omegas[1] * (m + 0.5)
def HO_Eigenstate_2D(x, y, t, ns, omegas=(1,1)):
    n_x, n_y = ns
    omega_x, omega_y = omegas
    return HO_Eigenstate_1D(x, t, n_x, omega_x) * HO_Eigenstate_1D(y, t, n_y, omega_y)
# Superposition periods
def HO_Superpos_1D_Period(qn, omega=1):
    return 4 * np.pi / (omega * np.gcd.reduce([2*n + 1 for n in qn]))
def HO_Superpos_2D_Period(quantum_numbers, omega_x=1, omega_y=1):
    return 4 * np.pi / np.gcd.reduce([omega_x*(2*n + 1) + omega_y*(2*m + 1) for n,m in quantum_numbers])
# 1D Coherent State (defualt 10 terms in sum)
def HO_Coherent_State_1D(x, t, alpha, omega=1, num_components=10):
    qns = np.arange(num_components)
    coeff = np.power(alpha, qns) * np.exp(-1 * np.abs(alpha)**2 / 2) / np.sqrt(gamma(qns+1))
    return Superpos_1D(HO_Eigenstate_1D, x, t, coeff, qns, {'omega':omega})

# 2D Coherent State
def HO_Coherent_State_2D(x, y, t, alphas, omegas=(1,1), num_components=10):
    alpha_x, alpha_y = alphas
    omega_x, omega_y = omegas
    return HO_Coherent_State_1D(x, t, alpha_x, omega_x, num_components) * HO_Coherent_State_1D(y, t, alpha_y, omega_y, num_components)
#endregion

#region Free Particle
# 1D free gaussian wave packet
def Free_Gaussian_1D(x, t, sigma, k):
    zeta = lambda t: t - 2j * sigma * sigma
    f1 = lambda t: np.sqrt(zeta(0) / (np.sqrt(2*np.pi) * sigma * zeta(t)))
    f2 = lambda x, t: np.exp(1j * (x - k*t)**2 / (2 * zeta(t)))
    f3 = lambda x, t: np.exp(1j * k * (x - k * t / 2))
    return f1(t) * f2(x, t) * f3(x, t)
# 2D free gaussian wave packet
def Free_Gaussian_2D(x, y, t, sigmas, ks):
    sigma_x, sigma_y = sigmas
    k_x, k_y = ks
    return Free_Gaussian_1D(x, t, sigma_x, k_x) * Free_Gaussian_1D(y, t, sigma_y, k_y)
#endregion

#region Double Delta Well

# Even-state wave number
def even_kappa(beta, L):
    return beta + lambertw(beta*L*np.exp(-1*beta*L)) / L

# Odd-state wave number
def odd_kappa(beta, L):
    return beta + lambertw(-1*beta*L*np.exp(-beta*L)) / L

# Even-state energy
def even_energy(beta, L):
    return -0.5 * even_kappa(beta, L)**2

# Odd-state energy
def odd_energy(beta, L):
    return -0.5 * odd_kappa(beta, L)**2

# Even Energy Eigenstate
def DD_Even_Eigenstate_1D(x, t, beta, L):
    x = np.array(x, dtype=complex)
    k = even_kappa(beta, L)
    a_1 = np.sqrt(2*k) * np.cosh(0.5*k*L) / np.sqrt(1 + (1+(k*L))*np.exp(-1*k*L))
    a_2 = a_1 * np.exp(-0.5*k*L) / (2 * np.cosh(0.5*k*L))
    phase = np.exp(-1j * even_energy(beta, L) * t)
    region1 = lambda x: a_1 * np.exp(k * x) * phase
    region2 = lambda x: a_2 * 2 * np.cosh(k*x) * phase
    region3 = lambda x: a_1 * np.exp(-1*k*x) * phase
    return np.piecewise(x, [x < -0.5*L, (x >= -0.5*L) & (x < 0.5*L), x >= 0.5*L], [region1, region2, region3])

# Odd Energy Eigenestate
def DD_Odd_Eigenstate_1D(x, t, beta, L):
    x = np.array(x, dtype=complex)
    k = odd_kappa(beta, L)
    a_1 = np.sqrt(2*k) * np.sinh(0.5*k*L) / np.sqrt(1 - (1+(k*L))*np.exp(-k*L))
    a_2 = a_1 * np.exp(-0.5*k*L) / (2 * np.sinh(0.5*k*L))
    phase = np.exp(-1j * odd_energy(beta, L) * t)
    region1 = lambda x: a_1 * np.exp(k*x) * phase
    region2 = lambda x: -2 * a_2 * np.sinh(k*x) * phase
    region3 = lambda x: -1 * a_1 * np.exp(-1*k*x) * phase
    return np.piecewise(x, [np.real(x) < -0.5*L, (np.real(x) >= -0.5*L) & (np.real(x) < 0.5*L), np.real(x) >= 0.5*L], [region1, region2, region3])

# Superposition of even/odd eigenstates
def DD_Superpos_1D(x, t, even_coeff, odd_coeff, beta, L):
    norm = np.sqrt(np.abs(even_coeff)**2 + np.abs(odd_coeff)**2)
    return (even_coeff * DD_Even_Eigenstate_1D(x, t, beta, L) + odd_coeff * DD_Odd_Eigenstate_1D(x, t, beta, L)) / norm
#endregion