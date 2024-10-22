import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from matplotlib.colors import Normalize, to_rgb
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import lambertw
from scipy.special import gamma
from functools import reduce

def Phase(phi):
    return np.exp(1j * phi)

def Superposition_1D(func, x, t, coeff, qn, param=None):
    param = {} if param is None else param
    norm = np.linalg.norm(coeff)
    return np.sum([c * func(x, t, n, **param) for c,n in zip(coeff, qn)], axis=0) / norm

def Superposition_2D(func, x, y, t, coeff, qn, param=None):
    param = {} if param is None else param
    norm = np.linalg.norm(coeff)
    return np.sum([c * func(x, y, t, qn, **param) for c, qn in zip(coeff, qn)], axis=0) / norm

def Superposition_Period(energy_func, qn, param={}, epsilon=1e-5):
    def gcd(a, b):
        while b > epsilon:
            t, b = b, a % b
            a = t
        return a
    def gcd_list(ls):
        return reduce(gcd, ls)
    
    qn = np.asarray(qn)
    energies = energy_func(qn[...,0], qn[...,1], **param)
    energy_gcd = gcd_list(energies)
    return 2 * np.pi / energy_gcd

class ISW:
    @staticmethod
    def Eigenstate_Energy_1D(n: int, L: float = 1) -> float:
        return np.pi * np.pi * n * n / (2 * L * L)
    
    @staticmethod
    def Eigenstate_Energy_2D(n: int, m: int, Ls: tuple[float, float]=(1,1)) -> float:
        return ISW.Eigenstate_Energy_1D(n, Ls[0]) + ISW.Eigenstate_Energy_1D(m, Ls[1])

    @staticmethod
    def Eigenstate_1D(x: float, t: float, n: int, L: float = 1):
        return np.sqrt(2/L)*np.sin(n*np.pi*x/L)*Phase(-t*ISW.Eigenstate_Energy_1D(n, L))*np.heaviside(L/2 - np.abs(x - L/2), 0)
    
    @staticmethod
    def Eigenstate_2D(x: float, y: float, t: float, ns: tuple[int, int], Ls: tuple[float, float] = (1,1)):
        return ISW.Eigenstate_1D(x, t, ns[0], Ls[0]) * ISW.Eigenstate_1D(x, t, ns[1], Ls[1])
    
    @staticmethod
    def Superposition_Period_1D(qn: list[int], L: float = 1):
        return 4 * L*L / (np.pi * np.gcd.reduce([n*n for n in qn]))
    
    @staticmethod
    def Superposition_Period_2D(qn: list[tuple[int, int]], L_x: float = 1, L_y: float = 1):
        return 4 * L_x*L_x * L_y*L_y / (np.pi * np.gcd.reduce([n*n*L_y*L_y + m*m*L_x*L_x for n,m in qn]))

class HermiteCoefficients:
    _coeff = {0:[1]}
    
    @staticmethod
    def _differentiate_polynomial(coeff):
        return [0] if len(coeff) == 1 else [i*coeff[i] for i in range(1, len(coeff))]
    @staticmethod
    def _multiply_by_minus2x(coeff):
        return [-2*x for x in [0] + coeff]
    @staticmethod
    def _add_polynomials(coeff1, coeff2):
        if len(coeff2) > len(coeff1):
            coeff1, coeff2 = coeff2, coeff1
        for i in range(len(coeff2)):
            coeff1[i] += coeff2[i]
        return coeff1
    @staticmethod
    def __class_getitem__(n: int):
        if n not in HermiteCoefficients._coeff:
            prev = HermiteCoefficients[n-1]
            differentiated = HermiteCoefficients._differentiate_polynomial(prev)
            multiplied = HermiteCoefficients._multiply_by_minus2x(prev)
            HermiteCoefficients._coeff[n] = HermiteCoefficients._add_polynomials(differentiated, multiplied)
        return HermiteCoefficients._coeff[n]
    
    @staticmethod
    def evaluate_poly(x, coeff):
        return sum([coeff[i]*(x**i) for i in range(len(coeff))])

class HO:
    @staticmethod
    def Eigenstate_Energy_1D(n: int, omega: float = 1):
        return (n + 1/2)*omega

    @staticmethod
    def Eigenstate_Energy_2D(ns: tuple[int, int], omegas: tuple[float, float] = (1,1)):
        return HO.Eigenstate_Energy_1D(ns[0], omegas[0]) * HO.Eigenstate_Energy_1D(ns[1], omegas[1])

    @staticmethod
    def Eigenstate_1D(x, t, n, omega=1):
        coeff = 1 / np.sqrt((2**n) * gamma(n+1) * np.sqrt(np.pi / omega))
        gauss = np.exp(-1 * omega * x**2 / 2)
        phase = Phase(-omega * (n + 0.5) * t)
        herm = HermiteCoefficients.evaluate_poly(x*np.sqrt(omega), HermiteCoefficients[n])
        return coeff * gauss * phase * herm
    
    @staticmethod
    def Eigenstate_2D(x, y, t, ns, omegas=(1,1)):
        return HO.Eigenstate_1D(x, t, ns[0], omegas[0]) * HO.Eigenstate_1D(y, t, ns[1], omegas[1])
    
    @staticmethod
    def Superposition_Period_1D(qn: list[int], omega: float = 1):
        return 4 * np.pi / (omega * np.gcd.reduce([2*n + 1 for n in qn]))
    
    @staticmethod
    def Superposition_Period_2D(qn: list[tuple[int, int]], omegas: tuple[float, float] = (1,1)):
        return 4 * np.pi / np.gcd.reduce([omegas[0]*(2*n + 1) + omegas[1]*(2*m + 1) for n,m in qn])
    
    @staticmethod
    def Coherent_State_1D(x, t, alpha, omega=1, num_components=10):
        qns = np.arange(num_components)
        coeff = np.power(alpha, qns) * np.exp(-1 * np.abs(alpha)**2 / 2) / np.sqrt(gamma(qns+1))
        return Superposition_1D(HO.Eigenstate_1D, x, t, coeff, qns, {'omega':omega})

    @staticmethod
    def Coherent_State_2D(x, y, t, alphas, omegas=(1,1), num_components=10):
        alpha_x, alpha_y = alphas
        omega_x, omega_y = omegas
        return HO.Coherent_State_1D(x, t, alpha_x, omega_x, num_components) * HO.Coherent_State_1D(y, t, alpha_y, omega_y, num_components)

class FreeParticle:
    @staticmethod
    def Gaussian_1D(x, t, sigma, k):
        zeta = lambda t: t - 2j * sigma * sigma
        f1 = lambda t: np.sqrt(zeta(0) / (np.sqrt(2*np.pi) * sigma * zeta(t)))
        f2 = lambda x, t: Phase((x - k*t)**2 / (2 * zeta(t)))
        f3 = lambda x, t: Phase(k * (x - k * t / 2))
        return f1(t) * f2(x, t) * f3(x, t)
    
    @staticmethod
    def Gaussian_2D(x, y, t, sigmas, ks):
        return FreeParticle.Gaussian_1D(x, t, sigmas[0], ks[0]) * FreeParticle.Gaussian_1D(y, t, sigmas[1], ks[1])

class DD:
    @staticmethod
    def _even_kappa(beta, L):
        return beta + lambertw(beta*L*np.exp(-1*beta*L)) / L

    # Odd-state wave number
    @staticmethod
    def _odd_kappa(beta, L):
        return beta + lambertw(-1*beta*L*np.exp(-beta*L)) / L

    # Even-state energy
    @staticmethod
    def Even_Energy(beta, L):
        return -0.5 * DD._even_kappa(beta, L)**2

    # Odd-state energy
    @staticmethod
    def Odd_Energy(beta, L):
        return -0.5 * DD._odd_kappa(beta, L)**2

    # Even Energy Eigenstate
    @staticmethod
    def Even_Eigenstate(x, t, beta, L):
        x = np.array(x, dtype=complex)
        k = DD._even_kappa(beta, L)
        a_1 = np.sqrt(2*k) * np.cosh(0.5*k*L) / np.sqrt(1 + (1+(k*L))*np.exp(-1*k*L))
        a_2 = a_1 * np.exp(-0.5*k*L) / (2 * np.cosh(0.5*k*L))
        phase = Phase(DD.Even_Energy(beta, L) * t)
        region1 = lambda x: a_1 * np.exp(k * x) * phase
        region2 = lambda x: a_2 * 2 * np.cosh(k*x) * phase
        region3 = lambda x: a_1 * np.exp(-1*k*x) * phase
        return np.piecewise(x, [x < -0.5*L, (x >= -0.5*L) & (x < 0.5*L), x >= 0.5*L], [region1, region2, region3])

    # Odd Energy Eigenstate
    @staticmethod
    def Odd_Eigenstate_1D(x, t, beta, L):
        x = np.array(x, dtype=complex)
        k = DD._odd_kappa(beta, L)
        a_1 = np.sqrt(2*k) * np.sinh(0.5*k*L) / np.sqrt(1 - (1+(k*L))*np.exp(-k*L))
        a_2 = a_1 * np.exp(-0.5*k*L) / (2 * np.sinh(0.5*k*L))
        phase = Phase(DD.Odd_Energy(beta, L) * t)
        region1 = lambda x: a_1 * np.exp(k*x) * phase
        region2 = lambda x: -2 * a_2 * np.sinh(k*x) * phase
        region3 = lambda x: -1 * a_1 * np.exp(-1*k*x) * phase
        return np.piecewise(x, [np.real(x) < -0.5*L, (np.real(x) >= -0.5*L) & (np.real(x) < 0.5*L), np.real(x) >= 0.5*L], [region1, region2, region3])

    # Superposition of even/odd eigenstates
    @staticmethod
    def Superposition(x, t, even_coeff, odd_coeff, beta, L):
        norm = np.sqrt(np.abs(even_coeff)**2 + np.abs(odd_coeff)**2)
        return (even_coeff * DD.Even_Eigenstate(x, t, beta, L) + odd_coeff * DD.Odd_Eigenstate_1D(x, t, beta, L)) / norm
