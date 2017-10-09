# -*- coding: utf-8 -*-
"""
Created on Thursday Feb 16 10:21:44 2017

@author: johninglesfield
"""
import numpy as np
from numpy import sqrt, exp, sin, cos, pi, trace
from scipy.special import kv, kvp   # Modified Bessel function
from scipy.integrate import quad
from scipy.linalg import eig, inv
from scipy.optimize import bisect
from mpmath import whitm   # Whittaker M-function
"""
This class contains functions for the Dirac equation embedding notebook. The
class is initiated with the value of c, the speed of light. 
"""

class Relativity:
    def __init__(self, c):
        self.c = c
        
    """
    The following functions are for the H-atom in a square-well.
    """
    
    def hydrogen(self, R, V, l, twoj):
        """
        Sets up the H-atom in a spherical cavity (Crampin, JPCM 16, 8875, 
        (2004)). R = radius of cavity, V = height of constant potential 
        outside cavity, l = orbital ang. mom., twoj = 2*j, 
        with j the total ang. mom.
        """
        self.R = R
        self.V = V
        self.l = l
        if twoj == 2*l + 1:
            self.kappa = -(l + 1)
            self.lbar = l + 1
        else:
            self.kappa = l
            self.lbar = l - 1
            

    def h_match(self, e):
        """
        Evaluates (large comp./small comp.) for internal (H-atom) solution
        and external (constant potential) solution. The function returns
        the difference between the ratios, evaluated at radius of cavity. The
        exact eigenvalue is the energy at which the difference = 0.
        e is the energy (without mc^2) and w is the mass-energy.
        This is based on notes by S. Crampin.
        """
        c2 = self.c**2
        w = e + c2
        lam = 1.0/self.c
        gami = sqrt(c2**2 - w**2+0j)/self.c
        a = lam*w/(gami*self.c)
        b = sqrt(self.kappa**2 - lam**2)
        s = whitm(a-0.5, b, 2.0*gami*self.R)
        c = (self.kappa - lam*self.c/gami)/(a - b)
        d = c*whitm(a+0.5, b, 2.0*gami*self.R)
        fac = (c2 + w)/(gami*self.c)
        ratin = (fac*(s + d)/(s - d)).real
        gamo = sqrt(c2**2 - (w-self.V)**2)/self.c
        u_out = kv(self.l+0.5, gamo*self.R)
        fac = sqrt((c2 - w + self.V)/(c2 + w - self.V))
        v_out = -fac*kv(self.lbar+0.5, gamo*self.R)
        ratout = u_out/v_out
        return ratin - ratout
    
    
    def h_embed_pot(self, w):
        """
        Calculates the relativistic embedding potential in spherical 
        coordinates to replace the constant potential outsde the cavity. 
        This uses eq. (20) in Crampin's 2004 article; this formula is 
        equivalent to (8.44) in my book. w is the mass-energy.
        """
        w = w + self.c**2
        self.w = w
        k = sqrt(self.c**4 - (w - self.V)**2)/self.c
        g = self.c*k/(w - self.V + self.c**2)
        K1 = kv(self.lbar+0.5, k*self.R)
        K2 = kv(self.l+0.5, k*self.R)
        ratio1 = K1/K2
        self.sigma = g*ratio1/(self.c*self.R*self.R)
        dkdw = (self.V - w)/(k*self.c*self.c)
        dgdw = self.c*((dkdw - k/(w - self.V + self.c**2))/
                (w - self.V + self.c**2))
        K1p = kvp(self.lbar+0.5, k*self.R)
        K2p = kvp(self.l+0.5, k*self.R)
        ratio2 = K1p/K2
        ratio3 = K2p/K2
        dKdw = (ratio2 - ratio1*ratio3)*self.R*dkdw 
        self.dsigma = (dgdw*ratio1 + g*dKdw)/(self.c*self.R*self.R)
        return
    
    def g(self, r, n):
        """
        Trial function for large component spinor.
        """
        g = exp(-r)*r**n
        return g
    
    def gp(self, r, n):
        """
        Derivative of trial function for large component.
        """
        gp = (n*r**(n-1) - r**n)*exp(-r)
        return gp
    
    def f(self, r, n):
        """
        Trial function for small component spinor.
        """
        f = ((n+self.kappa)*r**(n-1) - r**n)*exp(-r)
        return f
    
    def fp(self, r, n):
        """
        Derivative of trial function for small component.
        """
        fp = ((n+self.kappa)*(n-1)*r**(n-2) - n*r**(n-1) - 
              (n+self.kappa)*r**(n-1) + r**n)*exp(-r)
        return fp
    
    """
    The following are integrands for the matrix elements, taken from 
    Crampin, eq. (24).
    """
        
    def h_hamll(self, r, m, n):
        """
        large-large Hamiltonian
        """
        hamll = self.g(r, m)*(-1.0/r + self.c*self.c)*self.g(r, n)
        return hamll
    
    def h_hamss(self, r, m, n):
        """
        small-small Hamiltonian
        """
        hamss = self.f(r, m)*(-1.0/r - self.c*self.c)*self.f(r, n)
        return hamss
    
    def h_hamls(self, r, m, n):
        """
        large-small Hamiltonian
        """
        hamls = self.g(r, m)*(self.fp(r, n) - self.kappa*self.f(r, n)/r)
        return hamls
    
    def h_hamsl(self, r, m, n):
        """
        small-large Hamiltonian
        """
        hamsl = self.f(r, m)*(self.gp(r, n) + self.kappa*self.g(r, n)/r)
        return hamsl
    
    def h_ovlpll(self, r, m, n):
        """
        large-large overlap
        """
        ovlpll = self.g(r, m)*self.g(r, n)
        return ovlpll
    
    def h_ovlpss(self, r, m, n):
        """
        small-small overlap
        """
        ovlpss = self.f(r, m)*self.f(r, n)
        return ovlpss
    
    def h_matel(self, N):
        """
        Evaluates the matrix elements. The basis set size (large + small) is
        2N.
        """
        self.N = N
        self.ham = np.zeros((2*N, 2*N))
        self.ovlp = np.zeros((2*N, 2*N))
        for m in range(N):
            for n in range(N):
                hll, err = quad(self.h_hamll, 0.0, self.R, args=(m+1, n+1))
                hll = hll + (self.c**2*self.R**2*self.g(self.R, m+1)*
                      self.g(self.R, n+1)*(self.sigma - self.w*self.dsigma))
                hss, err = quad(self.h_hamss, 0.0, self.R, args=(m+1, n+1))
                hls, err = quad(self.h_hamls, 0.0, self.R, args=(m+1, n+1))
                hls = self.c*(-hls + self.g(self.R, m+1)*self.f(self.R, n+1))
                hsl, err = quad(self.h_hamsl, 0.0, self.R, args=(m+1, n+1))
                hsl = self.c*hsl
                oll, err = quad(self.h_ovlpll, 0.0, self.R, args=(m+1, n+1))
                oll = oll - (self.c**2*self.R**2*self.g(self.R, m+1)*
                      self.g(self.R, n+1)*self.dsigma)
                oss, err = quad(self.h_ovlpss, 0.0, self.R, args=(m+1, n+1))
                self.ham[m, n] = hll
                self.ham[m+N, n+N] = hss
                self.ham[m, n+N] = hls
                self.ham[m+N, n] = hsl
                self.ovlp[m, n] = oll
                self.ovlp[m+N, n+N] = oss
        return

    def h_eigen(self):
        """
        Evaluates the eigenvalues of the relativistic H atom in a spherical
        cavity.
        """
        eig_value, eig_vector = eig(self.ham, self.ovlp, left=False, 
                                    right=True)
        eigen = []
        for i in range(2*self.N):
            eigen.append(eig_value[i].real - self.c**2)
        eigen = sorted(eigen)
        return eigen
    
    """
    The following functions are for continuum states in a one-dimensional 
    square well, both Dirac and Schrödinger (section 8.1.4 of my book).
    """
    
    def square_well(self, d, D, v):
        """
        Sets up the square well (figure 8.2). d = width of well, D defines
        basis functions, and V is the well-depth.
        """
        self.d = d
        self.D = D
        self.v = v
        
    def sw_matel(self, N):
        """
        Evaluates matrix elements for relativistic square well. Basis functions
        are given in (8.54). Note that in this case large and small matrix 
        elements add, as in (8.57) and (8.58), and the matrices all have
        dimensions N x N.
        """
        self.N = N
        self.ham = np.zeros((N, N))
        self.ovlp = np.zeros((N, N))
        self.embd = np.zeros((N, N))
        k = np.zeros(N)
        gm = np.zeros(N)
        for m in range(N):
            k[m] = m*pi/self.D
            gm[m] = k[m]/(sqrt(k[m]*k[m]+self.c*self.c) + self.c)
        for m in range(0, N, 2):
            for n in range(0, N, 2):
                hll = (self.v + self.c*self.c)*self.sw_cc_int(m, n)
                hls = (self.c*gm[n]*k[n]*self.sw_cc_int(m, n) - 2.0*self.c*
                       gm[n]*cos(0.5*k[m]*self.d)*sin(0.5*k[n]*self.d))
                hsl = self.c*gm[m]*k[n]*self.sw_ss_int(m, n)
                hss = (self.v - self.c*self.c)*gm[m]*gm[n]*self.sw_ss_int(m, n)
                oll = self.sw_cc_int(m, n)
                oss = gm[m]*gm[n]*self.sw_ss_int(m, n)
                self.ham[m, n] = hll + hls + hsl + hss
                self.ovlp[m, n] = oll + oss
                self.embd[m, n] = self.c**2*cos(0.5*k[m]*self.d)*cos(
                        0.5*k[n]*self.d)
        for m in range(1, N, 2):
            for n in range(1, N, 2):
                hll = (self.v + self.c*self.c)*self.sw_ss_int(m, n)
                hls = (self.c*gm[n]*k[n]*self.sw_ss_int(m, n) + 2.0*self.c*
                       gm[n]*sin(0.5*k[m]*self.d)*cos(0.5*k[n]*self.d))
                hsl = self.c*gm[m]*k[n]*self.sw_cc_int(m, n)
                hss = (self.v - self.c*self.c)*gm[m]*gm[n]*self.sw_cc_int(m, n)
                oll = self.sw_ss_int(m, n)
                oss = gm[m]*gm[n]*self.sw_cc_int(m, n)
                self.ham[m, n] = hll + hls + hsl +hss
                self.ovlp[m, n] = oll + oss
                self.embd[m, n] = self.c**2*sin(0.5*k[m]*self.d)*sin(
                        0.5*k[n]*self.d)
        return
                
    def sw_cc_int(self, m, n):
        """
        Integrals of cos x cos over square well.
        """
        if m != n:
            cc = self.D*(sin(0.5*(m-n)*pi*self.d/self.D)/(m-n) + 
                         sin(0.5*(m+n)*pi*self.d/self.D)/(m+n))/pi
        elif m != 0:
            cc = 0.5*self.d + self.D*sin(m*pi*self.d/self.D)/(2.0*m*pi)
        else:
            cc = self.d
        return cc
    
    def sw_ss_int(self, m, n):
        """
        Integrals of sin x sin over square well.
        """
        if m != n:
            ss = self.D*(sin(0.5*(m-n)*pi*self.d/self.D)/(m-n) -
                         sin(0.5*(m+n)*pi*self.d/self.D)/(m+n))/pi
        elif m != 0:
            ss = 0.5*self.d - self.D*sin(m*pi*self.d/self.D)/(2.0*m*pi)
        else:
            ss = 0.0
        return ss
    
    def sw_dos(self, energy):
        """
        Evaluates dos for relativistic square well. The embedding potential
        is given by (8.51).
        """
        w = energy + self.c*self.c
        sigma = 1j*sqrt(w*w - self.c**4)/(self.c*(w + self.c**2))
        hs = self.ham - 2.0*sigma*self.embd - w*self.ovlp
        hs = inv(hs)
        dos = trace(np.dot(hs, self.ovlp)).imag/pi
        return dos
    
    def sw_nonrel_matel(self, N):
        """
        Matrix elements for non-relativistic square well, N basis functions.
        """
        self.N = N
        self.ham = np.zeros((N, N))
        self.ovlp = np.zeros((N, N))
        self.embd = np.zeros((N, N))
        fac = 0.5*(pi/self.D)**2
        for m in range (0, N, 2):
            for n in range (0, N, 2):
                self.ham[m, n] = m*n*fac*self.sw_ss_int(m, n)
                self.ovlp[m, n] = self.sw_cc_int(m, n)
                self.embd[m, n] = (cos(0.5*m*pi*self.d/self.D)*
                                     cos(0.5*n*pi*self.d/self.D))
        for m in range (1, N, 2):
            for n in range (1, N, 2):
                self.ham[m, n] = m*n*fac*self.sw_cc_int(m, n)
                self.ovlp[m, n] = self.sw_ss_int(m, n)
                self.embd[m, n] = (sin(0.5*m*pi*self.d/self.D)*
                                     sin(0.5*n*pi*self.d/self.D))
        self.ham = self.ham
        return
    
    def sw_nonrel_dos(self, energy, s):
        """
        dos for non-relativistic square well. s is a string: if s = 'no' then
        the non-relativistic electron mass M = 1 is used, otherwise (s = 'yes')
        the relativistic mass (8.80) is used in the well.
        """
        if s[0] == 'n':
            M = 1.0
        else:
            M = 1.0 + (energy - self.v)/(2.0*self.c*self.c)
        sigma = -0.5j*sqrt(2.0*energy)
        hs = self.ham/M + 2.0*sigma*self.embd + (self.v - energy)*self.ovlp
        hs = inv(hs)
        dos = trace(np.dot(hs, self.ovlp)).imag/pi
        return dos