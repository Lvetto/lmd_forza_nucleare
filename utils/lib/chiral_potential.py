import numpy as np
import scipy.special as special_function
import utils.lib.constants as const
import sys

# Copyright 2023 Andreas Ekström

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#####

# A chiral two-nucleon potential up to NLO in Weinberg counting.
# no isospin breaking, electromagnetic effects, ... very vanilla.
#
# Andreas Ekström, 20230613
# andreas.ekstrom@chalmers.se
#


class two_nucleon_potential:
    """
    This class implements a chiral two-nucleon potential up to NLO in Weinberg counting.
    This module defines a class `two_nucleon_potential` that implements a chiral two-nucleon potential up to NLO in Weinberg counting.

    The potential includes one-pion exchange, two-pion exchange, and contact interactions. 

    The class provides methods for calculating the potential in partial-wave decomposition and can be used in conjunction with the `nn_studio` class for solving the nucleon-nucleon scattering problem.
    
    """

    def __init__(self,chiral_order, Lambda):
        """
        This is the constructor method for the `two_nucleon_potential` class. It initializes the potential based on the specified chiral order and cutoff parameter Lambda. The method sets up the necessary parameters, such as the masses of the nucleons and pion, the axial coupling constant, and the pion decay constant. It also precomputes the Legendre polynomials needed for the partial-wave decomposition of the potential.

        Args:
            chiral_order (int): the chiral order of the potential, which determines which terms are included in the potential (e.g., LO, NLO, etc.). This parameter is used to specify the level of approximation in the chiral effective field theory expansion.
            Lambda (float): the cutoff parameter used in the regularization of the potential. This parameter controls the high-momentum behavior of the potential and is typically chosen to be around 500 MeV in chiral effective field theory calculations. The cutoff is implemented through a regulator function that suppresses contributions from momenta above Lambda.
            
        Returns:
            None
        """

        self.chiral_order = chiral_order
        
        self.mp  = const.Mn
        self.mn  = const.Mp
        self.mpi = const.Mpi
        self.gA  = const.gA
        self.fpi = const.fpi
        self.Lambda = Lambda

        #quadrature points for cos(theta) integration
        self.ntheta = 32
        #max j (affects precomputation of meshes for angular integrals)
        self.jmax   = 10
        # z = cos(theta)
        self.z, self.w  = np.polynomial.legendre.leggauss(self.ntheta)
        self.zJ = [self.z**J for J in range(0,self.jmax+1)]
        self.P = []
        self.q2 = 0
        
        self.m    = (self.mp*self.mn)/(self.mn+self.mp)

        self.setup_legendre_polynomials()
        
        

    @staticmethod
    def legP(m,n,x):
        """
        this method precomputes the Legendre polynomials P_j(z) for j=0,...,jmax at the quadrature points z.
        The Legendre polynomials are needed for the partial-wave decomposition of the potential.
        The method uses the scipy.special.lpmv function to compute the associated Legendre polynomials, which are then stored in the list self.P for later use in the potential calculation.
        
        Args:
            m (int): order of the associated Legendre polynomial (not used here, set to 0)
            n (int): degree of the associated Legendre polynomial (corresponds to j in P_j(z))
            x (array-like): input array of cos(theta) values where the Legendre polynomials are evaluated
            
        Returns:
            array-like: values of the Legendre polynomial P_n(x) evaluated at the input x
        """
        # [0]: get function value (not derivative)
        # [-1,-1]: get 'm', 'n' values
        #return special_function.lpmn(m,n,x)[0][-1,-1]
        return special_function.lpmv(m, n, x)


    def setup_legendre_polynomials(self):
        """This method precomputes the Legendre polynomials P_j(z) for j=0,...,jmax at the quadrature points z.
        The precomputed polynomials are stored in the list self.P for later use in the partial-wave decomposition of the potential.
        The method uses the legP function to compute the Legendre polynomials at the quadrature points defined by self.z, which are the roots of the Legendre polynomial of degree self.ntheta. The computed polynomials are stored in the list self.P, where each entry corresponds to a different degree j of the Legendre polynomial.
        """
        fP = np.vectorize(self.legP, excluded={0, 1}, otypes=[np.float64])
                     
        for j in range(0,self.jmax):
            this_P = fP(0,j,self.z)
            self.P.append(this_P)
        

    def pwd_integral(self,W,l,j):
        """This method calculates the partial-wave decomposition integral for a given potential component W, orbital angular momentum l, and total angular momentum j.

        Args:
            W (array-like): array of potential values evaluated at the quadrature points for cos(theta)
            l (int): orbital angular momentum quantum number
            j (int): total angular momentum quantum number

        Returns:
            float: value of the partial-wave decomposition integral
        """

        if j<0:
            return
        return np.pi*np.sum(W*self.zJ[l]*self.P[j]*self.w)
        
    @staticmethod
    def iso(t):
        """ This static method calculates the isospin factor for the one-pion exchange potential based on the total isospin (t) of the two-nucleon system. The isospin factor is -3 for an isosinglet state (t=0) and +1 for an isotriplet state (t=1). If the input t is not 0 or 1, the method exits with an error message.

        Args:
            t (int): isospin of the two-nucleon system (0 for isosinglet, 1 for isotriplet)

        Returns:
            float: isospin factor for the one-pion exchange potential, which is -3 for isosinglet (t=0) and +1 for isotriplet (t=1)
        """
        if t==0:
            return -3.0
        elif t==1:
            return +1.0
        else:
            sys.exit("isospin must be zero or one")

    def freg(self,p):
        """
        This method calculates the regulator function for the potential based on the momentum p and the cutoff parameter Lambda. The regulator function is defined as f(p) = exp(-(p/Lambda)^6), which suppresses contributions from high momenta above the cutoff scale Lambda. The method takes as input the momentum p and returns the value of the regulator function, which is used to regularize the potential in momentum space.

        Args:
            p (float): momentum variable used in the calculation of the regulator function. This variable represents the momentum of the nucleons in the potential and is used to determine how much the potential is suppressed at high momenta based on the cutoff parameter Lambda.

        Returns:
            float: value of the regulator function f(p) = exp(-(p/Lambda)^6), which is used to regularize the potential in momentum space. The function suppresses contributions from high momenta above the cutoff scale Lambda, ensuring that the potential remains well-behaved at high energies.
        """
        return np.exp(-(p/self.Lambda)**6)

    # partial-wave decomposition of central force
    def pwd_C(self,W,pp,p,coup,s,j):
        """
        This method performs the partial-wave decomposition of the central force component of the nucleon-nucleon potential. It takes as input the potential values W evaluated at the quadrature points for cos(theta), the momenta pp and p of the bra and ket states in the partial-wave decomposition, a flag coup indicating whether the channel is coupled or uncoupled, the total spin s of the two-nucleon system, and the total angular momentum j. The method calculates the central force components based on these inputs and returns an array V containing the decomposed components.

        Args:
            W (array-like): array of potential values evaluated at the quadrature points for cos(theta)
            pp (float): momentum of the bra state in the partial-wave decomposition
            p (float): momentum of the ket state in the partial-wave decomposition
            coup (bool): flag indicating whether the channel is coupled (True) or uncoupled (False)
            s (int): total spin of the two-nucleon system (0 for singlet, 1 for triplet)
            j (int): total angular momentum of the two-nucleon system


        Returns:
            np.ndarray: array of partial-wave decomposed central force components
        """
        V = np.zeros(6)
        
        # uncoupled singlet
        if not coup and s==0:
            V[0] = 2*self.pwd_integral(W,0,j)
        # uncoupled triplet
        elif not coup and s==1:
            V[1] = 2*self.pwd_integral(W,0,j)
        elif coup:
            V[2] = 2*self.pwd_integral(W,0,j+1)

            # 3P0 case 
            if j==0:
                return V

            V[3] = 2*self.pwd_integral(W,0,j-1)
            V[4] = 0
            V[5] = 0
            
        return V

    # partial-wave decomposition of spin-spin force
    def pwd_S(self,W,pp,p,coup,s,j):
        """This method performs the partial-wave decomposition of the spin-spin force component of the nucleon-nucleon potential. It takes as input the potential values W evaluated at the quadrature points for cos(theta), the momenta pp and p of the bra and ket states in the partial-wave decomposition, a flag coup indicating whether the channel is coupled or uncoupled, the total spin s of the two-nucleon system, and the total angular momentum j. The method calculates the spin-spin force components based on these inputs and returns an array V containing the decomposed components.

        Args:
            W (array-like): array of potential values evaluated at the quadrature points for cos(theta)
            pp (float): momentum of the bra state in the partial-wave decomposition
            p (float): momentum of the ket state in the partial-wave decomposition
            coup (bool): flag indicating whether the channel is coupled (True) or uncoupled (False)
            s (int): total spin of the two-nucleon system (0 for singlet, 1 for triplet)
            j (int): total angular momentum of the two-nucleon system

            
        Returns:
            np.ndarray: array of partial-wave decomposed spin-spin force components
        """
        V = np.zeros(6)
        
        # uncoupled singlet
        if not coup and s==0:
            V[0] = -6*self.pwd_integral(W,0,j)
        # uncoupled triplet
        elif not coup and s==1:
            V[1] = 2*self.pwd_integral(W,0,j)
        elif coup:
            V[2] = 2*self.pwd_integral(W,0,j+1)

            # 3P0 case 
            if j==0:
                return V

            V[3] = 2*self.pwd_integral(W,0,j-1)
            V[4] = 0
            V[5] = 0
            
        return V
    
    # partial-wave decomposition of tensor force
    def pwd_T(self,W,pp,p,coup,s,j):
        """This method performs the partial-wave decomposition of the tensor force component of the nucleon-nucleon potential. It takes as input the potential values W evaluated at the quadrature points for cos(theta), the momenta pp and p of the bra and ket states in the partial-wave decomposition, a flag coup indicating whether the channel is coupled or uncoupled, the total spin s of the two-nucleon system, and the total angular momentum j. The method calculates the tensor force components based on these inputs and returns an array V containing the decomposed components.

        Args:
            W (array-like): array of potential values evaluated at the quadrature points for cos(theta)
            pp (float): momentum of the bra state in the partial-wave decomposition
            p (float): momentum of the ket state in the partial-wave decomposition
            coup (bool): flag indicating whether the channel is coupled (True) or uncoupled (False)
            s (int): total spin of the two-nucleon system (0 for singlet, 1 for triplet)
            j (int): total angular momentum of the two-nucleon system

        Returns:
            np.ndarray: array of partial-wave decomposed tensor force components
        """
        V = np.zeros(6)
        
        jj  = 2*j+1
        jj1 = j*(j+1) 

        pp2 = pp**2
        p2  = p**2

        # uncoupled singlet
        if not coup and s==0:
            V[0] = 2*( -1*(pp2+p2)*self.pwd_integral(W,0,j) + 2*pp*p*self.pwd_integral(W,1,j))
        # uncoupled triplet
        elif not coup and s==1:
            V[1] = 2*( (pp2+p2)*self.pwd_integral(W,0,j) - 2*pp*p*(j*self.pwd_integral(W,0,j+1) + (j+1)*self.pwd_integral(W,0,j-1))/jj)
        elif coup:
            V[2] = 2*( -1*(pp2+p2)*self.pwd_integral(W,0,j+1) + 2*pp*p*self.pwd_integral(W,0,j))/jj

            # 3P0 case 
            if j==0:
                return V

            V[3] = 2*( (pp2+p2)*self.pwd_integral(W,0,j-1) - 2*pp*p*self.pwd_integral(W,0,j))/jj
            V[4] = -4*np.sqrt(jj1)*(p*p*self.pwd_integral(W,0,j+1) + pp*pp*self.pwd_integral(W,0,j-1) - 2*pp*p*self.pwd_integral(W,0,j))/jj
            V[5] = -4*np.sqrt(jj1)*(p*p*self.pwd_integral(W,0,j-1) + pp*pp*self.pwd_integral(W,0,j+1) - 2*pp*p*self.pwd_integral(W,0,j))/jj
            
        return V
        
    def contact_lo(self,pp,p,coup,s,j,t,tz,lecs):
        """This method calculates the leading-order contact interaction component of the nucleon-nucleon potential in partial-wave decomposition. It takes as input the momenta pp and p of the bra and ket states in the partial-wave decomposition, a flag coup indicating whether the channel is coupled or uncoupled, the total spin s of the two-nucleon system, the total angular momentum j, the total isospin t, the projection of the total isospin tz, and a dictionary lecs containing the low-energy constants (LECs) for the contact interactions at leading order. The method determines which contact interaction terms contribute based on the quantum numbers of the channel and returns an array V containing the decomposed contact interaction components.

        Args:
            pp (float): momentum of the bra state in the partial-wave decomposition
            p (float): momentum of the ket state in the partial-wave decomposition
            coup (bool): flag indicating whether the channel is coupled (True) or uncoupled (False)
            s (int): total spin of the two-nucleon system (0 for singlet, 1 for triplet)
            j (int): total angular momentum of the two-nucleon system
            t (int): total isospin of the two-nucleon system
            tz (int): projection of the total isospin along the z-axis
            lecs (dict): dictionary containing the low-energy constants (LECs) for the contact interactions at leading order

        Returns:
            np.ndarray: array of partial-wave decomposed contact interaction components at leading order
        """
        V = np.zeros(6)
        #10^4 GeV^-2 - > 1e-2 MeV^-2
        if j==0 and s==0 and not coup:
             #1S0
            V[0] += lecs['C_1S0']*1e-2 
        elif j==1 and s==1 and coup:
            #3S1
            V[3] += lecs['C_3S1']*1e-2 

        return V

    def contact_nlo(self,pp,p,coup,s,j,t,tz,lecs):
        """This method calculates the next-to-leading-order contact interaction component of the nucleon-nucleon potential in partial-wave decomposition. It takes as input the momenta pp and p of the bra and ket states in the partial-wave decomposition, a flag coup indicating whether the channel is coupled or uncoupled, the total spin s of the two-nucleon system, the total angular momentum j, the total isospin t, the projection of the total isospin tz, and a dictionary lecs containing the low-energy constants (LECs) for the contact interactions at next-to-leading order. The method determines which contact interaction terms contribute based on the quantum numbers of the channel and returns an array V containing the decomposed contact interaction components.

        Args:
            pp (float): momentum of the bra state in the partial-wave decomposition
            p (float): momentum of the ket state in the partial-wave decomposition
            coup (bool): flag indicating whether the channel is coupled (True) or uncoupled (False)
            s (int): total spin of the two-nucleon system (0 for singlet, 1 for triplet)
            j (int): total angular momentum of the two-nucleon system
            t (int): total isospin of the two-nucleon system
            tz (int): projection of the total isospin along the z-axis
            lecs (dict): dictionary containing the low-energy constants (LECs) for the contact interactions at next-to-leading order

        Returns:
            np.ndarray: array of partial-wave decomposed contact interaction components at next-to-leading order
        """
        V = np.zeros(6)
        #10^4 GeV^-4 - > 1e-8 MeV^-4
        if j==0 and s==0 and not coup:
            #1S0
            V[0] += (lecs['D_1S0']*1e-8)*(p*p+pp*pp)
        elif j==0 and s==1 and coup:
            #3P0
            V[2] += (lecs['D_3P0']*1e-8)*p*pp
        elif j==1 and s==0 and not coup:
            #1P1
            V[0] += (lecs['D_1P1']*1e-8)*p*pp
        elif j==1 and s==1 and not coup:
            #3P1
            V[1] += (lecs['D_3P1']*1e-8)*p*pp
        elif j==1 and s==1 and coup:
            #3S1
            V[3] += (lecs['D_3S1']*1e-8)*(p*p + pp*pp)
            #3S1-3D1
            V[5] += (lecs['D_3S1-3D1']*1e-8)*p*p
            #3D1-3S1
            V[4] += (lecs['D_3S1-3D1']*1e-8)*pp*pp
        elif j==2 and s==1 and coup:
            #3P2
            V[3] += (lecs['D_3P2']*1e-8)*p*pp
            
        return V
    
    def one_pion_exchange(self,q2):
        """This method calculates the one-pion exchange potential in momentum space based on the squared momentum transfer q^2 between the two nucleons. The potential is calculated using the formula V_1pi(q^2) = - (g_A^2 / (4 f_pi^2)) * (1 / (q^2 + m_pi^2)), where g_A is the axial coupling constant, f_pi is the pion decay constant, and m_pi is the pion mass. The method returns the value of the one-pion exchange potential, which is negative and depends on the momentum transfer q^2 and the properties of the pion.

        Args:
            q2 (float): squared momentum transfer between the two nucleons, defined as q^2 = (p' - p)^2 where p' and p are the momenta of the outgoing and incoming nucleons, respectively. This variable is used in the calculation of the one-pion exchange potential.

        Returns:
            float: value of the one-pion exchange potential in momentum space, calculated using the formula V_1pi(q^2) = - (g_A^2 / (4 f_pi^2)) * (1 / (q^2 + m_pi^2)), where g_A is the axial coupling constant, f_pi is the pion decay constant, and m_pi is the pion mass. The potential is negative and depends on the momentum transfer q^2 and the properties of the pion.
        """
        prefactor = -1.0*self.gA**2/(4*self.fpi**2) 
        return prefactor / (q2 + self.mpi**2)
    
    def leading_two_pion_exchange_WC(self,q2,loop):
        """This method calculates the leading two-pion exchange central potential component in momentum space based on the squared momentum transfer q^2 between the two nucleons and the value of the two-pion exchange loop function evaluated at that q^2. The potential is calculated using the formula V_2pi_C(q^2) = - (1/(384 pi^2 f_pi^4)) * loop * [4 m_pi^2 (5 g_A^4 - 4 g_A^2 - 1) + q^2 (23 g_A^4 - 10 g_A^2 - 1) + (48 g_A^4 m_pi^4) / w(q^2)^2], where g_A is the axial coupling constant, f_pi is the pion decay constant, m_pi is the pion mass, and w(q^2) = sqrt(4 m_pi^2 + q^2). The method returns the value of the leading two-pion exchange central potential component, which depends on the momentum transfer q^2, the properties of the pion, and the value of the two-pion exchange loop function.

        Args:
            q2 (float): squared momentum transfer between the two nucleons, defined as q^2 = (p' - p)^2 where p' and p are the momenta of the outgoing and incoming nucleons, respectively. This variable is used in the calculation of the leading two-pion exchange central potential component.
            loop (float): value of the two-pion exchange loop function evaluated at the given q^2. This variable is used in the calculation of the leading two-pion exchange central potential component.

        Returns:
            float: value of the leading two-pion exchange central potential component in momentum space, calculated using the formula V_2pi_C(q^2) = - (1/(384 pi^2 f_pi^4)) * loop * [4 m_pi^2 (5 g_A^4 - 4 g_A^2 - 1) + q^2 (23 g_A^4 - 10 g_A^2 - 1) + (48 g_A^4 m_pi^4) / w(q^2)^2], where g_A is the axial coupling constant, f_pi is the pion decay constant, m_pi is the pion mass, and w(q^2) = sqrt(4 m_pi^2 + q^2). The potential depends on the momentum transfer q^2, the properties of the pion, and the value of the two-pion exchange loop function.
        """
        prefactor = -1*loop/(384*np.pi**2*self.fpi**4)
        w = np.sqrt(4*self.mpi**2 + q2)
        return prefactor * ( 4*self.mpi**2*(5*self.gA**4 - 4*self.gA**2 - 1) + q2*(23*self.gA**4 - 10*self.gA**2 - 1) + (48*self.gA**4*self.mpi**4)/w**2)

    def leading_two_pion_exchange_VS(self,q2,loop):
        
        """
        This method calculates the leading two-pion exchange spin-spin potential component in momentum space based on the squared momentum transfer q^2 between the two nucleons and the value of the two-pion exchange loop function evaluated at that q^2. The potential is calculated using the formula V_2pi_S(q^2) = (3 g_A^4 loop / (64 pi^2 f_pi^4)) * q^2, where g_A is the axial coupling constant, f_pi is the pion decay constant, and loop is the value of the two-pion exchange loop function evaluated at the given q^2. The method returns the value of the leading two-pion exchange spin-spin potential component, which depends on the momentum transfer q^2, the properties of the pion, and the value of the two-pion exchange loop function.
        
        Args:
            q2 (float): squared momentum transfer between the two nucleons, defined as q^2 = (p' - p)^2 where p' and p are the momenta of the outgoing and incoming nucleons, respectively. This variable is used in the calculation of the leading two-pion exchange spin-spin potential component.
            loop (float): value of the two-pion exchange loop function evaluated at the given q^2. This variable is used in the calculation of the leading two-pion exchange spin-spin potential component.

        Returns:
            float: value of the leading two-pion exchange spin-spin potential component in momentum space, calculated using the formula V_2pi_S(q^2) = (3 g_A^4 loop / (64 pi^2 f_pi^4)) * q^2, where g_A is the axial coupling constant, f_pi is the pion decay constant, and loop is the value of the two-pion exchange loop function evaluated at the given q^2. The potential depends on the momentum transfer q^2, the properties of the pion, and the value of the two-pion exchange loop function.
        """
        
        prefactor = 3*self.gA**4*loop/(64*np.pi**2*self.fpi**4)
        return prefactor * q2

    def leading_two_pion_exchange_VT(self,q2,loop):
        """
        This method calculates the leading two-pion exchange tensor potential component in momentum space based on the squared momentum transfer q^2 between the two nucleons and the value of the two-pion exchange loop function evaluated at that q^2. The potential is calculated using the formula V_2pi_T(q^2) = - (3 g_A^4 loop / (64 pi^2 f_pi^4)), where g_A is the axial coupling constant, f_pi is the pion decay constant, and loop is the value of the two-pion exchange loop function evaluated at the given q^2. The method returns the value of the leading two-pion exchange tensor potential component, which depends on the momentum transfer q^2, the properties of the pion, and the value of the two-pion exchange loop function.

        Args:
            q2 (float): squared momentum transfer between the two nucleons, defined as q^2 = (p' - p)^2 where p' and p are the momenta of the outgoing and incoming nucleons, respectively. This variable is used in the calculation of the leading two-pion exchange tensor potential component.
            loop (float): value of the two-pion exchange loop function evaluated at the given q^2. This variable is used in the calculation of the leading two-pion exchange tensor potential component.

        Returns:
            float: value of the leading two-pion exchange tensor potential component in momentum space, calculated using the formula V_2pi_T(q^2) = - (3 g_A^4 loop / (64 pi^2 f_pi^4)), where g_A is the axial coupling constant, f_pi is the pion decay constant, and loop is the value of the two-pion exchange loop function evaluated at the given q^2. The potential depends on the momentum transfer q^2, the properties of the pion, and the value of the two-pion exchange loop function.
        """
        prefactor = 3*self.gA**4*loop/(64*np.pi**2*self.fpi**4)
        return -1*prefactor
    
    def leading_two_pion_exchange_loop_DR(self,q2):
        """
        
        This method calculates the leading two-pion exchange loop function in dimensional regularization based on the squared momentum transfer q^2 between the two nucleons. The loop function is calculated using the formula L(q^2) = w(q^2) * log((w(q^2) + q) / (2 m_pi)) / q, where w(q^2) = sqrt(4 m_pi^2 + q^2) and m_pi is the pion mass. The method returns the value of the leading two-pion exchange loop function, which depends on the momentum transfer q^2 and the properties of the pion. This loop function is used in the calculation of the leading two-pion exchange potential components at next-to-leading order in chiral effective field theory.

        Args:
            q2 (float): squared momentum transfer between the two nucleons, defined as q^2 = (p' - p)^2 where p' and p are the momenta of the outgoing and incoming nucleons, respectively. This variable is used in the calculation of the leading two-pion exchange loop function.

        Returns:
            float: value of the leading two-pion exchange loop function in dimensional regularization, calculated using the formula L(q^2) = w(q^2) * log((w(q^2) + q) / (2 m_pi)) / q, where w(q^2) = sqrt(4 m_pi^2 + q^2) and m_pi is the pion mass. The loop function depends on the momentum transfer q^2 and the properties of the pion.
        """
        q = np.sqrt(q2)
        w = np.sqrt(4*self.mpi**2 + q2)
        L = w*np.log((w+q)/(2*self.mpi))/q
        return L
        
    def potential(self,pp,p,coup,s,j,t,tz,lecs):
        """
        This method calculates the nucleon-nucleon potential in momentum space for given momenta pp and p of the bra and ket states in the partial-wave decomposition, a flag coup indicating whether the channel is coupled or uncoupled, the total spin s of the two-nucleon system, the total angular momentum j, the total isospin t, the projection of the total isospin tz, and a dictionary lecs containing the low-energy constants (LECs) for the contact interactions. The method computes the potential by summing contributions from one-pion exchange, two-pion exchange, and contact interactions based on the specified chiral order (LO or NLO) and returns the total potential in momentum space.

        Args:
            pp (float): momentum of the bra state in the partial-wave decomposition
            p (float): momentum of the ket state in the partial-wave decomposition
            coup (bool): flag indicating whether the channel is coupled (True) or uncoupled (False)
            s (int): total spin of the two-nucleon system (0 for singlet, 1 for triplet)
            j (int): total angular momentum of the two-nucleon system
            t (int): total isospin of the two-nucleon system
            tz (int): projection of the total isospin along the z-axis
            lecs (dict): dictionary containing the low-energy constants (LECs) for the contact interactions at leading and next-to-leading order

        Returns:
            np.ndarray: array containing the values of the nucleon-nucleon potential components in momentum space, calculated by summing contributions from one-pion exchange, two-pion exchange, and contact interactions based on the specified chiral order (LO or NLO). The potential is returned as an array of values corresponding to the different components of the potential in the partial-wave decomposition.
        """

        if coup:
            assert s == 1, 'coupled NN channel must be a spin triplet'
       
        V = np.zeros(6)
        
        q2 = pp**2 + p**2 - 2*pp*p*self.z
        if self.chiral_order == 'LO':
        
            WT = self.iso(t) * self.one_pion_exchange(q2)
            V  += self.pwd_T(WT,pp,p,coup,s,j)
            V  += self.contact_lo(pp,p,coup,s,j,t,tz,lecs)
            
        elif self.chiral_order == 'NLO':

            # In this version of NLO we ignore isospin breaking effects due to the pion-mass splitting
            # and use the isospin symmetric one-pion exhange
            WT = self.iso(t) * self.one_pion_exchange(q2)
            V  += self.pwd_T(WT,pp,p,coup,s,j)

            V  += self.contact_lo(pp,p,coup,s,j,t,tz,lecs)
            
            # two-pion exchange loop function in dimensional regularization
            loop = self.leading_two_pion_exchange_loop_DR(q2)

            WC = self.iso(t) * self.leading_two_pion_exchange_WC(q2,loop)
            V += self.pwd_C(WC,pp,p,coup,s,j)

            VS = self.leading_two_pion_exchange_VS(q2,loop)
            V += self.pwd_S(VS,pp,p,coup,s,j)

            VT = self.leading_two_pion_exchange_VT(q2,loop)
            V += self.pwd_T(VT,pp,p,coup,s,j)

            V += self.contact_nlo(pp,p,coup,s,j,t,tz,lecs)
                        
        else:
            print(f'only implemented LO and NLO in Weinberg counting.')        

        # 1/2pi^3 normalization
        return (0.125/np.pi**3)*self.freg(pp)*V*self.freg(p)
        
    
 

