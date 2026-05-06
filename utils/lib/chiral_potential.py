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

    def __init__(self,chiral_order, Lambda):

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
        # [0]: get function value (not derivative)
        # [-1,-1]: get 'm', 'n' values
        #return special_function.lpmn(m,n,x)[0][-1,-1]
        return special_function.lpmv(m, n, x)

    def setup_legendre_polynomials(self):

        fP = np.vectorize(self.legP, excluded={0, 1}, otypes=[np.float64])
                     
        for j in range(0,self.jmax):
            this_P = fP(0,j,self.z)
            self.P.append(this_P)
        
    def pwd_integral(self,W,l,j):

        if j<0:
            return
        return np.pi*np.sum(W*self.zJ[l]*self.P[j]*self.w)
        
    @staticmethod
    def iso(t):
        if t==0:
            return -3.0
        elif t==1:
            return +1.0
        else:
            sys.exit("isospin must be zero or one")

    def freg(self,p):
        return np.exp(-(p/self.Lambda)**6)

    # partial-wave decomposition of central force
    def pwd_C(self,W,pp,p,coup,s,j):

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
        prefactor = -1.0*self.gA**2/(4*self.fpi**2) 
        return prefactor / (q2 + self.mpi**2)
    
    def leading_two_pion_exchange_WC(self,q2,loop):
        prefactor = -1*loop/(384*np.pi**2*self.fpi**4)
        w = np.sqrt(4*self.mpi**2 + q2)
        return prefactor * ( 4*self.mpi**2*(5*self.gA**4 - 4*self.gA**2 - 1) + q2*(23*self.gA**4 - 10*self.gA**2 - 1) + (48*self.gA**4*self.mpi**4)/w**2)

    def leading_two_pion_exchange_VS(self,q2,loop):
        prefactor = 3*self.gA**4*loop/(64*np.pi**2*self.fpi**4)
        return prefactor * q2

    def leading_two_pion_exchange_VT(self,q2,loop):
        prefactor = 3*self.gA**4*loop/(64*np.pi**2*self.fpi**4)
        return -1*prefactor
    
    def leading_two_pion_exchange_loop_DR(self,q2):
        q = np.sqrt(q2)
        w = np.sqrt(4*self.mpi**2 + q2)
        L = w*np.log((w+q)/(2*self.mpi))/q
        return L
        
    def potential(self,pp,p,coup,s,j,t,tz,lecs):

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
        
    
 

