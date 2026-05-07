import utils.lib.constants as const
import numpy as np

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

class nn_studio:
    """
        A class to compute NN scattering observables from a given potential, and to perform model order reduction for emulation.
    """

    def __init__(self, jmin, jmax, tzmin, tzmax, Np=75, mesh_type='gauleg_infinite'):

        """
        Initialize the nn_studio class with the specified parameters for the NN basis and channels, the momentum mesh, and the laboratory kinetic energies for which to compute the T-matrix and phase shifts.

        Args:
            jmin (int): The minimum total angular momentum J to include in the NN basis and channels.
            jmax (int): The maximum total angular momentum J to include in the NN basis and channels.
            tzmin (int): The minimum isospin projection tz to include in the NN basis and channels.
            tzmax (int): The maximum isospin projection tz to include in the NN basis and channels.
            Np (int, optional): The number of momentum points in the mesh. Defaults to 75.
            mesh_type (str, optional): The type of momentum mesh to use, either 'gauleg_infinite' for a Gauss-Legendre mesh on the interval [0, inf), or 'gauleg_finite' for a Gauss-Legendre mesh on a finite interval. Defaults to 'gauleg_infinite'.

        """

        self.jmin = jmin
        self.jmax = jmax
        self.tzmin = tzmin
        self.tzmax = tzmax
        
        self.basis = self.setup_NN_basis()
        self.channels = self.setup_NN_channels()
        
        self.Np = Np
        if mesh_type == 'gauleg_infinite':
            self.pmesh, self.wmesh = self.gauss_legendre_inf_mesh()
        elif mesh_type == 'gauleg_finite':
            self.pmesh, self.wmesh = self.gauss_legendre_line_mesh(1e-16,1000)
            
        self.lecs = None

        self.Tlabs = None
        
        #potential
        self.V = None

        #Tmatrices
        self.Tmtx = []

        #phase shifts
        self.phase_shifts = []
        
    def setup_NN_basis(self):
        
        """
        Construct the NN basis states in the JLS coupling scheme, with isospin projection tz, and parity pi=(-1)^L. We include all states with jmin <= J <= jmax and tzmin <= tz <= tzmax, and we only include states that satisfy the Pauli principle, i.e., (L+S+T) must be odd.

        Returns:
            list of dicts: A list of dictionaries representing the NN basis states.
        """

        basis = []
        for tz in range(self.tzmin,self.tzmax+1,1):
            for J in range(self.jmin,self.jmax+1,1):
                for S in range(0,2,1):
                    for L in range(abs(J-S),J+S+1,1):
                        for T in range(abs(tz),2,1):
                            if ((L+S+T)%2 != 0):
                                basis_state = {}
                                basis_state['tz'] = tz
                                basis_state['l']  = L
                                basis_state['pi'] = (-1)**L
                                basis_state['s']  = S
                                basis_state['j']  = J
                                basis_state['t']  = T
                                basis.append(basis_state)
        return basis

    def setup_NN_channels(self):

        """
        Construct the NN channels in the JLS coupling scheme, with isospin projection tz, and parity pi=(-1)^L. We include all states with jmin <= J <= jmax and tzmin <= tz <= tzmax, and we only include states that satisfy the Pauli principle, i.e., (L+S+T) must be odd.

        Returns:
            list of lists of dicts: A list of lists of dictionaries representing the NN channels.
        """

        from itertools import groupby
        from operator import itemgetter
        
        states = []
        
        for bra in self.basis:
            for ket in self.basis:
            
                if self.kroenecker_delta(bra,ket,'j','tz','s','pi'):
                
                    state = {}
                    
                    state['l']  = bra['l']
                    state['ll'] = ket['l']
                    
                    state['s']  = bra['s']
                    state['j']  = bra['j']
                    state['t']  = bra['t']
                    state['tz'] = bra['tz']
                    state['pi'] = bra['pi']
                    states.append(state)
                    
        grouper = itemgetter("s", "j", "tz", "pi")
        NN_channels = []
    
        for key, grp in groupby(sorted(states, key = grouper), grouper):
            NN_channels.append(list(grp))
        

        for chn_idx, chn in enumerate(NN_channels):
            for block in chn:
                block.update({"chn_idx":chn_idx})

        return NN_channels

    def lookup_channel_idx(self, **kwargs):

        """
        Lookup the channel index for a given set of quantum numbers. The quantum numbers are specified as keyword arguments, e.g., s=1, j=0, tz=0, pi=1. The function will return the indices of the channels that match the specified quantum numbers.

        Args:
            **kwargs: The quantum numbers to match, specified as keyword arguments.
        
        Returns:
            tuple: A tuple containing the indices of the matching channels and the channels themselves.
        """

        matching_indices = []
        channels = []
        for idx, chn in enumerate(self.channels):
            for block in chn:
                #this will return a channel if any of the partial-wave couplings of a block match 
                if (kwargs.items() <= block.items()):
                    matching_indices.append(idx)
                    channels.append(chn)

        matching_indices = list(dict.fromkeys(matching_indices))

        return matching_indices, channels

    def linear_mesh(self):
        """
        Construct a linear mesh for the momentum integration. The mesh points are evenly spaced between 1e-6 and 650 MeV, and the weights are given by the spacing between the points.

        Returns:
            np.ndarray: An array of mesh points.
        """

        return np.linspace(1e-6,650,self.Np)
    
    def gauss_legendre_line_mesh(self,a,b):

        """
        Construct a Gauss-Legendre mesh for the momentum integration. The mesh points and weights

        Args:
            a (float): The lower bound of the interval for the momentum integration.
            b (float): The upper bound of the interval for the momentum integration.

        Returns:
            tuple: A tuple containing the mesh points and weights.
        """

        x, w = np.polynomial.legendre.leggauss(self.Np)
        # Translate x values from the interval [-1, 1] to [a, b]
        t = 0.5*(x + 1)*(b - a) + a
        u = w * 0.5*(b - a)

        return t,u
    
    def gauss_legendre_inf_mesh(self):

        """
        Construct a Gauss-Legendre mesh for the momentum integration on the interval [0, inf). The mesh points and weights are given by the Gauss-Legendre quadrature on the interval [-1, 1], but translated to the interval [0, inf) using the transformation t = scale * tan(pi/4 * (x + 1)), where x are the Gauss-Legendre points. The weights are also transformed accordingly.

        Returns:
            tuple: A tuple containing the mesh points and weights.
        """

        scale=100.0
        
        x, w = np.polynomial.legendre.leggauss(self.Np)
        
        # Translate x values from the interval [-1, 1] to [0, inf)
        pi_over_4 = np.pi/4.0
        
        t = scale*np.tan(pi_over_4*(x+1.0))
        u = scale*pi_over_4/np.cos(pi_over_4*(x+1.0))**2*w
        
        return t,u

    @staticmethod
    #a static method is bound to a class rather than the objects for that class
    def triag(a, b, ab):
        """
        Check if the triangle inequality is satisfied for the given values of a, b, and ab. This is used to check if the coupling of two angular momenta a and b can give a total angular momentum ab.

        Args:
            a (float): The first angular momentum.
            b (float): The second angular momentum.
            ab (float): The total angular momentum.

        Returns:
            bool: True if the triangle inequality is satisfied, False otherwise.
        """

        if( ab < abs(a - b) ):
            return False
        if ( ab > a + b ):
            return False
        return True
    
    @staticmethod
    #a static method is bound to a class rather than the objects for that class
    def kroenecker_delta(bra,ket,*args):

        """
        Check if the specified quantum numbers of the bra and ket states are equal. The quantum numbers to check are specified as arguments, e.g., 'j', 'tz', 's', 'pi'. The function will return True if all specified quantum numbers are equal, and False otherwise.

        Args:
            bra (dict): The bra state.
            ket (dict): The ket state.
            *args: The quantum numbers to check.

        Returns:
            bool: True if all specified quantum numbers are equal, False otherwise.
        """


        for ar in args:
            if bra[ar] != ket[ar]:
                return False
        return True

    def lab2rel(self,Tlab,tz):

        """
        Convert the laboratory kinetic energy Tlab to the relative momentum ko and the reduced mass mu for the given isospin projection tz. The formulas for the conversion depend on the value of tz, which determines the type of scattering (pp, np, or nn).

        Args:
            Tlab (float): The laboratory kinetic energy in MeV.
            tz (int): The isospin projection, where tz=-1 corresponds to pp scattering, tz=0 corresponds to np scattering, and tz=+1 corresponds to nn scattering.
        
        Returns:
            tuple: A tuple containing the relative momentum ko and the reduced mass mu.
        """
    
        if tz == -1:
            mu = const.Mp/2
            ko2 = 2*const.Mp*Tlab
        elif tz ==  0:
            mu = const.Mp*const.Mn/(const.Mp+const.Mn)
            ko2 = const.Mp**2*Tlab*(Tlab+2*const.Mn)/((const.Mp+const.Mn)**2 + 2*Tlab*const.Mp)
        elif tz == +1:
            mu = const.mN/2
            ko2 = 2*const.Mp*Tlab
        else:
            exit('unknown isospin projection')

        if ko2<0:
            ko = np.complex(0,np.sqrt(np.abs(ko2)))
        else:    
            ko = np.sqrt(ko2)

        return ko,mu

    @staticmethod
    #a static method is bound to a class rather than the objects for that class
    def map_to_coup_idx(ll,l,s,j):

        """
        Map the quantum numbers of the partial-wave coupling to the index of the potential matrix element. The mapping is based on the coupling scheme and the order of the potential matrix elements in the code.
        The mapping is as follows:

        l=ll, s=1, l<j: idx=3 (--)

        l=ll, s=1, l>j: idx=2 (++)

        l=ll, s=1, l=j: idx=1 (uncoupled)

        l=ll, s=0, l=j: idx=0 (uncoupled)

        l!=ll, l<j: idx=5 (-+)

        l!=ll, l>j: idx=4 (+-)

        Args:
            ll (int): The orbital angular momentum of the ket state.
            l (int): The orbital angular momentum of the bra state.
            s (int): The total spin of the two nucleons.
            j (int): The total angular momentum of the two nucleons.

        Returns:
            tuple: A tuple containing a boolean indicating if the coupling is valid and the index of the potential matrix element.
        """

        if l == ll:
        
            if l<j:
                # --
                coup = True
                idx  = 3
            elif l>j:
                # ++
                coup = True
                idx  = 2
            else:
                if s==1:
                    coup = False
                    idx  = 1
                else:
                    coup = False
                    idx  = 0
        else:
            if l<j:
                # -+
                coup = True
                idx  = 5
            else:
                # +-
                coup = True
                idx  = 4

        return coup,idx
    
    def Vmtx(self,this_mesh,ll,l,s,j,t,tz):

        """
        Construct the potential matrix for the given quantum numbers and the momentum mesh.
        The potential matrix is constructed by evaluating the potential function for each pair of momentum points in the mesh, and for the given quantum numbers of the partial-wave coupling.

        Args:
            this_mesh (np.ndarray): The momentum mesh for which to construct the potential matrix.
            ll (int): The orbital angular momentum of the ket state.
            l (int): The orbital angular momentum of the bra state.
            s (int): The total spin of the two nucleons.
            j (int): The total angular momentum of the two nucleons.
            t (int): The total isospin of the two nucleons.
            tz (int): The isospin projection of the two nucleons.

        Returns:
            np.ndarray: The potential matrix for the given quantum numbers and momentum mesh.
        """
    
        coup,idx = self.map_to_coup_idx(ll,l,s,j)
        mtx = np.zeros((len(this_mesh), len(this_mesh)))
        for pidx, p in enumerate(this_mesh):
            for ppidx, pp in enumerate(this_mesh):
                mtx[ppidx][pidx] = self.V.potential(pp,p,coup,s,j,t,tz,self.lecs)[idx]
        return np.array(mtx)
    
    def setup_Vmtx(self,this_channel,ko=False):

        """
        Construct the potential matrix for the given channel and the momentum mesh.
        The potential matrix is constructed by evaluating the potential function for each pair of momentum points in the mesh, and for the quantum numbers of the partial-wave coupling specified in the channel.

        Args:
            this_channel (list): The channel for which to construct the potential matrix. The channel is a list of blocks, where each block is a dictionary containing the quantum numbers of the partial-wave coupling.
            ko (bool, optional): Whether to include the on-shell momentum ko in the potential matrix. Defaults to False.

        Returns:
            np.ndarray: The potential matrix for the given channel and momentum mesh.
        """

        if ko==False:
            this_mesh = self.pmesh
        else:
            this_mesh = np.hstack((self.pmesh,ko))
        m = []

        for idx, block in enumerate(this_channel):
            
            l  = block['l']
            ll = block['ll']
            s  = block['s']
            j  = block['j']
            t  = block['t']
            tz = block['tz']
                
            mtx = np.copy(self.Vmtx(this_mesh,ll,l,s,j,t,tz))

            m.append(mtx)

        if len(this_channel) >1:
            V = np.copy(np.vstack((np.hstack((m[0],m[1])),
                                   np.hstack((m[2],m[3])))))
        else:
            V = np.copy(m[0])
                   
        return V, m

    def setup_G0_vector(self,ko,mu):

        """
        Construct the G0 vector for the given relative momentum ko and reduced mass mu.
        The G0 vector is constructed by evaluating the free Green's function for each momentum point in the mesh, and for the on-shell momentum ko.

        Args:
            ko (float): The relative momentum.
            mu (float): The reduced mass.

        Returns:
            np.ndarray: The G0 vector for the given relative momentum ko and reduced mass mu.
        """
            
        G = np.zeros((2*self.Np+2), dtype=complex)

        # note that we index from zero, and the N+1 point is at self.Np
        G[0:self.Np] = self.wmesh*self.pmesh**2/(ko**2 - self.pmesh**2)		# Gaussian integral

        #print('   G0 pole subtraction')
        G[self.Np]  = -np.sum( self.wmesh/(ko**2 - self.pmesh**2 ) )*ko**2 	# 'Principal value'
        G[self.Np] -= 1j*ko * (np.pi/2)

        #python vec[0:n] is the first n elements, i.e., 0,1,2,3,...,n-1
        G[self.Np+1:2*self.Np+2] = G[0:self.Np+1]
        return G*2*mu
    
    def setup_GV_kernel(self,channel,Vmtx,ko,mu):

        """
        Construct the GV kernel for the given channel, potential matrix, relative momentum ko, and reduced mass mu.

        Args:
            channel (list): The channel for which to construct the GV kernel. The channel is a list of blocks, where each block is a dictionary containing the quantum numbers of the partial-wave coupling.
            Vmtx (np.ndarray): The potential matrix for the given channel and momentum mesh.
            ko (float): The relative momentum.
            mu (float): The reduced mass.

        Returns:
            np.ndarray: The GV kernel for the given channel, potential matrix, relative momentum ko, and reduced mass mu.
        """
    
        Np = len(self.pmesh)
        nof_blocks = len(channel)
        Np_chn = int(np.sqrt(nof_blocks)*(self.Np+1))
        # Go-vector dim(u) = 2*len(p)+2
        G0 = self.setup_G0_vector(ko,mu)
        
        g = np.copy(G0[0:Np_chn])
        GV = np.zeros((len(g),len(g)),dtype=complex)
        
        for g_idx, g_elem in enumerate(g):
            GV[g_idx,:] = g_elem * Vmtx[g_idx,:]
            
        return GV

    def setup_VG_kernel(self,channel,Vmtx,ko,mu):

        """
        Construct the VG kernel for the given channel, potential matrix, relative momentum ko, and reduced mass mu.

        Args:
            channel (list): The channel for which to construct the VG kernel. The channel is a list of blocks, where each block is a dictionary containing the quantum numbers of the partial-wave coupling.
            Vmtx (np.ndarray): The potential matrix for the given channel and momentum mesh.
            ko (float): The relative momentum.
            mu (float): The reduced mass.

        Returns:
            np.ndarray: The VG kernel for the given channel, potential matrix, relative momentum ko, and reduced mass mu.
        """

        Np = len(self.pmesh)
        nof_blocks = len(channel)
        Np_chn = int(np.sqrt(nof_blocks)*(self.Np+1))
        
        # Go-vector dim(u) = 2*len(p)+2
        G0 = self.setup_G0_vector(ko,mu)
        g = np.copy(G0[0:Np_chn])
        VG = np.zeros((len(g),len(g)),dtype=complex)
        
        for g_idx, g_elem in enumerate(g):
            VG[:,g_idx] = g_elem * Vmtx[:,g_idx]
        
        return VG
    
    def solve_lippmann_schwinger(self,channel,Vmtx,ko,mu):

        """
        Solve the Lippmann-Schwinger equation for the given channel, potential matrix, relative momentum ko, and reduced mass mu.
        The Lippmann-Schwinger equation is solved by matrix inversion, using the GV and VG kernels.

        Args:
            channel (list): The channel for which to solve the Lippmann-Schwinger equation. The channel is a list of blocks, where each block is a dictionary containing the quantum numbers of the partial-wave coupling.
            Vmtx (np.ndarray): The potential matrix for the given channel and momentum mesh.
            ko (float): The relative momentum.
            mu (float): The reduced mass.

        Returns:
            np.ndarray: The solution to the Lippmann-Schwinger equation for the given channel, potential matrix, relative momentum ko, and reduced mass mu.
        """

        # matrix inversion:
        # T = V + VGT
        # (1-VG)T = V
        # T = (1-VG)^{-1}V
        
        VG = self.setup_VG_kernel(channel,Vmtx,ko,mu)
        VG = np.eye(VG.shape[0]) - VG
        # golden rule of linear algebra: avoid matrix inversion if you can
        #T = np.matmul(np.linalg.inv(VG),Vmtx)
        T = np.linalg.solve(VG,Vmtx)

        return T

    @staticmethod
    #a static method is bound to a class rather than the objects for that class
    def compute_phase_shifts(ko,mu,on_shell_T):

        """
        Compute the phase shifts for the given relative momentum ko, reduced mass mu, and on-shell T-matrix elements.
        The phase shifts are computed using the formulas for the Blatt-Biedenharn convention for coupled channels, and the standard formula for uncoupled channels.

        Args:
            ko (float): The relative momentum.
            mu (float): The reduced mass.
            on_shell_T (list): A list of on-shell T-matrix elements, where the first element is T11, the second element is T12, and the third element is T22 for coupled channels, and the first element is T11 for uncoupled channels.

        Returns:
            np.ndarray: The computed phase shifts for the given relative momentum ko, reduced mass mu, and on-shell T-matrix elements.
        """

        rad2deg = 180.0/np.pi
        
        fac  = np.pi*mu*ko
        
        if len(on_shell_T) == 3:
            
            T11 = on_shell_T[0]
            T12 = on_shell_T[1]
            T22 = on_shell_T[2]
        
            # Blatt-Biedenharn (BB) convention
            twoEpsilonJ_BB = np.arctan(2*T12/(T11-T22))	# mixing parameter
            delta_plus_BB  = -0.5*1j*np.log(1 - 1j*fac*(T11+T22) + 1j*fac*(2*T12)/np.sin(twoEpsilonJ_BB))
            delta_minus_BB = -0.5*1j*np.log(1 - 1j*fac*(T11+T22) - 1j*fac*(2*T12)/np.sin(twoEpsilonJ_BB))

            # this version has a numerical instability that I should fix.
            # Stapp convention (bar-phase shifts) in terms of Blatt-Biedenharn convention
            #twoEpsilonJ = np.arcsin(np.sin(twoEpsilonJ_BB)*np.sin(delta_minus_BB - delta_plus_BB))      # mixing parameter
            #delta_minus = 0.5*(delta_plus_BB + delta_minus_BB + np.arcsin(np.tan(twoEpsilonJ)/np.tan(twoEpsilonJ_BB)))
            #delta_plus  = 0.5*(delta_plus_BB + delta_minus_BB - np.arcsin(np.tan(twoEpsilonJ)/np.tan(twoEpsilonJ_BB)))
            #epsilon     = 0.5*twoEpsilonJ

            # numerially stable conversion
            cos2e = np.cos(twoEpsilonJ_BB/2)*np.cos(twoEpsilonJ_BB/2)
            cos_2dp = np.cos(2*delta_plus_BB)
            cos_2dm = np.cos(2*delta_minus_BB)
            sin_2dp = np.sin(2*delta_plus_BB)
            sin_2dm = np.sin(2*delta_minus_BB)
            
            aR = np.real(cos2e*cos_2dm + (1-cos2e)*cos_2dp)
            aI = np.real(cos2e*sin_2dm + (1-cos2e)*sin_2dp)
            delta_minus = 0.5*np.arctan2(aI,aR)

            aR = np.real(cos2e*cos_2dp + (1-cos2e)*cos_2dm)
            aI = np.real(cos2e*sin_2dp + (1-cos2e)*sin_2dm)
            delta_plus = 0.5*np.arctan2(aI,aR)

            tmp = 0.5*np.sin(twoEpsilonJ_BB)
            aR = tmp*(cos_2dm - cos_2dp)
            aI = tmp*(sin_2dm - sin_2dp)
            tmp = delta_plus + delta_minus
            epsilon = 0.5*np.arcsin(aI*np.cos(tmp) - aR*np.sin(tmp)) 
            
            if ko <150:
                if delta_minus*rad2deg<0:
                    delta_minus += np.pi
                    epsilon *= -1.0
            return [np.real(delta_minus*rad2deg), np.real(delta_plus*rad2deg), np.real(epsilon*rad2deg)]
        
        else:
            # uncoupled
            T = on_shell_T[0]
            Z = 1-fac*2j*T
            # S=exp(2i*delta)
            delta = (-0.5*1j)*np.log(Z)

            return np.real(delta*rad2deg)
   
    def compute_Tmtx(self, channels, verbose=False):
        """
        Compute the T-matrix for the given channels and the momentum mesh.
        The T-matrix is computed by solving the Lippmann-Schwinger equation for each channel, using the potential matrix and the G0 vector.

        Args:
            channels (list): The channels for which to compute the T-matrix. Each channel is a list of blocks, where each block is a dictionary containing the quantum numbers of the partial-wave coupling.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        """


        if verbose:
            print(f'computing T-matrices for')

        self.Tmtx = []
        self.phase_shifts = []

        for idx, channel in enumerate(channels):
            if verbose:
                print(f'channel = {channel}')

            phase_shifts_for_this_channel = []

            nof_blocks = len(channel)
                            
            for Tlab in self.Tlabs:

                if verbose:
                    print(f'Tlab = {Tlab} MeV')

                ko,mu= self.lab2rel(Tlab,channel[0]['tz'])
                Vmtx = self.setup_Vmtx(channel,ko)[0] # get only V, not the list of submatrices
                this_T = self.solve_lippmann_schwinger(channel,Vmtx,ko,mu)
                self.Tmtx.append(this_T)

                Np = this_T.shape[0]
                # extract the on-shell T elements
                if nof_blocks > 1:
                    #coupled
                    Np = int((Np-2)/2)
                    T11 = this_T[Np,Np]
                    T12 = this_T[2*Np+1,Np]
                    T22 = this_T[2*Np+1,2*Np+1]
                    on_shell_T = [T11,T12,T22]
                else:
                    # uncoupled
                    Np = Np-1
                    T11 = this_T[Np,Np]
                    on_shell_T = [T11]

                this_phase_shift = self.compute_phase_shifts(ko,mu,on_shell_T)
                phase_shifts_for_this_channel.append(this_phase_shift)

            self.phase_shifts.append(np.array(phase_shifts_for_this_channel))      

    def get_Vmtx_from_split(self, V_split, lec_vector):

        """
        Get the potential matrix for the given split potential terms and the lec vector.
        The potential matrix is constructed by summing the split potential terms weighted by the corresponding lec values in the lec vector.
        
        Args:
            V_split (list): A list of split potential terms, where each term is a potential matrix corresponding to a specific variation of the LECs.
            lec_vector (list): A list of LEC values corresponding to the split potential terms, where the first element is the constant part and the subsequent elements are the variations in the specified directions.

        Returns:
            np.ndarray: The computed potential matrix.
        """

        V = 0
        for idx, this_V in enumerate(V_split):

            V += lec_vector[idx]*this_V

        return V
            
    def model_order_reduction(self,channel,Tlab,directions,training_points,verbose=False):

        """
        Perform model order reduction for the given channel, laboratory kinetic energy Tlab, directions of variation for the low-energy constants (LECs), and training points for the LECs.
        The model order reduction is performed by splitting the potential matrix into a constant part and split parts corresponding to the variation of the LECs in the specified directions, and then constructing the emulator by computing the relevant kernels and solving the Lippmann-Schwinger equation for the training points.

        Args:
            channel (list): The channel for which to perform model order reduction. The channel is a list of blocks, where each block is a dictionary containing the quantum numbers of the partial-wave coupling.
            Tlab (float): The laboratory kinetic energy for which to perform model order reduction.
            directions (list): A list of directions of variation for the low-energy constants (LECs), where each direction is specified by the name of the LEC to vary.
            training_points (list): A list of training points for the LECs, where each training point is a list of LEC values corresponding to the specified directions.
            verbose (bool): Whether to print verbose output.

        Returns:
            function: The reduced model function.
        """

        if len(channel) > 1:
            exit('model order reduction: limited to one channel at the time')

        if len(directions) != len(training_points[0]):
            exit('model order reduction: need training in every direction')
            
        nof_blocks = len(channel[0])
        Np_chn = int(np.sqrt(nof_blocks)*(self.Np+1))
        Np = Np_chn-1
        if nof_blocks == 4:
            Np = int((Np_chn-2)/2)
                    
        if verbose:
            print(f'MOR:')
            print(f'channel          = {channel}')
            print(f'nof_blocks       = {nof_blocks}')
            print(f'Np_chn           = {Np_chn}')
            print(f'Tlab             = {Tlab}')
            print(f'directions       = {directions}')
            print(f'#training points = {len(training_points)}')
            print(f'{training_points}')

        # we use nof_blocks to identify coupled (=4) and uncoupled (=1) channels
                
        ko,mu= self.lab2rel(Tlab,channel[0][0]['tz'])
        
        # we construct a diagonal matrix from the vector G0
        G = np.diag(self.setup_G0_vector(ko,mu)[0:Np_chn])
        
        # set relevant lecs to zero to extract constant part
        for lec, value in self.lecs.items():
            if lec in directions:
                self.lecs[lec] = 0
                
        V_split = []
        V_split.append(self.setup_Vmtx(channel[0],ko)[0]) # get only V, not the list of submatrices

        # set lecs in each direction to 1.0 and get the split potential term
        for lec, value in self.lecs.items():
            if lec in directions:
                self.lecs[lec] = 1.0
                V_direction = self.setup_Vmtx(channel[0],ko)[0] # get only V, not the list of submatrices
                V_split.append(V_direction - V_split[0])
                self.lecs[lec] = 0.0
                    
        # we need GV, VG for each split V term
        GV_split = []
        VG_split = []
        GVG_split = []
        Vi_split_T11 = [] # the on-shell Vi-split terms
        Vi_split_T12 = [] # the on-shell Vi-split terms
        Vi_split_T22 = [] # the on-shell Vi-split terms
        for this_V in V_split:
            GV = self.setup_GV_kernel(channel[0],this_V,ko,mu)
            VG = self.setup_VG_kernel(channel[0],this_V,ko,mu)
            Vi_split_T11.append(this_V[Np,Np])
            if (nof_blocks == 4):
                Vi_split_T12.append(this_V[2*Np+1,Np])
                Vi_split_T22.append(this_V[2*Np+1,2*Np+1])
            GV_split.append(GV)
            VG_split.append(VG)
            GVG_split.append(GV@G)
                
        # loop over the training lec values and setup the relevant lec vector
        # to assemble the potential for the split terms
        Ti = []
        for training_point in training_points:
            lec_training_vector = []
            lec_training_vector.append(1.0)
            for lec, value in self.lecs.items():
                if lec in directions:
                    # training points and directions are identically ordered wrt lec names
                    lec_training_vector.append(training_point[directions.index(lec)])

            V = self.get_Vmtx_from_split(V_split,lec_training_vector)
            this_T = self.solve_lippmann_schwinger(channel[0],V,ko,mu)
            Ti.append(self.solve_lippmann_schwinger(channel[0],V,ko,mu))

        # construct the emulator
        mi_split_T11  = np.zeros((len(training_points)), dtype=object)
        Mij_split_T11 = np.zeros((len(training_points),len(training_points)), dtype=object)
        Mij_const_T11 = np.zeros((len(training_points),len(training_points)), dtype=object)

        if (nof_blocks == 4):
            mi_split_T12  = np.zeros((len(training_points)), dtype=object)
            mi_split_T22  = np.zeros((len(training_points)), dtype=object)
            Mij_split_T12 = np.zeros((len(training_points),len(training_points)), dtype=object)
            Mij_split_T22 = np.zeros((len(training_points),len(training_points)), dtype=object)
            Mij_const_T12 = np.zeros((len(training_points),len(training_points)), dtype=object)
            Mij_const_T22 = np.zeros((len(training_points),len(training_points)), dtype=object)
            
        for r,training_point in enumerate(training_points):
            mi_split_T11[r] = []
            if (nof_blocks == 4):
                mi_split_T12[r] = []
                mi_split_T22[r] = []
            for c,training_point in enumerate(training_points):
                Mij_split_T11[r,c] = []
                Mij_const_T11[r,c] = []
                if (nof_blocks == 4):
                    Mij_split_T12[r,c] = []
                    Mij_const_T12[r,c] = []
                    Mij_split_T22[r,c] = []
                    Mij_const_T22[r,c] = []
                    
        for split_idx, _ in enumerate(GV_split):
            for r,training_point in enumerate(training_points):
                this_mi_TGV = np.copy(Ti[r]@GV_split[split_idx])
                this_mi_VGT = np.copy(VG_split[split_idx]@Ti[r])
                mi_split_T11[r].append((this_mi_TGV + this_mi_VGT)[Np,Np])
                if (nof_blocks == 4):
                    mi_split_T12[r].append((this_mi_TGV + this_mi_VGT)[2*Np+1,Np])
                    mi_split_T22[r].append((this_mi_TGV + this_mi_VGT)[2*Np+1,2*Np+1])
                for c,training_point in enumerate(training_points):
                    this_TGVGT_rc = np.copy(-Ti[r]@GVG_split[split_idx]@Ti[c])
                    this_TGVGT_cr = np.copy(-Ti[c]@GVG_split[split_idx]@Ti[r])
                    Mij_split_T11[r,c].append((this_TGVGT_rc + this_TGVGT_cr)[Np,Np])
                    Mij_const_T11[r,c] = (Ti[c]@G@Ti[r] + Ti[r]@G@Ti[c])[Np,Np]
                    if (nof_blocks == 4):
                        Mij_split_T12[r,c].append((this_TGVGT_rc + this_TGVGT_cr)[2*Np+1,Np])
                        Mij_const_T12[r,c] = (Ti[c]@G@Ti[r] + Ti[r]@G@Ti[c])[2*Np+1,Np]
                        Mij_split_T22[r,c].append((this_TGVGT_rc + this_TGVGT_cr)[2*Np+1,2*Np+1])
                        Mij_const_T22[r,c] = (Ti[c]@G@Ti[r] + Ti[r]@G@Ti[c])[2*Np+1,2*Np+1]
                       
        def emulator(emulator_lecs,verbose=False):

            ntp = len(training_points)
            this_ko   = ko
            this_mu   = mu
            this_Tlab = Tlab
            this_channel = channel
            this_nof_blocks = nof_blocks
            
            if verbose:
                print(f'emulating for lecs = {emulator_lecs}')
            
            # construct m and M
            nof_split_terms = len(mi_split_T11[0])
            nof_emulator_lecs = len(emulator_lecs)
            
            m_i_T11 = np.zeros((ntp), dtype=complex)
            M_ij_T11 = np.zeros((ntp,ntp), dtype=complex)
            V_T11 = 0
            if this_nof_blocks==4:
                m_i_T12 = np.zeros((ntp), dtype=complex)
                M_ij_T12 = np.zeros((ntp,ntp), dtype=complex)
                V_T12 = 0
                m_i_T22 = np.zeros((ntp), dtype=complex)
                M_ij_T22 = np.zeros((ntp,ntp), dtype=complex)
                V_T22 = 0
                
            assert nof_split_terms == nof_emulator_lecs , f"{nof_split_terms} /= {nof_emulator_lecs}"
            for r in range(0,ntp):
                for idx,lec in enumerate(emulator_lecs):
                    m_i_T11[r]  += lec*mi_split_T11[r][idx]
                    if this_nof_blocks == 4:
                        m_i_T12[r]  += lec*mi_split_T12[r][idx]
                        m_i_T22[r]  += lec*mi_split_T22[r][idx]
                for c in range(0,ntp):
                    for idx,lec in enumerate(emulator_lecs):
                        M_ij_T11[r,c] += lec*Mij_split_T11[r,c][idx]
                        if this_nof_blocks == 4:
                            M_ij_T12[r,c] += lec*Mij_split_T12[r,c][idx]
                            M_ij_T22[r,c] += lec*Mij_split_T22[r,c][idx]
                            
                    M_ij_T11[r,c] += Mij_const_T11[r,c]
                    if this_nof_blocks == 4:
                        M_ij_T12[r,c] += Mij_const_T12[r,c]
                        M_ij_T22[r,c] += Mij_const_T22[r,c]
                        
            for idx,lec in enumerate(emulator_lecs):
                V_T11 += lec*Vi_split_T11[idx]
                if this_nof_blocks == 4:
                    V_T12 += lec*Vi_split_T12[idx]
                    V_T22 += lec*Vi_split_T22[idx]
                    
            T11 = V_T11 + 0.5*m_i_T11.T@np.linalg.inv(M_ij_T11)@m_i_T11
            if this_nof_blocks == 4:
                T12 = V_T12 + 0.5*m_i_T12.T@np.linalg.inv(M_ij_T12)@m_i_T12
                T22 = V_T22 + 0.5*m_i_T22.T@np.linalg.inv(M_ij_T22)@m_i_T22

            if this_nof_blocks == 4:
                return [T11,T12,T22],this_Tlab,this_ko,this_mu
            return [T11],this_Tlab,this_ko,this_mu

        return emulator
