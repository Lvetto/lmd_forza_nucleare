import sys
sys.path.append('./../lib')

import numpy as np
import nn_studio as nn_studio
import chiral_potential as chiral_potential
from scipy import linalg
from scipy.special import spherical_jn as jn
import auxiliary as aux
import matplotlib.pyplot as plt
import lec_values as lec_values

# initialize an object for computing T-matrices, phase shifts,
nn = nn_studio.nn_studio(jmin=0,jmax=1,tzmin=0,tzmax=0,Np=130,mesh_type='gauleg_finite')

# initialize an object for the chiral interaction (isospin symmetric LO, NLO in WPC available)
potential = chiral_potential.two_nucleon_potential('LO',Lambda=500.0)

# give the potential to the nn-analyzer
nn.V = potential

# give the LECS to the potential (via the nn-analyzer)
input_lecs = lec_values.lo_lecs
nn.lecs = input_lecs.copy()

# the deuteron channel is a coupled triplet and consists of four possible
# lket (ll) and lbra (l) combinations ('blocks'), i.e., S-S, S-D, D-S, D-D. This lookup function
# scans all the channels and if it can find a matching block it will return the
# entire channel to which the block belongs to. That way you will get the deuteron channel
# if you specify l=0,ll=2,s=1,j=1 or l=2,ll=2,s=1,j=1, or l=0,ll=0,s=1,j=1, or l=0,ll=0,s=1,j=1.
#
# print the all channels (a list of dictionaries), and it hopefully more clear. You can of course
# loop over all channels and pick the deuteron channel manually.
#
_,deuteron_channel = nn.lookup_channel_idx(l=0,ll=2,s=1,j=1)
print(deuteron_channel)

_,mu = nn.lab2rel(0,0)

N = 2*(nn.Np)

H = np.zeros((N,N))
T = np.zeros((N,N))

ww = np.hstack((nn.wmesh,nn.wmesh))
pp = np.hstack((nn.pmesh,nn.pmesh))

V = nn.setup_Vmtx(deuteron_channel[0])[0]

for i, p_bra in enumerate(pp):
    for j, p_ket in enumerate(pp):
        Tij = 0
        if i == j:
            Tij = p_bra**2/(2*mu)
            T[i][j] = Tij
            
        V[i][j] = V[i][j]*p_bra*p_ket*np.sqrt(ww[i]*ww[j])

        H = T+V

eigvals, eigvecs = linalg.eigh(H)
s = np.argsort(eigvals)
E = eigvals[s[0]]
psi_k = eigvecs[:,s[0]]

print(f'E = {E}')




