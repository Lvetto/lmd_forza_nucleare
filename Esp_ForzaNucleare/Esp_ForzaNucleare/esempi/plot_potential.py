import sys
sys.path.append('./../lib')

import numpy as np
import nn_studio as nn_studio
import chiral_potential as chiral_potential
import matplotlib.pyplot as plt
import lec_values as lec_values

# initialize an object for computing T-matrices, phase shifts, 
nn = nn_studio.nn_studio(jmin=0,jmax=0,tzmin=0,tzmax=0,Np=100,mesh_type='gauleg_finite')

# initialize an object for the chiral interaction (isospin symmetric LO, NLO in WPC available)
potential = chiral_potential.two_nucleon_potential('NLO',Lambda=500.0)

# give the potential to the nn-analyzer
nn.V = potential

# give the LECS to the potential (via the nn-analyzer)
nn.lecs = lec_values.nlo_lecs

idx ,selected_channel = nn.lookup_channel_idx(l=0,ll=0,s=0,j=0)
_, potential_matrix = nn.setup_Vmtx(selected_channel[0])

# for plotting potential
mtx = potential_matrix[0]

pp, p = np.meshgrid(nn.pmesh, nn.pmesh)

z_min, z_max = -np.abs(mtx).max(), np.abs(mtx).max()

fig, ax = plt.subplots()
c = ax.pcolormesh(p, pp, mtx, cmap='RdBu', vmin=z_min, vmax=z_max)
fig.colorbar(c, ax=ax)
ax.set_xlabel(r'$p$ (MeV)')
ax.set_ylabel(r"$p'$ (MeV)")
print(f'you should find a plot of the NLO potential in potential_matrix.pdf')
plt.savefig('potential_matrix.pdf')
