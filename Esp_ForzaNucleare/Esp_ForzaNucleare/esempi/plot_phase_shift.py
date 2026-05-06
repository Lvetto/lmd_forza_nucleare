import sys
sys.path.append('./../lib')

import numpy as np
import nn_studio as nn_studio
import chiral_potential as chiral_potential
import matplotlib.pyplot as plt
import granada_phases as granada
import auxiliary as aux
import lec_values as lec_values

from scipy.optimize import least_squares

# initialize an object for computing T-matrices, phase shifts
# Np is the number of quadrature points used in the solution of the LS equation.
# the mesh_type defaults to 'gauleg_infinite' if nothing else. Use this for scattering.
# (use gauleg_finite for bound states).
nn = nn_studio.nn_studio(jmin=0,jmax=1,tzmin=0,tzmax=0,Np=30)

# define the lab neutron-proton kinetic energies that you want to analyze (denser for low T in this case)
nn.Tlabs = [1e-6] + [x/10 for x in np.arange(1,11,1)]+[x for x in np.arange(2,31,1)] + [x for x in np.arange(40,360,10)]
## you can inspect results for the channels <ll s j || l s j> in your basis
_,selected_channel = nn.lookup_channel_idx(l=0,ll=2,s=1,j=1)

#get the 'empirical' granada values for plotting
exp_phases = granada.delta_3S1
err_phases = granada.delta_3S1_errors

# initialize an object for the chiral interaction (isospin symmetric LO, NLO in WPC available)
potential_lo = chiral_potential.two_nucleon_potential('LO',Lambda=500.0)
potential_nlo = chiral_potential.two_nucleon_potential('NLO',Lambda=500.0)

# give the potential to the nn-analyzer
nn.V = potential_lo
# give the LECS to the potential (via the nn-analyzer)
nn.lecs = lec_values.lo_lecs
# solve the Lippmann-Schwinger equation and compute phase shifts (for selected_channel)
nn.compute_Tmtx(selected_channel,verbose=True)
delta_lo = nn.phase_shifts[0][:,0]

#now solve for NLO
nn.V = potential_nlo
nn.lecs = lec_values.nlo_lecs
nn.compute_Tmtx(selected_channel,verbose=True)
delta_nlo = nn.phase_shifts[0][:,0]

# plot the result
plt.errorbar(granada.Tlabs,exp_phases, yerr=err_phases,label = 'Granada PWA', color='black',ls='none',marker='o',markersize=5.)
plt.plot(nn.Tlabs,delta_lo, label = r'$\chi$EFT LO ($\Lambda=500$ MeV)',color='blue',alpha=0.8,lw=2)
plt.plot(nn.Tlabs,delta_nlo, label = r'$\chi$EFT NLO ($\Lambda=500$ MeV)',color='green',alpha=0.8,lw=2)
plt.xlabel(r'$T_\mathrm{Lab}$ (MeV)')
plt.ylabel(r'phase shif (deg)')
plt.title(r'$^3$s$_1$')
plt.legend()
plt.savefig('lo_nlo_phases.pdf')
plt.show()


