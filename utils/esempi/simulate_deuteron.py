import numpy as np
import utils.lib.nn_studio as nn_studio  # Modulo per la gestione delle matrici T e canali nucleari
import utils.lib.chiral_potential as chiral_potential  # Modulo per il potenziale chirale
from scipy import linalg  # Per l'algebra lineare (autovalori, autovettori)
from scipy.special import spherical_jn as jn  # Funzione di Bessel sferica (non usata qui)
import utils.lib.auxiliary as aux  # Funzioni ausiliarie (non usate direttamente qui)
import matplotlib.pyplot as plt  # Per eventuali plot (non usato qui)
import utils.lib.lec_values as lec_values  # Valori delle costanti di accoppiamento (LECs)


# Inizializza un oggetto per il calcolo delle matrici T e delle fasi di scattering
nn = nn_studio.nn_studio(jmin=0, jmax=1, tzmin=0, tzmax=0, Np=130, mesh_type='gauleg_finite')


# Inizializza il potenziale chirale a Leading Order (LO) con cutoff Lambda=500 MeV
potential = chiral_potential.two_nucleon_potential('LO', Lambda=500.0)


# Assegna il potenziale all'oggetto nn (analizzatore nucleone-nucleone)
nn.V = potential


# Assegna i valori delle costanti di accoppiamento a bassa energia (LECs) al potenziale tramite nn
input_lecs = lec_values.lo_lecs
nn.lecs = input_lecs.copy()


# Il canale del deuterone è un tripletto accoppiato e consiste in quattro possibili combinazioni di l (momento angolare orbitale)
# e ll (momento angolare orbitale accoppiato): S-S, S-D, D-S, D-D. La funzione lookup_channel_idx cerca il canale corretto
# dato l, ll, s (spin totale) e j (momento angolare totale). Restituisce il canale del deuterone se i parametri corrispondono.
_, deuteron_channel = nn.lookup_channel_idx(l=0, ll=2, s=1, j=1)
print(deuteron_channel)  # Stampa le informazioni sul canale deuterone trovato


# Calcola la massa ridotta (mu) per il sistema nucleone-nucleone
_, mu = nn.lab2rel(0, 0)


# Numero totale di punti di mesh (Np per ciascun blocco, 2 blocchi)
N = 2 * (nn.Np)


# Inizializza le matrici Hamiltoniana (H) e cinetica (T)
H = np.zeros((N, N))
T = np.zeros((N, N))


# Unisce i pesi e i punti di mesh per i due blocchi
ww = np.hstack((nn.wmesh, nn.wmesh))
pp = np.hstack((nn.pmesh, nn.pmesh))


# Costruisce la matrice del potenziale per il canale deuterone
V = nn.setup_Vmtx(deuteron_channel[0])[0]


# Costruisce le matrici T (cinetica) e V (potenziale) e l'Hamiltoniana totale H = T + V
for i, p_bra in enumerate(pp):
    for j, p_ket in enumerate(pp):
        Tij = 0
        # Solo i termini diagonali della matrice T sono diversi da zero
        if i == j:
            Tij = p_bra ** 2 / (2 * mu)
            T[i][j] = Tij
        # Normalizza il potenziale con i pesi di quadratura e i momenti
        V[i][j] = V[i][j] * p_bra * p_ket * np.sqrt(ww[i] * ww[j])

# Somma le matrici per ottenere l'Hamiltoniana totale
H = T + V


# Calcola autovalori e autovettori dell'Hamiltoniana (soluzione dell'equazione di Schrödinger)
eigvals, eigvecs = linalg.eigh(H)
s = np.argsort(eigvals)  # Ordina gli autovalori
E = eigvals[s[0]]  # Prende il più basso (stato fondamentale, energia del deuterone)

# Autovettore corrispondente all'autovalore più basso (funzione d'onda del deuterone)
psi_k = eigvecs[:, s[0]]

# Stampa l'energia del deuterone trovata
print(f'E = {E}')