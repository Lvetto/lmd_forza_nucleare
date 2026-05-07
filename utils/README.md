# Forza Nucleare - Laboratorio Modelli e Dati (LMD)

Questo modulo contiene una libreria per il calcolo delle osservabili di scattering nucleone-nucleone (NN) e degli stati legati (deuterone) utilizzando la **Teoria di Campo Efficace Chirale ($\chi$EFT)**. 
Il codice permette di costruire matrici di potenziale fino al Next-to-Leading Order (NLO), risolvere l'equazione di Lippmann-Schwinger per la matrice T e calcolare i phase shift (sfasamenti) per diversi canali ad onda parziale.

*Codice originale basato sul lavoro di Andreas Ekström (2023).*

---

## 🗂️ Struttura del Pacchetto

La directory `utils` è divisa in due sezioni principali: il motore fisico/matematico (`lib`) e gli script di applicazione pratica (`esempi`).

### 1. Librerie Principali (`utils/lib/`)

Questa cartella contiene le fondamenta computazionali del progetto:

* **`chiral_potential.py`**: Definisce la classe `two_nucleon_potential`. È il cuore fisico del pacchetto. Costruisce il potenziale nucleare nello spazio dei momenti calcolando lo scambio di un pione (OPE), lo scambio di due pioni (TPE) e i termini di contatto, applicando la decomposizione in onde parziali e le funzioni di regolarizzazione (cutoff $\Lambda$). Gestisce sia il Leading Order (LO) che il Next-to-Leading Order (NLO).
* **`nn_studio.py`**: Definisce la classe `nn_studio`, il motore risolutivo. Imposta le mesh di integrazione (Gauss-Legendre), costruisce le matrici hamiltoniane e risolve l'equazione integrale di Lippmann-Schwinger per ricavare la matrice T di scattering. Contiene anche i metodi per derivare i phase shift (tramite convenzione Blatt-Biedenharn) e le routine per la *Model Order Reduction* (emulazione).
* **`constants.py`**: Contiene le costanti fisiche hard-coded necessarie per i calcoli, come le masse dei nucleoni ($M_p, M_n$), la massa del pione ($M_\pi$), le costanti di accoppiamento ($g_A, f_\pi$) e il fattore di conversione $\hbar c$.
* **`lec_values.py`**: Memorizza i set di *Low-Energy Constants (LECs)* pre-calibrati tramite inferenza bayesiana sui dati di Granada. Contiene i dizionari per i potenziali LO e NLO.
* **`granada_phases.py`**: Contiene i dati sperimentali e le incertezze per i phase shift estratti dalla *Granada Partial-Wave Analysis*, usati come *ground-truth* per validare i modelli (canali $^1S_0$, $^3S_1$, $^3D_1$, ecc.).
* **`auxiliary.py`**: Fornisce funzioni di utilità generica, come la stampa formattata di matrici 2D (`matprint`) e lo scaling lineare di intervalli.

### 2. Script di Esempio (`utils/esempi/`)

Gli script in questa cartella mostrano come istanziare le classi della libreria per risolvere problemi fisici reali. Sono il punto di partenza ideale per capire come utilizzare il codice.

* **`plot_phase_shift.py`**: Calcola gli sfasamenti per lo scattering n-p nel canale accoppiato $^3S_1$ (energie di laboratorio da 0 a 350 MeV) e confronta graficamente le predizioni dei modelli LO e NLO con i dati sperimentali di Granada.
* **`plot_potential.py`**: Genera una mappa di calore (heatmap) degli elementi di matrice del potenziale NLO nello spazio dei momenti ($p, p'$), salvando il risultato come PDF.
* **`simulate_deuteron.py`**: Ricerca gli stati legati del sistema nucleone-nucleone. Costruisce l'Hamiltoniana totale $H = T + V$ sul canale del deuterone ($^3S_1-^3D_1$) e la diagonalizza per trovare l'energia dello stato fondamentale.

---

## 🚀 Guida all'Utilizzo (Workflow Tipico)

Per eseguire una simulazione, la logica seguita dagli script di esempio è sempre strutturata in **4 passaggi fondamentali**:

**1. Inizializzazione dello spazio e dello "Studio":**
Si crea un'istanza di `nn_studio` definendo lo spazio dei momenti angolari ($J, T, S$), la discretizzazione e il tipo di mesh (infinita per scattering, finita per stati legati).

```python
import utils.lib.nn_studio as nn_studio
nn = nn_studio.nn_studio(jmin=0, jmax=1, tzmin=0, tzmax=0, Np=30)
```

**2. Creazione del Potenziale:**
Si istanzia il potenziale specificando l'ordine di troncamento dello sviluppo chirale ('LO' o 'NLO') e il parametro di cutoff $\Lambda$.

```python
import utils.lib.chiral_potential as cp
potential = cp.two_nucleon_potential('NLO', Lambda=500.0)
```

**3. Assegnazione di Potenziale e LECs:**
Il calcolatore `nn` deve essere associato al potenziale scelto e a un set di Costanti a Bassa Energia (LECs).

```python
import utils.lib.lec_values as lecs
nn.V = potential
nn.lecs = lecs.nlo_lecs
```

**4. Calcolo (Scattering o Stati Legati):**
Si cerca il canale quantistico di interesse e si lancia il calcolo desiderato (es. Matrice T o diagonalizzazione Hamiltoniana).

```python
# Cerca il canale desiderato (es: l=0, ll=2, s=1, j=1)
_, canali = nn.lookup_channel_idx(l=0, ll=2, s=1, j=1)

# Calcola T-matrix e phase shifts
nn.compute_Tmtx(canali)
```

---

## 📦 Dipendenze

Il progetto fa uso esclusivo di librerie scientifiche standard Python:

* `numpy` (operazioni vettoriali e matriciali, integrazione)
* `scipy` (polinomi di Legendre, algebra lineare per autovalori)
* `matplotlib` (per la generazione dei grafici negli script di esempio)
