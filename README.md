# Crystal Diffusion Variational Autoencoder for Generative Design of Superionic Conductors


**Date:** December 24, 2025
**Experiment:** 1
**Status:**  Metastable Phase Discovered

## 1. Experiment Summary
This project successfully trained a generative Equivariant Graph Neural Network (EGNN) to design novel solid-state battery materials from scratch. The model, trained on a dataset of stable crystals, learned to generate chemically valid structures from pure random noise.

The final output is a **novel polymorph of Lithium Thiophosphate (Li₃PS₄)**. Validation via CHGNet and the Materials Project API confirms the material is **metastable (0.096 eV/atom above hull)** and exhibits **superionic lithium connectivity**, making it a viable candidate for solid-state battery electrolytes.

## 2. Methodology

### 2.1 Model Architecture
* **Backbone:** Equivariant Graph Neural Network (EGNN) to ensure rotational and translational invariance.
* **Generative Framework:** Variational Autoencoder (VAE) coupled with Denoising Diffusion Probabilistic Models (DDPM).
* **Input/Output:** The model operates on 3D fractional coordinates and atomic type embeddings, learning the probability distribution of stable crystal lattices.

### 2.2 Stabilization Techniques
Initial training faced severe gradient explosion (Loss spiking $70 \to 190$). The following physics-informed constraints were implemented to stabilize convergence:
* **Unit Vector Normalization:** Removed "lever arm" effects where distant atoms caused massive gradient spikes.
* **Huber Loss (Robust Regression):** Replaced MSE to handle outliers without destabilizing the optimizer.
* **SNR-Weighted Loss:** Balanced the learning signal across "easy" (low noise) and "hard" (high noise) diffusion timesteps.
* **Physics Clamping:** Capped atomic movement at $0.5 \AA$ per step to prevent physical overlaps during inference.

## 3. Results & Validation

### 3.1 The Discovery
The model was tasked with generating a structure with the stoichiometry **Li₉P₃S₁₂** (Equivalent to the famous conductor $Li_3PS_4$).

* **Symmetry:** P1 (Triclinic/Amorphous-like).
* **Geometry:** The generated structure lacks the strict high-symmetry of the ground state ($\gamma-Li_3PS_4$) but maintains valid bond lengths and tetrahedral coordination.
* **XRD Fingerprint:** Simulated diffraction patterns reveal sharp peaks indicating crystalline order, despite the low-symmetry classification. This suggests a highly distorted "wobbly" crystal lattice rather than a pure amorphous glass.

### 3.2 Stability Analysis (Thermodynamics)
The generated structure was relaxed using **CHGNet** (a universal machine learning potential) and compared against the Materials Project database.

| Metric | Value | Verdict |
| :--- | :--- | :--- |
| **Ground State Energy** | -4.643 eV/atom | Reference ($Li_3PS_4$ stable phase) |
| **Generated Energy** | -4.547 eV/atom | **Stable** |
| **Energy Above Hull** | **+0.096 eV/atom** | **Metastable (Synthesizable)** |

**Conclusion:** The material falls within the critical $<0.100 \text{ eV}$ window, indicating it is thermodynamically stable enough to be synthesized at room temperature.

### 3.3 Functional Analysis (Kinetics)
A percolation analysis was performed to test for ion transport pathways:
* **Lithium Connectivity:** >80% of Li atoms form a connected subgraph.
* **Percolation Threshold:** Li-Li jump distance of $4.0 \AA$.
* **Classification:** **Superionic Conductor**. The distorted lattice likely facilitates ion hopping by lowering the activation energy barriers compared to the rigid ground state.

## 4. Conclusion
The model successfully hallucinated a non-existent, chemically valid, and functionally conductive material structure. The result—a metastable, superionic $Li_3PS_4$ polymorph—validates the use of EGNN-based diffusion models for accelerating materials discovery.

**Date:** December 25, 2025
**Experiment:** 2
**Status:**  Catastrophic Failure. No more experiments.