# Surface Analysis Toolkit: XPS & LEIS Integration GUIs

**Disclaimer:** *This repository provides independent, open-source analytical tools. It is not affiliated with, endorsed by, or associated with Casa Software Ltd. CasaXPS is a registered trademark of Casa Software Ltd.*

This repository contains two standalone Python Graphical User Interfaces (GUIs) designed to streamline the visualization, analysis, and data extraction of X-ray Photoelectron Spectroscopy (XPS) and Low-Energy Ion Scattering (LEIS) measurements. 

These tools are built using `PyQt5` and `matplotlib`, providing highly interactive, publication-ready plotting and rapid compositional analysis without relying exclusively on closed-source export pipelines.

## ⚙️ Requirements & Dependencies
To run these GUIs, ensure you have a standard Python 3 environment with the following packages installed. You can install them via your terminal:

    pip install PyQt5 numpy matplotlib scipy

---

## 🔬 Tool 1: XPS Analysis & Composition GUI (`XPS_GUI.py`)

### Overview
Commercial XPS software is powerful for spectral deconvolution but often restricts the bulk export of fitted data and parameters for external plotting. This GUI is specifically designed to bypass these workflow limitations. 

It reads standard text exports, allowing users to reconstruct and visualize the raw data, background (Shirley), envelope fit, and individual chemical state peaks in a fully customizable, colorblind-safe matplotlib canvas.

### Rapid Chemical Composition
Beyond plotting, the tool enables **rapid chemical composition analysis without requiring a full peak-by-peak fitting matrix**. By identifying core representative peaks and stripping out repeated/overlapping signals, the GUI calculates elemental atomic percentages (at%) using standard XPS quantification principles:

$$\text{at\%}_i = \frac{A_i / \text{ASF}_i}{\sum (A_n / \text{ASF}_n)} \times 100$$

Where $A$ is the integrated peak area and $\text{ASF}$ is the Atomic Sensitivity Factor.

### Key Features
* **Compatibility:** Imports standard tab/comma-separated spectra exported from commercial software.
* **Mathematical Fidelity:** The algorithms for Shirley background subtraction, Gaussian-Lorentzian (GL/SGL) lineshapes, and asymmetric FFT convolutions (LF) follow the established methodologies detailed in the CasaXPS Cookbook.
* **Interactive Data Canvas:** Pan, zoom, and visually isolate specific peaks or background models.
* **Composition Table:** Input Net Area and ASF values to instantly compute and update elemental atomic percentages.
* **Export:** Export visual plots as high-resolution images and compositional data directly to CSV.

---

## ☄️ Tool 2: LEIS Visualization Suite (`LEIS_analysis.py`)

### Overview
This module provides a streamlined, specialized interface for parsing and visualizing Low-Energy Ion Scattering (LEIS) data. 

Due to the highly surface-sensitive nature of LEIS, spectra require careful alignment and visualization to accurately interpret the outermost atomic layers. This GUI is specifically tailored for handling data related to **Tantalum (Ta), Niobium (Nb), Carbon (C), Nitrogen (N), and Oxygen (O)**.

### Key Features
* **VAMAS File Parsing:** Natively reads `.vms` (VAMAS) file formats, automatically correcting for interleaved counts and transmission data arrays to generate an accurate kinetic energy (eV) x-axis.
* **Visualization & Alignment:** Includes built-in controls for input angle adjustments and kinetic energy offsets to perfectly align spectra.
* **Quick Composition:** Offers a localized module for determining basic chemical composition ratios at different scattering angles and offsets. 
* **Data Export:** Aligned spectra and compositional fractions can be exported directly to CSV for downstream analysis.

---

## 📚 References & Acknowledgments
The mathematical frameworks for peak deconvolution, lineshape generation, and background subtraction utilized in the XPS GUI are based on standard surface analysis methodologies. For detailed algorithmic formulations, please refer to:
* **CasaXPS Manual & Cookbook:** [https://www.casaxps.com/manual.html](https://www.casaxps.com/manual.html)

**Note:** Both applications are built as modular object-oriented PyQt5 apps. You can launch either tool directly from your terminal or IDE by running `python XPS_GUI.py` or `python LEIS_analysis.py`.