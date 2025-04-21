# Potential Correction
This is a wrapper utility for **PyAutoLens** that implements the *gravitational‑imaging technique* (also known as the *potential correction method*). The algorithm recovers mass perturbations that a smooth, parametric lens‑mass model cannot capture by applying small corrections to the lensing potential on a pixelised grid. The potential correction method is particularly valuable for quantifying systematic errors introduced by oversimplified parametric mass models in several lensing applications, including substructure detection and time‑delay cosmography.


Traditional implementations of potential correction require the practitioner to adjust manually the regularisation strength applied to both the lensing potential and the source brightness. The inverse problem is then solved iteratively, and inappropriate choices of these hyper‑parameters can lead to erroneous reconstructions. Vernardos et al. (2022) mitigated this limitation by determining the optimal regularisation levels for the lens and source objectively, through sampling the Bayesian evidence. Nevertheless, the regularisation schemes adopted by Vernardos et al. (2022) are optimised for extended perturbations—such as those described by a Gaussian random field—and are therefore less suitable for localised subhalo perturbers.


Building on several seminal studies [1–6], we develop an new open‑source potential correction package which offers several key advantages over existing codes:    
* It reconstructs the pixelised source and the perturbative lensing potential simultaneously, fully accounting for their covariance.  
* It selects the optimal regularisation parameters for both the pixelised source and the perturbative lensing potential, by maximising the Bayesian evidence.  
* It flexibly recovers localised as well as extended lensing potential perturbations, facilitated by Mat\'ern regularisation.
* The entire modelling workflow is fully automated and requires no manual fine‑tuning.  


Our potential correction code is compatible with `autolens` version 2024.1.27.4. To install the software, download the Bash script `install.sh` from this repository and execute it (`bash install.sh` in your Linux terminal). This script creates a dedicated Conda environment and installs all necessary dependencies. 


If you make use of this software, please cite Cao et al. (2025, in preparation). To reproduce the results and figures presented in Cao et al. (2025, in preparation), please refer to the scripts within this GitHub repository: https://github.com/caoxiaoyue/potential_correction_paper.

---
[1] https://arxiv.org/abs/astro-ph/0506629  
[2] https://arxiv.org/abs/astro-ph/0501324  
[3] https://arxiv.org/abs/0804.2827  
[4] https://arxiv.org/abs/0805.0201  
[5] https://arxiv.org/abs/1811.03627  
[6] https://arxiv.org/abs/2202.09378