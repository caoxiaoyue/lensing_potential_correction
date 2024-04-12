# potential_correction
This is a wrapper utility of PyAuolens, aiming at implementing the "gravitational imaging technique" (also known as the "potential correction method") described in the following papers:
1. https://arxiv.org/abs/astro-ph/0506629 
2. https://arxiv.org/abs/astro-ph/0501324
3. https://arxiv.org/abs/0804.2827
4. https://arxiv.org/abs/0805.0201


The traditional potential correction method requires manual twisting of the level of regularization for both lensing potential and source brightness. We found that if we do not set the regularization parameters for the lensing perturbation properly, we are unable to recover the subhalo signal with the potential correction method. The example here demonstrates this point.

![image](https://github.com/caoxiaoyue/potential_correction/raw/main/demo/install_guide/reg_effect.jpg)

For this mock lens data, a small SIS subhalo is added at the location of the star symbol. Iterative potential correction, with different level of regularization for the lensing pertubation (from $10^6$ to $10^9$), is applied to this mock data. It is evident that only the $10^7$ regularization strength can recover the subhalo signal relatively well.

We are currently developing a new framework to determine the regularization parameters for the lens and source simultaneously and objectively by optimizing the Bayesian evidence. This new framework is a further improvement of Vernardos et al. 2022 (https://arxiv.org/abs/2202.09378). 

This code is compatible with autolens==2023.10.23.3. To install the code, you can follow the bash script `install.sh`, which create a virtual environment and install all the required packages to run the potential correction code using anaconda.

If you find this work useful, please cite the Cao et al. 2024 in prep.