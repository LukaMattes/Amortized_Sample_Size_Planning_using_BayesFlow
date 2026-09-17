# Amortized_Sample_Size_Planning_using_BayesFlow

This repository contains the code used for my work around amortized sample size planning (SSP) using BayesFlow (Kühmichel et al., 2026; Radev et al., 2020) and for benchmarking it against PyMC (Abril-Pla et al., 2023). Additionally, it contains the results ("Results/") obtained from performing SSP with this framework for four linear regression scenarios described in the top section of the file "src/Linear_Regression__BF_vs_PyMC.py".
All information concerning the trained networks can be found in "Models/", including the training times.
The code in "src/ValidateModel.py" generates diagnostic plots for the trained BayesFlow networks and executing "src/WorkingWithData.py" produces analytical figures of my findings.

An accompanying tutorial is presented, though this is still WIP.
