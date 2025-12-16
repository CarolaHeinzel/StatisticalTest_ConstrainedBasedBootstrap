# Constrained Based Bootstrap Model in the Admixture Model

This repository contains the code that corresponds to xxx.

## Background
We consider the Admixture Model, in which we aim to consider the ancestry of the individual from population $k$, called $q^0_k$. We test the hypothesis

$$    H_0: \max_{k = 1}^K q^0_k \geq \varepsilon \text{ vs. } H_1: \max_{k = 1}^K q^0_k < \varepsilon (1) $$

and 

$$    H_0: \max_{k = 1}^K q^0_k \leq \varepsilon \text{ vs. } H_1: \max_{k = 1}^K q^0_k > \varepsilon. (2)$$

## Code

It contains
* Joint_max.py: Code that helps to calculate the test statistic.
* Max_CBB_implementation.py: Implementation of test (2).
* Max_CBB_implementation_swap.py: Implementation of test (1).
* Max_CBB_swap_application.py: Application of the test to real data.
* Plot_Results: Folder that contains the code to plot the results.
* Simulation_Results: Folder that contains the results of the simulation. The data is from the 1000 G Project and downloaded from the [GitHub Website of Peter Pfaffelhuber](https://github.com/pfaffelh/recent-admixture/blob/master/data/1000G/1000G_AIMsetKidd.vcf.gz).

### Funding Acknowledgement

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project-ID 499552394 – SFB 1597.
