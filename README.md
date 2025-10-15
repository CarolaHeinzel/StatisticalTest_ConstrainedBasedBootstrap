This repository contains the code that corresponds to xxx.

We consider the Admixture Model, in which we aim to consider the ancestry of the individual from population $k$, called $q^0_k$. We test the hypothesis

$$    H_0: \max_{k = 1}^K q^0_k \geq \varepsilon \text{ vs. } H_1: \max_{k = 1}^K q^0_k < \varepsilon (1) $$

and 

$$    H_0: \max_{k = 1}^K q^0_k \leq \varepsilon \text{ vs. } H_1: \max_{k = 1}^K q^0_k > \varepsilon. (2)$$

It contains code to 

- apply the constrained based boostrap test for the Admixture Model to some data from the 1000 Genomes Project.
- example data from the 1000 Genomes Data to apply the constrained bootstrap test to.
- Code to evaluate the statisical test (1).
- Code to evaluate the statistical test (2).
