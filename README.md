# Sequential Conditional Independence Testing with Machine Learning



Conditional Independence Testing is a ubiquitous problem in scientific discovery. The widely employed Model-X assumption shifts the modelling burden from  the dependency of the output given the inputs, to the dependencies within inputs. Log-optimal e-variables have been studied in this setting, but it remains unclear how to optimally include machine learning models in these tests. 
    In this work, we exploit the performance drop of a model when a given feature is removed to construct a coin-betting e-value for bounded losses and an exponential e-value that mimics density-based approaches.
    Surprisingly, in a misspecified setting, we show theoretically and experimentally that neither e-value uniformly outperforms the other. Finally, we provide actionable algorithms, including an antisymmetric kernel density estimation of the loss distribution.



## Antisymmetric e-variables

Make use of an antisymmetric function $g$ (i.e. $`g(a,b) = -g(b,a)`$) and an importance statistic $q$ which we take to be the loss used to train the model.

$1 + g(q_t,\tilde{q}_t).$

We consider:

- Coin betting e-variables

```math
E_{\mathrm{cb}}^\lambda
=
1 + \lambda \left(
-\, l\!\left(m(X, Z), Y\right)
+ \mathbb{E}_{\widetilde{X}\mid Z}\!\left[
l\!\left(m(\widetilde{X}, Z), Y\right)
\right]
\right).
```

- Sign based e-variables

```math
E_{\mathrm{sgn}}^\lambda
= \mathbb{E}_{\widetilde{X}\mid Z}\!\left[
1 + \lambda \text{ sign} \left(
-\, l\!\left(m(X, Z), Y\right)
+ 
l\!\left(m(\widetilde{X}, Z), Y\right)
\right) \right].
```

- Tanh e-variables

```math
E_{\mathrm{tanh}}^\lambda
=
\mathbb{E}_{\widetilde{X}\mid Z}\!\left[
1 + \lambda \tanh \left(
-\, l\!\left(m(X, Z), Y\right)
+
l\!\left(m(\widetilde{X}, Z), Y\right)
\right) \right].
```


- Exchangeability based e-variables:

```math
 E_\mathrm{KDE}:= \mathbb{E}_{\widetilde{X}\mid Z}\!\left[ 1+\frac{f_q(q_t, \tilde{q}_t)-f_q(\tilde{q}_t, q_t)}{f_q(q_t, \tilde{q}_t)+f_q(\tilde{q}_t, q_t)} \right].
 ```



## Exponential e-variables

Approximaties the Model-X GRO with pseudo-probabilities

```math
E_{\mathrm{exp}}^\eta
:=
\frac{
\exp\!\left(
-\eta\, l\!\left(m(X, Z), Y\right)
\right)
}{
\mathbb{E}_{\widetilde{X} \mid Z}\!\left[
\exp\!\left(
-\eta\, l\!\left(m(\widetilde{X}, Z), Y\right)
\right)
\right]
}.
```



## Antisymmetric vs Exponential e-values

When the density of $Y$ given $X$ and $Z$ is not in the exponentialted-loss form, the exponential e-variable is no longer optimal. We compare it experimentally to e-variables of the antisymmetric form and find settings where the latter ones dominate (on [simulated data](src/experiments/simulations_dirichlet.py) and [real-world data](src/experiments/toy_hiv_exp_n_train_800.ipynb) ).


## How to run the code

- src
    In this folder you can find the backend code for running experiments:
    - [Sampler.py](src/Sampler.py): Classes for samplers including the 'DefaultSampler', which automatically determines whether to sample a cathegorical or continuos variable. 
    - [ML_e_process.py](src/ML_e_process.py): Runs the betting strategies provided as input and computes prequentially the optimal hyperparameters. 
- HIV data: in the 'data' folder it is possible to find the HIV mutations dataset used to produce the plots in the paper. The notebook for producing the plot is [this](src\experiments\toy_hiv_exp_n_train_800.ipynb) 




This repository contains the code associated to https://icml.cc/virtual/2026/80061 . It is ongoing work.


