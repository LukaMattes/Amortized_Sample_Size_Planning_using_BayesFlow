# Simulation-Based Bayesian Sample Size Planning with BayesFlow

### High-Level Pipeline

Before diving into the implementation details, it helps to understand the overall architecture of simulation-based sample size planning. The process follows a structured pipeline:

| Step | Description |
| :--- | :--- |
| 1. Define generative model | Formulate the likelihood mapping parameters to data |
| 2. Define priors | Establish baseline uncertainty for the parameters |
| 3. Simulate training data | Generate synthetic datasets with randomized sample sizes |
| 4. Train amortized posterior estimator | Let the neural network learn the inverse parameter mapping |
| 5. Generate validation datasets | Simulate data using fixed, true target parameters |
| 6. Draw posterior samples | Use the trained network for instant inference |
| 7. Compute Bayesian decision rule | Check for each validation dataset if effect is detected |
| 8. Estimate power across candidate N | Run a grid search to isolate the optimal sample size |

## Part I: Simulation-Based Power Analysis

### Introduction

Sometimes in empirical research, you need to design a study but are unsure about how many participants or trials you need to reliably detect an effect. In traditional statistics, we conduct a *power analysis* to determine that number. In a Bayesian system there is no true equivalent, but we can perform a *simulation-based power analysis*. Instead of relying on analytical shortcuts or crude approximations, we can simulate the entire data-generating mechanism under our model, apply our inference tools, and calculate how frequently our model correctly detects an effect.

In this tutorial, we will use `BayesFlow` which provides *Amortized Bayesian Inference (ABI)* via *neural posterior estimation*, *flow-based posterior approximators* and *simulation-based inference*.


Consider a standard problem where we want to estimate the slope parameter ($\beta_1$) of a continuous covariate $x$ on a target outcome variable $y$. When designing our study, a critical bottleneck is evaluating alternative sample sizes $N$. If our sample size is too small, our credible intervals will remain wide, and we will lack the power to rule out a zero effect. If it is too large, we waste resources.

To systematically study the impact of sample size, we can examine a sequence of prospective sample sizes $N$:

```python
# Define prospective sample sizes for design evaluation
POTENTIAL_SS = [100, 250, 500, 750, 1000]
```

For instance, at an initial target sample size of $N=100$, we expect data points to display substantial sampling variability. To find out how our architecture scales, we can systematically test sample sizes all the way up to $N=1000$.

### The Linear Data-Generating Mechanism

We assume that observations will follow a linear combination of an intercept ($\beta_0$) and a slope ($\beta_1$) parameter, plus some Gaussian noise ($\sigma$):

$$y_i \sim \mathcal{N}(\beta_0 + \beta_1 x_i, \sigma^2)$$

The following Python code defines this generative mechanism. The `likelihood` function takes parameter draws alongside a chosen sample size $N$, samples covariates $x$ uniformly between 0 and 1, and returns a synthetic data dictionary:

```python
import numpy as np

def likelihood(beta, sigma, N):
    # Sample covariate x uniformly
    x = np.random.uniform(0, 1, size=N)
    # Generate linear targets with Gaussian noise
    y = np.random.normal(beta[0] + beta[1] * x, sigma, size=N)
    return dict(y=y, x=x)
```

To complete our generative model, we must establish a `prior` distribution reflecting our initial expectations about parameter boundaries. We define a normal prior over both regression coefficients and a half-normal distribution over the noise standard deviation:

- Intercept Prior

$$
\beta_0 \sim \mathcal{N}(0,1)
$$

- Slope Prior

$$
\beta_1 \sim \mathcal{N}(0.1,0.25)
$$

- Noise Prior

$$
\sigma \sim |\mathcal{N}(0,0.38)|
$$

```python
# Prior parameters
BETA1_mean, BETA1_sd = 0.1, 0.25
BETA0_mean, BETA0_sd = 0, 1
SIGMA_sigma = 0.38

def prior():
    beta = np.random.normal([BETA0_mean, BETA1_mean], [BETA0_sd, BETA1_sd])
    sigma = np.abs(np.random.normal(loc=0.0, scale=SIGMA_sigma))
    return dict(beta=beta, sigma=sigma)
```

Crucially, because we want our model to adapt seamlessly across any incoming sample size $N$, we introduce a `meta` context function that randomizes the sample sizes during training. This enables the trained model to generalize across datasets of different sizes and learn sample size dependent posterior behavior. It is imperative that training sample sizes cover all candidate sample sizes as BayesFlow extrapolates badly. We can group these components into a modular `BayesFlow` simulator:

```python
import bayesflow as bf

def meta():
    # Vary trial/sample size dynamically across simulations
    N = np.random.randint(100, 1000)
    return dict(N=N)

# Build unified amortized simulator
simulator = bf.simulators.make_simulator([prior, likelihood], meta_fn=meta)
```

This completes the full simulation pipeline:

```text
Prior → Parameters → Likelihood → Simulated Dataset
```

---

## Part II: Amortized Bayesian Inference

### Constructing the BayesFlow Architecture

In traditional setups, we would run tedious Markov Chain Monte Carlo (MCMC) algorithms separately for every single dataset. If you have hundreds of synthetic datasets across multiple design choices, this becomes computationally impossible.

Instead, `BayesFlow` uses neural networks to learn an inverted model mapping directly from observed data to posterior distributions. Once trained, this network provides instantaneous posterior sampling – a paradigm known as *Amortized Bayesian Inference*.

Train once $$q_\phi(\theta \mid x) \approx p(\theta \mid x)$$ and reuse infinitely for future datasets.

In our case, we will use `BayesFlow` with `Keras` and a `pytorch` backend to run our neural network.

```python
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import bayesflow as bf
```

Our network needs to handle data matrices where the number of rows fluctuates according to $N$. We handle this by setting up an `Adapter` to reshape our inputs, a `SetTransformer` to compress arbitrary-length datasets into fixed-size summary representations, and a `CouplingFlow` network to warp simple distributions into accurate joint posteriors.

```python
import pandas as pd
import numpy as np

from pathlib import Path
from contextlib import redirect_stderr
import logging
import time

# Disable warnings for clean logs
logging.disable(logging.WARNING)
np.set_printoptions(suppress=True)

# 1. Prepare data formatting adapter
adapter = (
    bf.Adapter()
    .broadcast("N", to="x") # Associates the sample-size with each observation
    .as_set(["x", "y"]) # This allows permutation-invariant processing
    .constrain("sigma", lower=0) # Ensures posterior estimates obey positivity constraint on the variance
    .sqrt("N") # Uses the square root of N to stabilize scaling behavior
    .convert_dtype("float64", "float32")
    .concatenate(["beta", "sigma"], into="inference_variables") # Target inference vector
    .concatenate(["x", "y"], into="summary_variables") # The network receives paired observations as input summaries
    .rename("N", "inference_conditions")
)

# 2. Summary network produces fixed-dimensional representations of variable-sized datasets
summary_network = bf.networks.SetTransformer(summary_dim=64)

# 3. Inference network models complex joint posteriors via Normalizing Flows
inference_network = bf.networks.CouplingFlow(transform="affine")

# 4. Bind into one workflow
workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    inference_network=inference_network,
    summary_network=summary_network,
    standardize=["inference_variables", "summary_variables"],
    # this enables further training without losing the optimizer stat
    save_weights_only=False,
    checkpoint_name="Model_Name",
    checkpoint_filepath="../Models/"
)
```

### Online Network Training

We fit our network *online*. This means the neural network continuously requests batches of on-the-fly simulated data from our generative mechanism. It optimizes its weights over 40 training epochs until it can accurately approximate any posterior:

```python
MODEL = "Model_Name.keras"

start = time.time()

# Train the networks (summary and inference) online
history = workflow.fit_online(epochs=40, batch_size=64, num_batches_per_epoch=100, keep_optimizer=True)

# Save the amortized estimator
filepath = Path("../Models") / MODEL
workflow.approximator.save(filepath=filepath)

end = time.time()
print(f"finished training in {end - start} s")
```

Alternatively, we can use a pretrained model.

```python
workflow.approximator = keras.saving.load_model(
    f"../Models/Model_Name.keras"
)
```

### BayesFlow Model Validation & Calibration

The training finishes after 40 training epochs with a batch_size of 64 generated datasets and 100 batches per epoch (total data "seen" matters), because we programmed it as such.

In reality, knowing when to stop training can be tricky. `BayesFlow` calculates a training loss function after every training epoch. As a good rule of thumb, one can stop training the networks when the loss starts to stabilize over the duration of a few epochs. But for amortized inference with `BayesFlow`, validtion of the model is still recommended as neural networks are complex estimators.

To evaluate our trained model we take a look at the following diagnostic plots:

- **Parameter Recovery**: Checks if the posterior means closely track the true synthetic parameters used to generate the validation data.

- **Simulation-Based Calibration (SBC)**: Evaluates if the network's posteriors are probabilistically well-calibrated (i.e., neither overconfident nor underconfident). If calibrated, the rank histograms will appear uniform, and the empirical cumulative distribution function (ECDF) deviations will stay within the confidence boundaries.

- **Contraction Metrics**: Evaluates how much information the network extracts from the data relative to the prior distribution (e.g. how much narrower the credible intervals are compared to the priors) using Z-scores and posterior contraction.

The following script draws 500 validation datasets, samples from the trained network, and generates the diagnostic visualization suite:

```python
import matplotlib.pyplot as plt

# Define variable names for clean plotting labels
par_names = ["beta_0", "beta_1", "sigma"]
num_samples = 1000

# Generate validation datasets across varying sample sizes
val_sims = simulator.sample(500)

# Extract instantaneous posterior draws for all validation datasets
post_draws = workflow.sample(conditions=val_sims, num_samples=num_samples)

# --- Diagnostic Visualizations ---

# A. Bivariate Posterior Pairs Plot (Inspecting a single validation dataset)
f = bf.diagnostics.plots.pairs_posterior(
    estimates=post_draws,
    targets=val_sims,
    dataset_id=0,
    variable_names=par_names
)
plt.show()

# B. Parameter Recovery Plot (True vs. Estimated parameters)
f = bf.diagnostics.plots.recovery(
    estimates=post_draws,
    targets=val_sims,
    variable_names=par_names
)
plt.show()

# C. SBC Rank Histogram (Looking for uniform distributions)
f = bf.diagnostics.plots.calibration_histogram(
    estimates=post_draws,
    targets=val_sims,
    variable_names=par_names
)
plt.show()

# D. SBC ECDF Difference Plot (Verifying deviations remain within confidence bands)
f = bf.diagnostics.plots.calibration_ecdf(
    estimates=post_draws,
    targets=val_sims,
    variable_names=par_names,
    difference=True,
    rank_type="distance"
)
plt.show()

# E. Z-Score & Posterior Contraction (Quantifying uncertainty shrinkage over priors)
f = bf.diagnostics.plots.z_score_contraction(
    estimates=post_draws,
    targets=val_sims,
    variable_keys=["beta"],
    variable_names=par_names[0:2]
)
plt.show()
```

#### Interpreting Validation Outcomes

When analyzing the generated plots, look for these specific indicators of successful network training:

- **Recovery**: Points on the recovery plot should tightly cluster along the diagonal identity line, indicating that the network accurately uncovers the true generative parameters from data. -> Large $R^2$

- **Uniform Histograms & ECDF Bounds**: The rank histograms should look flat, and the ECDF difference curves must remain inside the gray shaded confidence bands. If the curves systematically break out of these boundaries, it indicates a structural miscalibration (e.g., severe under- or over-estimation of posterior variance), meaning the network requires more training epochs or a higher capacity architecture before moving on to design evaluation.

As we can see, the SBC plots for our inferred parameters show that the model underestimates the variance of $\beta_0$ and overestimates the error term $\sigma$. But since the plots for our parameter of interest (POI) $\beta_1$ look good we will carry on with this model.

---

## Part III: Power Calculation and Evalutation

### Quantifying Empirical Power

With our neural network trained, we can perform our simulation-based power analysis. For each hypothetical sample size $N$, we want to calculate the probability that the model passes a chosen Bayesian decision rule. In this case, we count it as a pass if our model detects a true positive effect ($\beta_1 > 0$) when the underlying effect is exactly $\beta_1 = 0.1$.

To decide this, we compute a 95% Bayesian Credible Interval for $\beta_1$ across 500 simulated studies. If the lower boundary of this interval is strictly greater than 0, we count it as a successful detection. Empirical power then is: $$\text{Power}(N)=P(CI_{95\%}(\beta_1)\text{ excludes }0)$$
In order to compute the credible intervals the invertible neural network draws from the posterior distribution it associates with a generated dataset:
$$
\theta^{(1)},\theta^{(2)},...,\theta^{(1000)} \sim p(\theta \mid x,y,N)
$$
This is called *Simulation-Based Inference (SBI)*.

The model can be evaluated for multiple effect-sizes of interest to answer the question "How many trials must we run to detect an effect size of at least ..."

This shows the benefit of using ABI.


```python
TRUE_PAR = [0.1, 0.15, 0.2]
BETA0 = 0
SIGMA = 0.3
NUMBER_OF_SIMS = 500
NUM_SAMPLES = 1000

for ss in POTENTIAL_SS:
    powers = []
    durations = []
    for tp in TRUE_PAR:
        start = time.time()
        
        # Generator anchored to fixed ground-truth parameters
        def fixed_true_prior():
            beta = np.array([BETA0, tp])
            sigma = np.float64(SIGMA)
            return dict(beta=beta, sigma=sigma)
            
        # Create new simulator which samples from the "true" distribution
        val_simulator = bf.simulators.make_simulator([fixed_true_prior, likelihood], meta_fn=(lambda: dict(N=ss)))
        count_positive_effect = 0
        
        # Loop over independent empirical realizations
        for _ in range(NUMBER_OF_SIMS):
            true_sims = val_simulator.sample(1)
            
            # Nearly instantaneous posterior sampling via the trained neural network
            # Redirect sampling notifications for clean sampling logs
            with open(os.devnull, "w") as f:
                with redirect_stderr(f):
                    post_draws = workflow.sample(conditions=true_sims, num_samples=NUM_SAMPLES)
            
            beta_draws = post_draws.get("beta")
            
            # Extract 95% equal-tailed Credible Interval for the slope parameter (index 1)
            beta1_ci_lower = np.percentile(beta_draws[0, :, 1], 2.5)
            beta1_ci_upper = np.percentile(beta_draws[0, :, 1], 97.5)
            
            # If the lower bound of the interval is greater than 0, a positive effect is reliably detected
            if beta1_ci_lower > 0:
                count_positive_effect += 1

        end = time.time()
        powers.append(count_positive_effect / NUMBER_OF_SIMS)
        durations.append(end - start)

        print(f"Power for ss {ss} and true parameter {tp} is {count_positive_effect / NUMBER_OF_SIMS} and took {end - start} s")

    # Export structured design metrics for later analysis
    output = pd.DataFrame({
        "NameOfModel": [MODEL] * len(TRUE_PAR),
        "SampleSize": [ss] * len(TRUE_PAR),
        "Slope_POI": TRUE_PAR,
        "Intercept": [2] * len(TRUE_PAR),
        "EstimatedPower": powers,
        "NumSamples": [NUM_SAMPLES] * len(TRUE_PAR),
        "NumberOfSims": [NUMBER_OF_SIMS] * len(TRUE_PAR),
        "SamplingTime": durations
    })

    output.to_excel(f"../Results/output_{MODEL.replace('.keras', '')}_{ss}.xlsx", index=False)
```

### Analyzing the Results

By comparing the output across alternative sample sizes, we can inspect how empirical power evolves:

* **Low Sample Sizes ($N = 100$ to $250$)**: Credible intervals remain wide because individual datasets contain high sampling noise. The lower boundary frequently falls below zero, resulting in lower power.
* **High Sample Sizes ($N \ge 800$)**: The large amount of data shrinks posterior uncertainty, pulling the credible intervals away from zero and driving empirical power upward.

#### Locating your Target Sample Size via Grid-Search

To choose the ideal sample size for your prospective study, evaluate the empirical powers generated by your `POTENTIAL_SS` grid against your target power threshold (e.g., 80% or $0.80$).

1. **Scan the Results**: Filter your generated outputs to isolate the sample sizes where the `EstimatedPower` matches or exceeds your target threshold (e.g., $\ge 0.80$).

2. **Select the Minimum Viable $N$**: Choose the smallest sample size from that group that satisfies the condition. For example, if $N=800$ yields $74.6\%$ power and $N=850$ yields $81.2\%$ power, $N=850$ is your chosen design size because it is the first grid step greater than (and closest to) your $80\%$ target.

3. **Refine**: If there is a massive gap between grid points (e.g., between 500 and 750), you can execute a secondary, narrower grid search (e.g., evaluating [550, 600, 650, 700]) using the exact same trained workflow without needing to re-train the neural network.

With this amortized simulation-based workflow, you can efficiently choose an optimized experimental sample size that hits your target power threshold.
