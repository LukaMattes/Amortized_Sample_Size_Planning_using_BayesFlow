import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import bayesflow as bf
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.disable(logging.WARNING)

MODEL = "SSP_Bayesflow_1_base.keras"

np.set_printoptions(suppress=True)

# Normal for slope (prior)
BETA1_mean = 0.1
BETA1_sd = 0.4
# Normal for intercept (prior)
BETA0_mean = 0.0
BETA0_sd = 0.25
# Half-Normal for variance (prior)
SIGMA_sigma = 0.125

np.set_printoptions(suppress=True)

def likelihood(beta, sigma, N):
    x = np.random.uniform(0, 1, size=N)
    y = np.random.normal(beta[0] + beta[1] * x, sigma, size=N)
    x = x - 0.5
    return dict(y=y, x=x)

def prior():
    beta = np.random.normal([BETA0_mean , BETA1_mean], [BETA0_sd, BETA1_sd])
    sigma = np.abs(np.random.normal(loc=0.0, scale=SIGMA_sigma))
    return dict(beta=beta, sigma=sigma)

def meta():
    N = np.random.randint(12, 240)
    return dict(N=N)

simulator = bf.simulators.make_simulator([prior, likelihood], meta_fn=meta)

par_keys = ["beta", "sigma"]
par_names = [r"$\beta_0$", r"$\beta_1$", r"$\sigma$"]

adapter = (
    bf.Adapter()
    .broadcast("N", to="x")
    .as_set(["x", "y"])
    .constrain("sigma", lower=0)
    #.log("sigma")
    .sqrt("N")
    .convert_dtype("float64", "float32")
    .concatenate(["beta", "sigma"], into="inference_variables")
    .concatenate(["x", "y"], into="summary_variables")
    .rename("N", "inference_conditions")
)

summary_network = bf.networks.DeepSet(summary_dim=16)
inference_network = bf.networks.CouplingFlow(transform="affine")

workflow = bf.BasicWorkflow(
    simulator=simulator,
    adapter=adapter,
    inference_network=inference_network,
    summary_network=summary_network,
    standardize=["inference_variables", "summary_variables"],
    save_weights_only=False
)

if __name__ == "__main__":

    workflow.approximator = keras.saving.load_model(f"../Models/{MODEL}")

    num_samples = 1000
    val_sims = simulator.sample(200)
    post_draws = workflow.sample(conditions=val_sims, num_samples=num_samples)

    f = bf.diagnostics.plots.calibration_ecdf(
        estimates=post_draws,
        targets=val_sims,
        variable_names=par_names,
        difference=True,
        rank_type="distance"
    )
    plt.savefig("calibration_ecdf")

    f = bf.diagnostics.plots.pairs_posterior(
        estimates=post_draws,
        targets=val_sims,
        dataset_id=0,
        variable_names=par_names
    )
    plt.savefig("pairs_posterior")

    f = bf.diagnostics.plots.recovery(
        estimates=post_draws,
        targets=val_sims,
        variable_names=par_names
    )
    plt.savefig("recovery")

    f = bf.diagnostics.plots.z_score_contraction(
        estimates=post_draws,
        targets=val_sims,
        variable_names=par_names
    )
    plt.savefig("z_score_contraction")
