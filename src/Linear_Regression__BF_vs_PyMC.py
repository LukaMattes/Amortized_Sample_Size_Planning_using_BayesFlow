# Make Keras use PyTorch
import os
os.environ["KERAS_BACKEND"] = "torch"
# Imports for BayesFlow
import bayesflow as bf
from keras.callbacks import EarlyStopping
# Imports for PyMC
import pymc as pm
# General imports
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
# Imports for logging and suppressing of excessive notifications
from contextlib import redirect_stderr
import logging
import time

# Cleaner logs for training etc.
logging.disable(logging.WARNING)
np.set_printoptions(suppress=True)

# Choose decision rule
RULE = "greater" # "lesser"
EFFECT_SIZE = 0.0 # Set threshold for "effect is significant" for "greater" or "lesser"
TARGET_WIDTH = 0.1 # Set target width of CI for "precision"
CI_SIZE = 0.95 # Size of credible interval which will be calculated for beta0 from its posterior distribution
low_ci = (100.0 - CI_SIZE * 100.0) / 2.0
high_ci = 100.0 - low_ci

# Set target power -> not yet used
TARGET_POWER = 0.8

# Define all different scenarios of linear regression
scenarios = [
    {
        "name": "1_base",
        "true_b1": 0.1, "true_b0": 0.0, "true_sigma": 0.1, # sampling prior
        "prior_b1_m": 0.1, "prior_b1_sd": 0.4, "prior_b0_m": 0.0, "prior_b0_sd": 0.25, "prior_sigma_sd": 0.125, # training prior
        "potential_sample_sizes": [20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
        "smallest_training_ss": 12, "largest_training_ss": 240, "smallest_x": 0, "largest_x": 1
    },
    {
        "name": "2_weak_signal",
        "true_b1": 0.025, "true_b0": 0.0, "true_sigma": 0.1,
        "prior_b1_m": 0.025, "prior_b1_sd": 0.4, "prior_b0_m": 0.0, "prior_b0_sd": 0.25, "prior_sigma_sd": 0.125,
        "potential_sample_sizes": [50, 150, 250, 350, 450, 550, 650, 750, 850, 950],
        "smallest_training_ss": 30, "largest_training_ss": 1140, "smallest_x": 0, "largest_x": 1
    },
    {
        "name": "3_uninformative",
        "true_b1": 0.1, "true_b0": 0.0, "true_sigma": 0.1,
        "prior_b1_m": 0.0, "prior_b1_sd": 1.0, "prior_b0_m": 0.0, "prior_b0_sd": 1.0, "prior_sigma_sd": 0.63,
        "potential_sample_sizes": [30, 60, 90, 120, 150, 180, 210, 240, 270, 300],
        "smallest_training_ss": 18, "largest_training_ss": 360, "smallest_x": 0, "largest_x": 1
    },
    {
        "name": "4_narrow_prior",
        "true_b1": 0.1, "true_b0": 0.0, "true_sigma": 0.1,
        "prior_b1_m": 0.1, "prior_b1_sd": 0.1, "prior_b0_m": 0.0, "prior_b0_sd": 0.25, "prior_sigma_sd": 0.125,
        "potential_sample_sizes": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
        "smallest_training_ss": 6, "largest_training_ss": 66, "smallest_x": 0, "largest_x": 1
    },
    {
        "name": "B_misspecified",
        "true_b1": 0.05, "true_b0": 0.0, "true_sigma": 0.3,
        "prior_b1_m": 0.5, "prior_b1_sd": 0.1, "prior_b0_m": 0.0, "prior_b0_sd": 1.0, "prior_sigma_sd": 0.38,
        "potential_sample_sizes": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "smallest_training_ss": 60, "largest_training_ss": 1200, "smallest_x": 0, "largest_x": 1
    },
]

# Number of simulations per sample size to determine that sample sizes power
NUMBER_OF_SIMS = 500

# BayesFlow
MODEL = "SSP_Bayesflow.keras"
NUM_SAMPLES = 1000 # Samples drawn from posterior distribution of beta1
SUMMARY_DIM = 16 # Output dimensions of summary network
TRANSFORM = "affine" # Transform method used by inference network
EPOCHS = 300 # Maximum training epochs
BATCH_SIZE = 64
NUM_BATCHES_PER_EPOCH = 100
MIN_IMPROVEMENT = 0.00 # Minimum improvement per epoch to continue training and not stop early
PATIENCE = 20 # Number of times no improvement should occur in order to not stop early

# PyMC
POSTERIOR_DRAWS = 4000 # Number of draws from posterior distribution of beta1
TUNING_STEPS = 500 # Number of tuning steps in markov chain before posterior draws are done
CHAINS = 4 # Number of chains
CORES = 4 # Number of cores used in parallel during MCMC process
TARGET_ACCEPT = 0.9 # Target accept limiter for MCMC random steps
MAX_RHAT = 1.01 # RHAT must be smaller than this for chain to be accepted
MIN_ESS = 1000 # Effective sample size of posterior draws must be larger than this
MAX_RETRIES = 5 # Retries done per dataset for markov chain if convergence statistics do not accept the result


if __name__ == "__main__":

    # Get option to decide in which mode to run the 
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=int, choices=[0, 1, 2], default=0,
                        help="0: BayesFlow and PyMC will be executed; 1: BayesFlow only; 2: PyMC only")
    args = parser.parse_args()

    # Loop over all given regression scenarios

    for sc in scenarios:

        model_scenario = MODEL.replace(".keras", f"_{sc['name']}.keras")

        if args.mode == 0 or args.mode == 1:

            ### BAYESFLOW SETUP ###

            # Generate dataset given parameters
            def likelihood(beta, sigma, N):
                x = np.random.uniform(sc["smallest_x"], sc["largest_x"], size=N)
                y = np.random.normal(beta[0] + beta[1] * x, sigma, size=N)
                x = x - 0.5 # Center x before inference is performed
                return dict(y=y, x=x)

            # Generate parameters from (fitting) priors
            def prior():
                beta = np.random.normal([sc["prior_b0_m"] , sc["prior_b1_m"]], [sc["prior_b0_sd"], sc["prior_b1_sd"]])
                sigma = np.abs(np.random.normal(loc=0.0, scale=sc["prior_sigma_sd"]))
                return dict(beta=beta, sigma=sigma)

            # Randomize N to train on datasets of different size
            def meta():
                N = np.random.randint(sc["smallest_training_ss"], sc["largest_training_ss"])
                return dict(N=N)

            # Simulate whole data generating process
            simulator = bf.simulators.make_simulator([prior, likelihood], meta_fn=meta)

            par_keys = ["beta", "sigma"]
            par_names = [r"$\beta_0$", r"$\beta_1$", r"$\sigma$"]

            # Adapter to transform datasets before passing them through the neural networks
            adapter = (
                bf.Adapter()
                .broadcast("N", to="x")
                .as_set(["x", "y"]) # Treat order of data points as invariant
                .convert_dtype("float64", "float32")
                .constrain("sigma", lower=0) # No negative variance, uses softplus transform to enforce this
                #.log("sigma") # Alternative transformation on sigma to enforce non-negativity
                .sqrt("N") # To stabilize numerical behavior for changes in N for larger N
                .concatenate(["beta", "sigma"], into="inference_variables")
                .concatenate(["x", "y"], into="summary_variables")
                .rename("N", "inference_conditions")
            )

            # Initialize neural networks
            # DeepSet: summarizes dataset to summary_dim many dimensions and treats order of data points as invariant
            summary_network = bf.networks.DeepSet(summary_dim=SUMMARY_DIM)
            # CouplingFlow: learns mapping from data to parameters and inverse
            inference_network = bf.networks.CouplingFlow(transform="affine")

            # Define the BayesFlow workflow
            workflow = bf.BasicWorkflow(
                simulator=simulator,
                adapter=adapter,
                inference_network=inference_network,
                summary_network=summary_network,
                standardize=["inference_variables", "summary_variables"],
                save_weights_only=False, # To enable further training afterwards by saving optimizer state etc.
                checkpoint_name="SSP_Bayesflow_CP",
                checkpoint_filepath="../Models/"
            )

            # BAYESFLOW TRAINING

            # Define early stopping on the loss for training phase
            early_stopping = EarlyStopping(
                monitor="loss",
                min_delta=MIN_IMPROVEMENT,
                patience=PATIENCE, # Number of epochs to wait for improvement
                mode="min", # Minimize the negative log-likelihood loss
                restore_best_weights=True
            )

            start = time.time()

            # Train the networks
            history = workflow.fit_online(epochs=EPOCHS, batch_size=BATCH_SIZE,
                                          num_batches_per_epoch=NUM_BATCHES_PER_EPOCH, keep_optimizer=True,
                                          callbacks=[early_stopping])
            
            filepath = Path("../Models") / model_scenario
            workflow.approximator.save(filepath=filepath)

            training_time = time.time() - start

            # BAYESFLOW MODEL INFO

            training_info_df = pd.DataFrame({
                "NameOfModel": [model_scenario.replace(".keras", "_BF")],
                "SmallestSS": [sc["smallest_training_ss"]],
                "LargestSS": [sc["largest_training_ss"]],
                "SummaryDim": [SUMMARY_DIM],
                "Transform": [TRANSFORM],
                "Epochs": [EPOCHS],
                "BatchSize": [BATCH_SIZE],
                "NumBatchesPerEpoch": [NUM_BATCHES_PER_EPOCH],
                "TrainingTime": [training_time]
            })

            out_p_i = 0

            while os.path.exists(f"../Models/model_info_{model_scenario.replace('.keras', '_BF')}_{out_p_i}.xlsx"):
                    out_p_i += 1

            training_info_df.to_excel(f"../Models/model_info_{model_scenario.replace('.keras', '_BF')}_{out_p_i}.xlsx", index=False)

            # BAYESFLOW SSD

            output_rows = [] # Append entries for output dataframe here

            for  ss in sc["potential_sample_sizes"]:
                start = time.time()

                # Returns the ground truth values for the parameters (sampling prior)
                def true_prior():
                    beta = np.array([sc["true_b0"], sc["true_b1"]])
                    sigma = np.float64(sc["true_sigma"])
                    return dict(beta=beta, sigma=sigma)

                # Create simulator for the "real" data
                val_simulator = bf.simulators.make_simulator([true_prior, likelihood], meta_fn=(lambda: dict(N=ss)))
                count_effect = 0
                count_precise = 0
                b1_precision = 0
                for _ in range(NUMBER_OF_SIMS):
                    true_sims = val_simulator.sample(1) # Generate one synthetic "real" dataset for sampling 
                    with open(os.devnull, "w") as f:
                        with redirect_stderr(f): # Cleaner logs
                            # Sample posterior draws from the synthetic dataset
                            post_draws = workflow.sample(conditions=true_sims, num_samples=NUM_SAMPLES)
                    beta_draws = post_draws.get("beta") # Get posterior draws for beta1
                    # Credible interval of beta1
                    beta1_ci_lower = np.percentile(beta_draws[0, :, 1], low_ci)
                    beta1_ci_upper = np.percentile(beta_draws[0, :, 1], high_ci)
                    
                    # Decide whether an effect was detected
                    if RULE == "greater":
                        if beta1_ci_lower > EFFECT_SIZE:
                            count_effect += 1

                    else:
                        if beta1_ci_upper < EFFECT_SIZE:
                            count_effect += 1

                    # Decide whether the estimate was precise enough
                    if beta1_ci_upper - beta1_ci_lower <= TARGET_WIDTH:
                        count_precise += 1

                    b1_precision += (beta1_ci_upper - beta1_ci_lower)

                duration_bf = time.time() - start

                power_bf = count_effect / NUMBER_OF_SIMS # Calculate directional power
                power_precision = count_precise / NUMBER_OF_SIMS # Calculate precision power
                average_precision = b1_precision / NUMBER_OF_SIMS # Record average precision

                print(f"Power for ss {ss} and true parameter {sc['true_b1']} is {power_bf} "
                      f"and took {duration_bf} s. Precision power is {power_precision}")

                # Add row to output with power and all covariates

                output_rows.append({
                    "NameOfModel": model_scenario.replace(".keras", "_BF"),
                    "Scenario": sc["name"],
                    "SampleSize": ss,
                    "Slope_POI": sc["true_b1"],
                    "Intercept": sc["true_b0"],
                    "Sigma": sc["true_sigma"],
                    "Smallest_x": sc["smallest_x"],
                    "Largest_x": sc["largest_x"],
                    "DecisionRule": RULE,
                    "TargetEffect": EFFECT_SIZE,
                    "TargetWidth": TARGET_WIDTH,
                    "CIsize": CI_SIZE,
                    "PriorBeta1Mean": sc["prior_b1_m"],
                    "PriorBeta1SD": sc["prior_b1_sd"],
                    "PriorBeta0Mean": sc["prior_b0_m"],
                    "PriorBeta0SD": sc["prior_b0_sd"],
                    "PriorSigmaSD": sc["prior_sigma_sd"],
                    "EstimatedPower": power_bf,
                    "PrecisionPower": power_precision,
                    "AveragePrecision": average_precision,
                    "NumSamples": NUM_SAMPLES,
                    "NumberOfSims": NUMBER_OF_SIMS,
                    "SamplingTime": duration_bf
                })

            # Output BayesFlow results for scenario
            output = pd.DataFrame(output_rows)

            out_p_i = 0

            while os.path.exists(f"../Results/output_{model_scenario.replace('.keras', '_BF')}_{out_p_i}.xlsx"):
                out_p_i += 1

            output.to_excel(f"../Results/output_{model_scenario.replace('.keras', '_BF')}_{out_p_i}.xlsx", index=False)

        if args.mode == 0 or args.mode == 2:

            ### PYMC SETUP ###

            # Define simulator (using sampling prior)
            def generate_data(ss, beta0, beta1, sigma):
                x = np.random.uniform(low=sc["smallest_x"], high=sc["largest_x"], size=ss)
                y = np.random.normal(beta0 + beta1 * x, sigma, size=ss)
                return x, y

            # PYMC INFO

            pymc_info_df = pd.DataFrame({
                "NameOfModel": [model_scenario.replace('.keras', '_PyMC')],
                "NumPosteriorDraws": [POSTERIOR_DRAWS],
                "NumTuningSteps": [TUNING_STEPS],
                "NumChains": [CHAINS],
                "NumCores": [CORES],
                "TargetAccept": [TARGET_ACCEPT],
                "MaxRHAT": [MAX_RHAT],
                "MinESS": [MIN_ESS]
            })

            out_p_i = 0

            while os.path.exists(f"../Models/model_info_{model_scenario.replace('.keras', '_PyMC')}_{out_p_i}.xlsx"):
                    out_p_i += 1

            pymc_info_df.to_excel(f"../Models/model_info_{model_scenario.replace('.keras', '_PyMC')}_{out_p_i}.xlsx", index=False)

            # PYMC SSD

            output_rows = [] # Append entries for output dataframe here

            for ss in sc["potential_sample_sizes"]:
                start = time.time()
                count_effect = 0
                count_precise = 0
                b1_precision = 0

                x_dummy, y_dummy = generate_data(ss, sc["true_b0"], sc["true_b1"], sc["true_sigma"])

                # Initialize mutable data structure using dummies
                with pm.Model() as model:

                    x_data = pm.Data("x_data", x_dummy)
                    y_data = pm.Data("y_data", y_dummy)

                    # Setup PyMC with (fitting) prior distributions of parameters
                    intercept = pm.Normal("intercept", mu=sc["prior_b0_m"], sigma=sc["prior_b0_sd"])
                    slope = pm.Normal("slope", mu=sc["prior_b1_m"], sigma=sc["prior_b1_sd"])
                    sigma_ = pm.HalfNormal("sigma", sigma=sc["prior_sigma_sd"])
                    mu = intercept + slope * x_data
                    pm.Normal("y_obs", mu=mu, sigma=sigma_, observed=y_data)

                for i in range(NUMBER_OF_SIMS):
                    # Generate one synthetic dataset the MCMC algorithm will be fitted to
                    x, y = generate_data(ss, sc["true_b0"], sc["true_b1"], sc["true_sigma"])

                    # Validation loop to only accept results of chains that converge
                    validated = False
                    retries = 0
                    while not validated and retries < MAX_RETRIES:
                        retries += 1
                        try:
                            with model:
                                pm.set_data({"x_data": x, "y_data": y})
                                with open(os.devnull, "w") as f:
                                    with redirect_stderr(f): # Cleaner logs
                                        # Sample from the Markov Chain
                                        idata = pm.sample(draws=POSTERIOR_DRAWS, tune=TUNING_STEPS, chains=CHAINS,
                                                          cores=CORES, target_accept=TARGET_ACCEPT, progressbar=False)

                            # Check convergence of chains
                            rhat = pm.rhat(idata)
                            ess = pm.ess(idata, method="bulk")
                            validated = True
                            if any(np.any(var.values >= MAX_RHAT) for var in rhat.data_vars.values()):
                                validated = False # At least one chain did not converge
                            if any(np.any(var.values <= MIN_ESS) for var in ess.data_vars.values()):
                                validated = False # Effective sample size of any chain was not large enough
                        except Exception as e:
                            validated = False

                    if retries == MAX_RETRIES:
                        print("Warning! Retries hit limit")

                    samples = idata.posterior["slope"].values.flatten() # Get posterior draws for beta1
                    lower, upper = np.percentile(samples, [low_ci, high_ci]) # Credible Interval of beta1

                    # Decide whether an effect was detected
                    if RULE == "greater":
                        if lower > EFFECT_SIZE:
                            count_effect += 1

                    elif RULE == "lesser":
                        if upper < EFFECT_SIZE:
                            count_effect += 1

                    # Decide whether the estimate was precise enough
                    if upper - lower <= TARGET_WIDTH:
                        count_precise += 1

                    b1_precision += (upper - lower)

                duration_pymc = time.time() - start

                power_pymc = count_effect / NUMBER_OF_SIMS # Calculate directional power
                power_precision = count_precise / NUMBER_OF_SIMS # Calculate precision power
                average_precision = b1_precision / NUMBER_OF_SIMS # Record average power

                print(f"Power for ss {ss} and true parameter {sc['true_b1']} is {power_pymc} "
                      f"and took {duration_pymc} s. Precision power is {power_precision}")

                # Add row to output with power and all covariates

                output_rows.append({
                    "NameOfModel": model_scenario.replace('.keras', '_PyMC'),
                    "Scenario": sc["name"],
                    "SampleSize": ss,
                    "Slope_POI": sc["true_b1"],
                    "Intercept": sc["true_b0"],
                    "Sigma": sc["true_sigma"],
                    "Smallest_x": sc["smallest_x"],
                    "Largest_x": sc["largest_x"],
                    "DecisionRule": RULE,
                    "TargetEffect": EFFECT_SIZE,
                    "TargetWidth": TARGET_WIDTH,
                    "CIsize": CI_SIZE,
                    "PriorBeta1Mean": sc["prior_b1_m"],
                    "PriorBeta1SD": sc["prior_b1_sd"],
                    "PriorBeta0Mean": sc["prior_b0_m"],
                    "PriorBeta0SD": sc["prior_b0_sd"],
                    "PriorSigmaSD": sc["prior_sigma_sd"],
                    "EstimatedPower": power_pymc,
                    "PrecisionPower": power_precision,
                    "AveragePrecision": average_precision,
                    "NumPosteriorDraws": POSTERIOR_DRAWS,
                    "NumberOfSims": NUMBER_OF_SIMS,
                    "SamplingTime": duration_pymc
                })

            # Output results for scenario with PyMC
            output = pd.DataFrame(output_rows)

            out_p_i = 0

            while os.path.exists(f"../Results/output_{model_scenario.replace('.keras', '_PyMC')}_{out_p_i}.xlsx"):
                out_p_i += 1

            output.to_excel(f"../Results/output_{model_scenario.replace('.keras', '_PyMC')}_{out_p_i}.xlsx", index=False)
