"""
utilities for inference
"""

import jax

from jax.lax import scan
from jax import random

from numpyro import handlers
from numpyro.infer.mcmc import MCMC
from numpyro.infer.hmc import NUTS


def run_posterior_predictive(rng_key, model, samples, global_vars, num_local_samples = 100,
                             mcmc_kwargs = {'num_warmup': 100, 'num_samples': 300},
                             model_kwargs = {}):
    # thin the posterior samples, keeping only the global parameters
    thinned_globals = {k: v[::(v.shape[0] // num_local_samples)] for k, v in samples.items() if k in global_vars}
    print(f"shape of thinned posterior global parameter: {thinned_globals['b_intercept'].shape}")
    # print(thinned_globals.keys())
    # for k, v in thinned_globals.items():
        # print(k, v.shape)

    # setup a mcmc object to run posterior predictions in the model with these global parameters
    def pp_model(global_sample):
        # condition on the global parameters
        with handlers.condition(data=global_sample):
            return model(**model_kwargs)

    # update fixed mcmc kwargs
    mcmc_kwargs.update({
        'num_chains': 1, 'jit_model_args': True, 'progress_bar': False
    })

    mcmc= MCMC(NUTS(pp_model), **mcmc_kwargs)

    # setup a single run fucntion
    def run_mcmc(rng_key, global_sample):
        k_mcmc, k_carry = random.split(rng_key)
        mcmc.run(k_mcmc, global_sample)
        # print(global_sample.keys())
        # for k, v in global_sample.items():
            # print(k, v.shape)
        samples = mcmc.get_samples()
        # grab only the last sample
        # print(samples.keys())
        # for k, v in samples.items():
            # print(k, v.shape)
        local_smps = {k: v[-1] for k, v in samples.items()}
        return k_carry, global_sample | local_smps

    # run the mcmc with the thinned global parameters, using jax.lax.scan
    _, samples = scan(run_mcmc, rng_key, thinned_globals)
    
    return samples

def calculate_marginal_difference(posterior_samples):
    # grab linear predictor for do(t=0) and do(t=1)
    eta0 = posterior_samples['eta_y'][:,0,:]
    eta1 = posterior_samples['eta_y'][:,1,:]

    # calculate probabilities and differences
    p_y0 = jax.scipy.special.expit(eta0)
    p_y1 = jax.scipy.special.expit(eta1)
    p_y_diff = (p_y1 - p_y0)
    p_y_diff_mean = p_y_diff.mean(axis=-1)

    # calculate log odds ratio
    eta_diff_mean = jax.scipy.special.logit(p_y_diff_mean)
    
    return p_y_diff_mean, eta_diff_mean