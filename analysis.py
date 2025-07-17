# %%
import jax

jax.config.update("jax_platform_name", "cpu")
import numpyro
import arviz as az
import numpy as np
from jax import numpy as jnp
from jax import random
import pandas as pd
from pathlib import Path
from numpyro.infer.initialization import init_to_value
from numpyro.infer.mcmc import MCMC
from numpyro.infer.hmc import NUTS
from numpyro.diagnostics import print_summary
import matplotlib.pyplot as plt

from functools import partial


from precious.models import *
from precious.inference import run_posterior_predictive, calculate_marginal_difference


numpyro.set_host_device_count(12)
k = random.PRNGKey(0)
kmcmc, k = random.split(k)

# helper function to scale the features, returning also the means and stds for later use:
def scale(x):
    return (x - x.mean()) / x.std()

# %%
df = pd.read_csv(Path("data") / "df_curated.csv")
num_obs = df.shape[0]

x_vars = ["age", "dm", "nihhs", "mrsprestroke", "sexmale", "diagnstroke"]
x_orig = {k: np.array(df[k]) for k in x_vars}
# xdata = {k: np.array(df[k]) for k in x_vars}
xdata = {}
scale_vars = ["age", "nihhs"]
for k, v in x_orig.items():
    if k in scale_vars:
        xdata[k] = scale(v)
    else:
        xdata[k] = v

country = np.array(df["country"])
num_countries = jnp.unique(country).shape[0]
earlyrest = np.array(df["earlyrest"])
deceased_or_mrsgeq3 = np.array(df["deceased_or_mrsgeq3"])



# %%
kmcmc, k = random.split(kmcmc)
num_chains = 8

mcmc_intercept = MCMC(
    NUTS(intercept_model),
    num_warmup=250,
    num_samples=1500,
    num_chains=num_chains,
)


mcmc_intercept.run(
    kmcmc,
    xdata,
    country,
    earlyrest,
    deceased_or_mrsgeq3=deceased_or_mrsgeq3,
    country_intercept=True,
)

# prepare arviz data structure
local_vars = [
    "u",
    "y0_prob",
    "u_log_prob",
    "logit_tau",
    "yhat",
    "eta_x",
    "eta_y",
    "eta_y0",
]
coords = {"patient": range(num_obs), "country": range(num_countries)}
dims = {k: ["patient"] for k in local_vars}
dims.update({k: ["country"] for k in ["country_threshold", "country_mu"]})

azd_intercept = az.from_numpyro(mcmc_intercept, coords=coords, dims=dims)

# %%
post_samples = mcmc_intercept.get_samples(group_by_chain=True)
# all_vars = list(azd_intercept.posterior.data_vars)
all_vars = list(post_samples.keys())
global_vars = [k for k in all_vars if k not in local_vars]

azs = az.summary(azd_intercept, global_vars, hdi_prob=0.95)

azs.to_csv("parameters_intercept.csv", index_label="prm_name")
print(azs)

# %%
az.plot_pair(azd_intercept, var_names=["s_u", "b_earlyrest"])
plt.savefig("pairplot_su_bearlyrest.png")

# %%
# calculate marginal difference
samples_intercept = mcmc_intercept.get_samples(group_by_chain=False)

num_local_samples = 120

## first setup a pp model that doesnt require any arguments
ppmodel = partial(
    intercept_model,
    xdata,
    country,
    earlyrest=None,
    deceased_or_mrsgeq3=None,
    country_intercept=True,
)

post_samples = run_posterior_predictive(
    kmcmc, ppmodel, samples_intercept, global_vars, num_local_samples
)

# %%
p_y_diff, eta_y_diff = calculate_marginal_difference(post_samples)
print(az.hdi(np.array(p_y_diff), hdi_prob=0.95))
print(p_y_diff.mean())


# %%
# run model with binary confounder

# %%
kmcmc, k = random.split(kmcmc)
mcmc_binaryu = MCMC(
    NUTS(binary_u_model),
    num_warmup=250,
    num_samples=1500,
    num_chains=num_chains,
)


mcmc_binaryu.run(
    kmcmc,
    xdata,
    country,
    earlyrest,
    deceased_or_mrsgeq3=deceased_or_mrsgeq3,
    country_intercept=True,
)

binaryu_samples = mcmc_binaryu.get_samples(group_by_chain=True)
azd_binaryu = az.from_dict(binaryu_samples)
globals_u = [k for k in global_vars if k != "s_u"] + ["b_u"]
azs_binaryu = az.summary(
    azd_binaryu, var_names=globals_u, hdi_prob=0.95
)

azs_binaryu.to_csv("parameters_binaryu.csv", index_label="prm_name")
print("binary u model")
print(azs_binaryu)

# %%
az.plot_pair(azd_binaryu, var_names=["b_u", "b_earlyrest"])
plt.savefig("pairplot_bu_bearlyrest_binary_u.png")
# %%

print(az.hdi(np.exp(azd_binaryu.posterior["b_earlyrest"]), hdi_prob=0.95))
print(np.median(np.exp(azd_binaryu.posterior["b_earlyrest"])))

# %%
