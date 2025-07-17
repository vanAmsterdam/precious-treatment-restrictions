
"""
define probabilistic models
"""

import numpyro
from functools import partial
from numpyro import distributions as dist, sample, plate

num_countries = 9

def ymodels(age, dm, nihhs, mrsprestroke, sexmale, diagnstroke, earlyrest=None, deceased_or_mrsgeq3=None, sample_y=False):
    # sample global parameters
    b_intercept = sample("b_intercept", dist.Normal(0, 10))
    b_age = sample("b_age", dist.Normal(0, 10))
    b_dm = sample("b_dm", dist.Normal(0, 10))
    b_nihhs = sample("b_nihhs", dist.Normal(0, 10))
    # b_mrsprestroke = sample("b_mrsprestroke", dist.Normal(0, 10))
    b_sexmale = sample("b_sexmale", dist.Normal(0, 10))
    b_diagnstroke = sample("b_diagnstroke", dist.Normal(0, 10))

    with plate('mrsprestroke', 3):
        b_mrsprestroke = sample("b_mrsprestroke", dist.Normal(0, 10))
    # b_earlyrest = sample("b_earlyrest", dist.Normal(0, 5))

    with plate("obs", age.shape[0]):
        eta  = b_intercept + b_age * age 
        eta += b_dm * dm + b_nihhs * nihhs
        # eta += b_mrsprestroke * mrsprestroke
        b_mrsprestroke = b_mrsprestroke[0] * (mrsprestroke == 1) + b_mrsprestroke[1] * (mrsprestroke == 2) + b_mrsprestroke[2] * (mrsprestroke >= 3)
        eta += b_mrsprestroke
        eta += b_sexmale * sexmale + b_diagnstroke * diagnstroke

        # eta += b_earlyrest * earlyrest

        if sample_y:
            sample("obs_y_t0", dist.Bernoulli(logits=eta), obs=deceased_or_mrsgeq3)

        return eta

ymodel = partial(ymodels, sample_y=True)
xmodel = partial(ymodels, sample_y=False)

def intercept_model(xdata, country, earlyrest=None, deceased_or_mrsgeq3=None, country_intercept=True):
    """
    probabilistic model with country specific intercept term for treatment
    """
    # global parameters
    s_u = sample("s_u", dist.HalfNormal(2.5)) # variance for latent factors
    s_mu = sample("s_mu", dist.HalfNormal(2.5)) # variance for intercepts per country
    s_threshold = sample("s_threshold", dist.HalfNormal(2.5)) # variance for country specific thresholds  

    # intercept for treatment
    mu_treatment = sample('mu_treatment', dist.Normal(0, 5))
    # effect of treatment restriction
    b_earlyrest = sample('b_earlyrest', dist.Normal(0, 5))

    # sample countryspecific threshold from hyperprior
    with plate('countries', num_countries):
        country_mu = sample('country_mu', dist.Normal())
        country_mu *= s_mu
        country_threshold = sample('country_threshold', dist.Normal())
        country_threshold *= s_threshold

    # run forward threshold model
    eta_x = xmodel(**xdata)
    numpyro.deterministic("eta_x", eta_x)

    with plate("obs", country.shape[0]):
        # create a patient specific latent factor
        u = sample('u', dist.Normal())
        
        # add scale and country-specific intercept to latent factor
        u *= s_u
        if country_intercept:
            u += country_mu[country]

        # estimated probability of bad outcome without treatment restriction
        eta_y0 = eta_x + u

        # sample treatment based on y0_eta and country specific intercept
        # get patient specific threshold based on their country
        threshold = country_threshold[country]
        eta_treatment = mu_treatment + eta_y0 - threshold
        that = sample('that', dist.BernoulliLogits(eta_treatment), obs=earlyrest)

        # sample outcome
        eta_y = eta_y0 + that * b_earlyrest
        numpyro.deterministic("eta_y0", eta_y0)
        numpyro.deterministic("eta_y", eta_y)

        return_val = eta_y

        if deceased_or_mrsgeq3 is not None:
            return_val = sample('obs_y', dist.Bernoulli(logits=eta_y), obs=deceased_or_mrsgeq3)

        return return_val


def binary_u_model(xdata, country, earlyrest=None, deceased_or_mrsgeq3=None, country_intercept=True):
    """
    probabilistic model with country specific intercept term for treatment
    """
    # global parameters
    b_u = sample("b_u", dist.HalfNormal(2.5)) # Effect of latent factor
    s_mu = sample("s_mu", dist.HalfNormal(2.5)) # variance for intercepts per country
    s_threshold = sample("s_threshold", dist.HalfNormal(2.5)) # variance for country specific thresholds  

    # intercept for treatment
    mu_treatment = sample('mu_treatment', dist.Normal(0, 5))
    # effect of treatment restriction
    b_earlyrest = sample('b_earlyrest', dist.Normal(0, 5))

    # sample countryspecific threshold from hyperprior
    with plate('countries', num_countries):
        country_mu = sample('country_mu', dist.Normal())
        country_mu *= s_mu
        country_threshold = sample('country_threshold', dist.Normal())
        country_threshold *= s_threshold

    # run forward threshold model
    eta_x = xmodel(**xdata)
    numpyro.deterministic("eta_x", eta_x)

    with plate("obs", country.shape[0]):
        # create a patient specific latent factor
        u = sample('u', dist.BernoulliProbs(0.5))
        
        # add scale and country-specific intercept to latent factor
        if country_intercept:
            u += country_mu[country]

        # estimated probability of bad outcome without treatment restriction
        eta_y0 = eta_x + b_u * u

        # sample treatment based on y0_eta and country specific intercept
        # get patient specific threshold based on their country
        threshold = country_threshold[country]
        eta_treatment = mu_treatment + eta_y0 - threshold
        that = sample('that', dist.BernoulliLogits(eta_treatment), obs=earlyrest)

        # sample outcome
        eta_y = eta_y0 + that * b_earlyrest
        numpyro.deterministic("eta_y0", eta_y0)
        numpyro.deterministic("eta_y", eta_y)

        return_val = eta_y

        if deceased_or_mrsgeq3 is not None:
            return_val = sample('obs_y', dist.Bernoulli(logits=eta_y), obs=deceased_or_mrsgeq3)

        return return_val

