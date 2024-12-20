import os

import jax.numpy as jnp
import jax

import numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC, MixedHMC, DiscreteHMCGibbs

numpyro.set_platform("cpu")
numpyro.set_host_device_count(1)

import pathlib

@jax.jit
def Gaussian_function(x, mu, s):
    ss = s ** 2.0
    z = 1.0 / jnp.sqrt( 2.0 * jnp.pi * ss  )
    return z * jnp.exp( - ( x - mu )**2.0 / (2.0 * ss) )

@jax.jit
def calc_mixture_ratio(x, mu, s):
    vec_g_func = jax.vmap(
        Gaussian_function, in_axes=[None, 0, 0]
    )
    m = vec_g_func(x, mu, s)
    g = m / jnp.sum(m, axis=0)
    return g

@jax.jit
def calc_mixture_of_experts(phi, w):
    vec_model_func = jax.vmap(
        lambda phi, w: jnp.dot(phi, w), in_axes=[None, 0]
    )
    h = vec_model_func(phi, w)
    return h


def model(K, x, phi, y, fixed_sigma):
    
    M = phi.shape[1]
    w = numpyro.sample("w", dist.Normal(0.0, 1.0).expand([K, M]))
    
    r = numpyro.sample("r", dist.Dirichlet(concentration=jnp.ones(K)))
    mu = jnp.cumsum(r) - ( r / 2.0 )

    s = numpyro.sample("s", dist.HalfNormal(0.1).expand([K]))

    g = numpyro.deterministic("g", calc_mixture_ratio(x=x, mu=mu, s=s))
    h = numpyro.deterministic("h", calc_mixture_of_experts(phi=phi, w=w))
    f = numpyro.deterministic("f", jnp.sum( g*h , axis=0))
  
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    sigma = jnp.where(fixed_sigma <= 0.0, sigma, fixed_sigma)
    
    with numpyro.plate("N", len(y)):
        categorical_dist = dist.Categorical(probs=g.T)
        labels = numpyro.sample("labels", categorical_dist)
        output = h[labels, jnp.arange(len(y))]
        numpyro.sample("obs", dist.Normal(output, sigma), obs=y)
