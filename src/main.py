import os, sys
sys.path.append("./")

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax.numpy as jnp
import jax

import numpy as np

from model import *

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=10'
numpyro.set_host_device_count(10)

import pathlib
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('data', type=str)
parser.add_argument('K', type=int)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('-o', '--output', type=str, default="result")
parser.add_argument('-i', '--iter', type=int, default=10000)

args = parser.parse_args()
input_path = pathlib.Path(args.data)
if( not input_path.exists() ):
    print("[Error] 入力ファイルが見つかりませんでした．")
    sys.exit()

if __name__=="__main__":
    
    data = np.loadtxt(input_path, delimiter=",", encoding='utf-8-sig')
    x = data[:, 0]; y = data[:, 1]
    # x = (x - np.min(x)) / (np.max(x) - np.min(x))
    # mask = ( (x==0.0) | (x==1.0) )
    # x = x[~mask]; y = y[~mask]
    
    N = len(x)# data.shape[0]
    # phi = np.array([np.ones(N), 1.0-2.0*x]).T
    phi = np.array([x, np.ones(N)]).T

    rng_key= jax.random.PRNGKey(args.seed)

    # NUTSでMCMCを実行する
    kernel = DiscreteHMCGibbs( NUTS(model), modified=True )
    mcmc = MCMC(
        kernel, num_warmup=3*args.iter,
        num_samples=args.iter, num_chains=10, progress_bar=False)
    mcmc.run(
        rng_key=rng_key, K=args.K,
        x=x, phi=phi, y=y, fixed_sigma=-1.0
    )
    estimated_sigma = np.mean( mcmc.get_samples()['sigma'] )
    import copy
    lowest_temp_mcmc = copy.deepcopy(mcmc)
    
    T_wbic = np.log(N)
    wbic_sigma = T_wbic * estimated_sigma
    kernel = DiscreteHMCGibbs( NUTS(model), modified=True )
    mcmc = MCMC(
        kernel, num_warmup=3*args.iter,
        num_samples=args.iter, num_chains=10, progress_bar=False)
    mcmc.run(
        rng_key=rng_key, K=args.K,
        x=x, phi=phi, y=y, fixed_sigma=wbic_sigma
    )
    log_likelihood = numpyro.infer.log_likelihood(
                                model=model, posterior_samples=mcmc.get_samples(),
                                K=args.K, x=x, phi=phi, y=y, fixed_sigma=estimated_sigma)
    neg_log_likelihood = -1.0 * np.sum( log_likelihood['obs'], axis=1)
    wbic = np.mean( neg_log_likelihood  )
    print(f"{args.K} : {wbic:.5f}")
    
    import pickle
    output_path = pathlib.Path(args.output+f"_{args.K:03}.pkl")
    output = {
        "x" : x,
        "y" : y,
        "K" : args.K,
        "mcmc" : lowest_temp_mcmc,
        "wbic" : wbic
    }
    with open(output_path, 'wb') as f:    
        pickle.dump(output, f)
