import tensorflow as tf
import factor_analysis
import numpy as np

data = np.random.random((10,10))
covariance_prior = np.random.random((10,10))
means = np.random.random((10,10))

f = factor_analysis.factors.Factor(data, factor_analysis.posterior.Posterior(covariance_prior, means))

noise = factor_analysis.noise.Noise(f, f.posterior)

with tf.Session() as sess:
    print(f.create_factor().eval())
    print(noise.create_noise(f.create_factor()).eval())