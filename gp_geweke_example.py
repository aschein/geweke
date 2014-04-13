from geweke_test import GewekeInterface
import numpy as np
import numpy.random as rn

class GammaPoissonModel(GewekeInterface):
    def __init__(self, N, shape, scale):
        self.N = N
        self._shape = shape
        self._scale = scale
        self._lambda = 0.0
        self._data = np.zeros(N, dtype=int)
        GewekeInterface.__init__(self)

    def sample_data(self, param):
        self._data[:] = rn.poisson(param, size=self.N)
        return self._data.copy()

    def sample_params_from_prior(self):
        self._lambda = rn.gamma(self._shape, self._scale)
        return self._lambda

    def sample_params_from_posterior(self, data, prev_params):
        """Conjugate case so only data is needed."""
        posterior_shape = self._shape + data.sum()
        posterior_scale = 1.0/((1.0/self._scale) + len(data))
        self._lambda = rn.gamma(posterior_shape, posterior_scale)
        return self._lambda

    def param_indicator(self, joint_sample):
        return joint_sample[1]

    def mean_data_minus_param(self, joint_sample):
        param, data = joint_sample
        return param - np.mean(data)

    def mean_data(self, joint_sample):
        return np.mean(joint_sample[0])

    def var_data(self, joint_sample):
        return np.var(joint_sample[0])

if __name__ == "__main__":
    gpm = GammaPoissonModel(N=5, shape=5.0, scale=1.0)
    gpm.geweke_test(num_prior=100000, num_posterior=2000000, skip=20)
    gpm.plot_statistic(gpm.param_indicator)
    gpm.plot_statistic(gpm.mean_data)
    gpm.plot_statistic(gpm.var_data)
