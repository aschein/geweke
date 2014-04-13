from geweke_test import GewekeInterface
import numpy as np
import numpy.random as rn

class GammaPoissonVecModel(GewekeInterface):
    def __init__(self, N, T, shape, scale):
        self.N = N
        self.T = T
        self._shape = shape
        self._scale = scale
        self._lambda_T = np.zeros(T)
        self._y_NT = np.zeros((N, T), dtype=int)
        GewekeInterface.__init__(self)

    def sample_data(self, params):
        self._y_NT[:, :] = rn.poisson(params.reshape((self.N, 1)), size=(self.N, self.T))
        return self._y_NT.copy()

    def sample_params_from_prior(self):
        self._lambda_T = rn.gamma(self._shape, self._scale, self.T)
        return self._lambda_T.copy()

    def sample_params_from_posterior(self, y_NT, prev_params):
        """Conjugate case so only data is needed."""
        posterior_shape = self._shape + y_NT.sum(axis=1)
        posterior_scale = 1.0/((1.0/self._scale) + self.T)
        self._lambda_T[:] = rn.gamma(posterior_shape, posterior_scale)
        return self._lambda_T.copy()

    def mean_param(self, joint_sample):
        params, data = joint_sample
        return np.mean(params)

    def var_param(self, joint_sample):
        params, data = joint_sample
        return np.var(params)

    def max_param(self, joint_sample):
        params, data = joint_sample
        return np.max(params)

    def min_param(self, joint_sample):
        params, data = joint_sample
        return np.abs(np.min(params))

    def mean_data(self, joint_sample):
        params, data = joint_sample
        return np.mean(data)

    def var_data(self, joint_sample):
        params, data = joint_sample
        return np.var(data)

if __name__ == "__main__":
    gpm = GammaPoissonVecModel(N=5, T=5, shape=5.0, scale=1.0)
    gpm.geweke_test(num_prior=1000, num_posterior=50000, skip=50)
    assert len(gpm._generative_samples) == len(gpm._inferential_samples)
    gpm.pp_plot_statistic(gpm.mean_param)
