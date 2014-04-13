import matplotlib.pyplot as plt
from pp_plot import pp_plot

class GewekeInterface:
    """Abstract interface that implements the Geweke test.

    A class representing an MCMC model should inherit from this class and implement:
        1. sample_data
        2. sample_params_from_prior
        3. sample_params_from_posterior

    Implement these methods, and you can check correctness of your inference procedure.

    For info on Geweke test, see "Getting it Right" by John Geweke:
    http://qed.econ.queensu.ca/pub/faculty/ferrall/quant/papers/04_04_29_geweke.pdf.

    This code is based in part on Jonathan Malmaud's Geweke test implementation:
    https://github.com/malmaud/damastes/blob/master/inference/geweke.py
    """

    def __init__(self):
        self._generative_samples = []
        self._inferential_samples = []

    def sample_data(self, params):
        """Returns a sample of data from likelihood.

        data ~ P(data|params)

        Arguments:
            params -- A setting of tbe model's latent parameters.
            data_dims -- Dimensions of data to sample.
        """
        raise NotImplementedError

    def sample_params_from_prior(self):
        """Returns a sample of latent params from prior.

        params ~ P(params)
        """
        raise NotImplementedError

    def sample_params_from_posterior(self, data, prev_params):
        """Returns a setting of latent params from one sweep of posterior inference.
        This is also referred to as the MCMC transition kernel/operator.

        params ~ MCMCTransitionKernel(data, prev_params)

        Arguments:
            data -- A setting of the model's observed variables.
        """
        raise NotImplementedError

    def generative_joint_sampler(self):
        """Returns a sample of the joint dist via generative process.
            1. Sample latent params from the prior.
            2. Sample data from the likelihood (conditioned on sampled latent params).
        """
        params = self.sample_params_from_prior()
        data = self.sample_data(params)
        return data, params

    def inferential_joint_sampler(self, joint_sample=None):
        """Returns a sample of the joint dist via MCMC transition kernel.
            1. Starts with a sample of the joint dist (obs, latent).
            2. Update latent params with one sweep of posterior inference (via MCMC transition kernel).
            3. Resample data from likelihood (conditioned on updated latent params).

        Arguments:
            joint_sample -- The initial state from which a sweep of posterior inference begins.
                            If None, this method will get forward sample from the generative sampler. 
        """
        if joint_sample is None:
            joint_sample = self.generative_joint_sampler()

        data, prev_params = joint_sample
        params = self.sample_params_from_posterior(data, prev_params)
        data = self.sample_data(params)
        return data, params

    def geweke_test(self, num_prior, num_posterior, skip):
        generative_joint_samples = [self.generative_joint_sampler() for _ in xrange(num_prior)]
        inferential_joint_samples = []
        joint_sample = None
        for i in xrange(num_posterior):
            joint_sample = self.inferential_joint_sampler(joint_sample)
            if i % skip == 0:
                inferential_joint_samples.append(joint_sample)
        self._generative_samples.extend(generative_joint_samples)
        self._inferential_samples.extend(inferential_joint_samples)

    def hist_statistic(self, statistic):
        f_generative_samples = [statistic(s) for s in self._generative_samples]
        f_inferential_samples = [statistic(s) for s in self._inferential_samples]
        _, ax = plt.subplots(2, 1, sharex=True)
        plt.sca(ax[0])
        parms = dict(bins=20)
        plt.hist(f_generative_samples, **parms)
        plt.grid()
        plt.title("Generative")
        plt.sca(ax[1])
        plt.hist(f_inferential_samples, **parms)
        plt.grid()
        plt.title("Inferential")
        plt.show()

    def pp_plot_statistic(self, statistic):
        f_generative_samples = [statistic(s) for s in self._generative_samples]
        f_inferential_samples = [statistic(s) for s in self._inferential_samples]
        pp_plot(f_generative_samples, f_inferential_samples)

    def evaluate(self, funcs, evals):
        for func, eval in zip(funcs, evals):
            gener_values = [func(s) for s in self._generative_samples]
            infer_values = [func(s) for s in self._inferential_samples]
            eval(gener_values, infer_values)
