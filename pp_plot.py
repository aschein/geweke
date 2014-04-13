"""Module for generating P-P plots.

For more on P-P plots see:
http://en.wikipedia.org/wiki/P-P_plot
"""

import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt

def pp_plot(F, G, num_samples=100):
    """Generates a P-P plot for given functions F and G.

    Arguments:
        F -- List of samples from function F.
        G -- List of samples from function G.
        num_samples -- Int for how many points to evaluate CDFs over.
    """
    lo = min(min(F), min(G))
    hi = max(max(F), max(G))

    step_size = (hi-lo)/float(num_samples)

    Z = np.arange(lo, hi, step=step_size)

    F_cdf = [np.mean(F < z) for z in Z]
    G_cdf = [np.mean(G < z) for z in Z]

    plt.plot(F_cdf, G_cdf, "b*", lw=0.5)
    plt.plot([-0.1, 1.1],[-0.1, 1.1],'g--', lw=3)
    plt.title("Probability-Probability plot")
    plt.xlabel("CDF of F")
    plt.ylabel("CDF of G")
    plt.show()

if __name__ == "__main__":
    F = rn.normal(10, size=1000)
    G = rn.normal(11, size=1000)
    pp_plot(F, G)


