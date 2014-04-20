"""Module for generating P-P plots.

For more on P-P plots see:
http://en.wikipedia.org/wiki/P-P_plot
"""

import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt

def pp_plot(F, G, title=None, xlabel=None, ylabel=None):
    """Generates a P-P plot for given functions F and G.

    Arguments:
        F -- List of samples from function F.
        G -- List of samples from function G.
    """
    F = np.array(F)
    G = np.array(G)
    F.sort()

    F_cdf = [np.mean(F < f) for f in F]
    G_cdf = [np.mean(G < f) for f in F]

    plt.plot(F_cdf, G_cdf, 'b*', lw=0.5)
    plt.plot([-0.1, 1.1],[-0.1, 1.1],'g--', lw=3)
    if title is None:
        title = 'Probability-Probability plot'
    plt.title(title)
    if xlabel is None:
        xlabel = 'CDF of F'
    plt.xlabel(xlabel)
    if ylabel is None:
        ylabel = 'CDF of G'
    plt.ylabel(ylabel)
    plt.show()


if __name__ == '__main__':
    F = rn.normal(10, size=1000)
    G = rn.normal(11, size=1000)
    pp_plot(F, G, title='Unidentical Normals')


