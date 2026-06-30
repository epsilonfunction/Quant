""" 
This generates Gaussian Brownian Motion (GBM) paths for stock prices. The GBM model is defined by the stochastic differential equation:

dS = S_0 * (mu * dt + sigma * dW)
    S_0 is initial price
    mu is drift
    sigma is volatility

"""

import numpy as np
import matplotlib.pyplot as plt


class GBM:
    def __init__(self, S0, mu, sigma):
        """
        Implmentation of Gaussian Brownian Motion

        Args:
            S0 (_type_): Initial Price
            mu (_type_): Drift Coefficient
            sigma (_type_): Volatility Coefficient
            
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
    
    def simulate (self, T, N, M):
        
        paths,timestep = self.generate_paths(T, N, M)
        self.plot_paths(paths, timestep)
        
    def generate_paths(self, T, N, M):
        """ 
        Generates SINGLE path for GBM

        Args:
            T (_type_): Time (in years)
            N (_type_): Number of steps
            M (_type_): Number of paths
        """

        dt = T / N
        timestep = np.linspace(0, T, N)
        dS = self.S0 * (
            self.mu * dt + 
            self.sigma * np.sqrt(dt) * np.random.randn(M, N))

        S = self.S0 + np.cumsum(dS, axis=1)
        return S,timestep
    
    def plot_paths (self, paths,timestep):
        """ 
        Plots the generated paths

        Args:
            paths (_type_): _description_
        """
        num_of_paths = paths.shape[0]
        for i in range(num_of_paths):
            plt.plot(timestep, paths[i, :])
        
        plt.xlabel("Time $t$", fontsize=14)
        plt.ylabel("Random Variable $S(t)$", fontsize=14)
        plt.title("GBM Paths", fontsize=14)
        axes = plt.gca()
        axes.set_xlim([0, timestep[-1]])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.show()
        
        return

if __name__ == "__main__":
    S0 = 100
    mu = 0.1
    sigma = 0.2
    T = 1
    N = 1000
    M = 5

    gbm = GBM(S0, mu, sigma)
    # paths, timestep = gbm.generate_paths(T, N, M)
    gbm.simulate(T,N,M)