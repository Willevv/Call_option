import numpy as np
import math
import matplotlib.pyplot as plt

"""Monte-Carlo to simulate (*)f = e^(-rT)E[max(S(T)-K,0)|S(0)= S_0]"""
""" Metohod
1. Simulate N samples of S(T)
2. calculate the expectation as the sum 1/N * sum (max(S(T)-K,0)|S(0)= S_0)
S(T) = exp((r-sigma^2/2)T + sigma*W(T))*S_0
"""

# Config
global T, S0, K, r, sigma, N
T = 1/2
S0 = 35
K = 35
r = 0.04
sigma = 0.2
N = 500000

W = np.random.normal(loc = 0, scale = np.sqrt(T), size = N) # N samples from W brownian process, vectorize computations for faster runtime 
S_T = np.exp((r-sigma**2/2)*T + sigma*W)*S0
f = np.exp(-r*T)*np.maximum(S_T - K, 0)
f = np.cumsum(f)
f = f/np.arange(1,len(f)+1) # vector of the values of f for increasing number of Samples. 

print(f[-1])
plt.plot(f) #Plotting the simulated price over increasing number of samples
plt.show()
