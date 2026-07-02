
"""
Principal Ideas Explorerd Here

1) Returns are log-normal 
    1.1) This is a consequence of compounding returns
    
    
2) Ln(S_t / S_0) := (mu - sigma**2/2) * t + sigma * W_t
    Therefore, St = S_0 * exp(A)
        where A = (mu - sigma**2/2) * t + sigma * W_t
        
    standard definitions apply
"""

import numpy as np
from scipy.stats import lognorm, norm


# def bsm_stddef:
    
def bs_call_stddef(S, r, sigma, T, K, q=0):
    """Well Known Analytical Solution

    Args:
        S (_type_): Underlying Price
        r (_type_): rfr
        sigma (_type_): volatility
        T (_type_): TTM
        K (_type_): Strike Price
        q (__float__, optional): continuous dividend yield.
    """
    d1 = 1/(sigma * np.sqrt(T)) * np.log(S/K) + (T*(r-q + 0.5*sigma**2))
                        
    # d1 /= sigma * np.sqrt(T)
    
    d2 = d1 - sigma * np.sqrt(T)
    
    return S * norm.cdf(d1) - K*np.exp(-r*T) * norm.cdf(d2)

if __name__ == "__main__":
    
    print(
        bs_call_stddef(
            S=100.0, 
            r=0.0, 
            sigma=0.2, 
            T=2, 
            K=105.0)
    )