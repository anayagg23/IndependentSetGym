import numpy as np
import scipy.special as ss

p = 0.5 # gnp with p=0.5

def prob(n, k):
    return (1-(1-p)**(ss.binom(k,2)))**(ss.binom(n,k))

def prob_product(n, k):
    d = 1
    for i in range(1, k):
        d = d*(1-prob(n, i))
    return d

def compute(n):
    s = 0
    for i in range(1, n+1):
        s = s+prob(n,i)*prob_product(n,i)
    return s/n

print(compute(20))

