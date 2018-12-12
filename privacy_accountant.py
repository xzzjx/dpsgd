# coding: utf-8

import numpy as np 
import torch
import math

class PrivacyAccountant(object):
    '''
    zcdp acountant
    '''
    def __init__(self, delta=1e-5):
        self.p = 0.0
        # self.sigmas = []
        # self.qs = []
        self.min_alpha = 10000
        self.min_q = 0 # q with respect to min_alpha
        self.min_sigma = 0 # sigma with respect to min_sigma
        self.delta = delta
    
    def update(self, sigma, q):
        # if sigma not in self.sigmas:
        #     sigmas.append(sigma)
        #     # min_alpha = min(min_alpha, sigma**2*math.log(1/(q*sigma)))
        # if q not in self.qs:
        #     qs.append(q)
            # min_alpha = min(min_alpha, sigma**2*math.log(1/(q*sigma)))
        if self.min_alpha > sigma**2*math.log(1/(q*sigma)):
            self.min_alpha = sigma**2*math.log(1/(q*sigma))
            self.min_q = q
            self.min_sigma = sigma
        p_ = q**2 / sigma**2
        self.p += p_
    
    def dp(self):
        q = self.min_q
        sigma = self.min_sigma
        delta = self.delta
        try:
            threshold = 1 / math.exp(self.p * sigma**4 * (math.log(1/(q*sigma))**2))
        except OverflowError:
            threshold = 0
        if delta >= threshold:
            epsilon = self.p + 2 * math.sqrt(self.p * math.log(1/delta))
        else:
            tmp = sigma**2 * math.log(1/(q*sigma))
            epsilon = self.p * (tmp+1) - math.log(delta) / tmp
        return epsilon, delta