import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random.uniform


class Gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def pdf(self, datapoint):
        return (1 / (self.sigma * (np.sqrt(2 * np.pi)))) * np.exp(
            -(pow(datapoint - self.mu, 2) / (2 * pow(self.sigma, 2))))


class GaussianMixtureModel:

    # Take initial guesses for the parameters
    def __init__(self, data):
        self.data = data
        self.N = data.shape[0]
        sigma = np.sum(np.square((np.mean(data) - np.std(data)))) / self.N
        mu = np.random.uniform(min(data), max(data), 2)
        self.y1 = Gaussian(mu[0], sigma)
        self.y2 = Gaussian(mu[1], sigma)
        self.mix = .5
        self.logLH = 0

    # Compute the responsibilities    
    def EStep(self):
        self.gamma = np.zeros([self.N, 1])
        for index, point in enumerate(self.data):
            phi1, phi2 = self.y1.pdf(point), self.y2.pdf(point)
            weight1, weight2 = (1 - self.mix) * phi1, self.mix * phi2
            self.logLH += np.log(weight1 + weight2)
            self.gamma[index] = weight2 / (weight1 - weight2)

    def MStep(self):
        self.y1.mu = np.sum((1 - self.gamma) * self.data) / np.sum(1 - self.gamma)
        self.y1.sigma = np.sum((1 - self.gamma) * np.square(self.data - self.y1.mu))
        self.y2.mu = np.sum(self.gamma * self.data) / np.sum(self.gamma)
        self.y2.sigma = np.sum(self.gamma * np.square(self.data - self.y2.mu)) / np.sum(self.gamma)
        self.mix = np.sum(self.gamma) / self.N
        print('y1mu', self.y1.mu)
        print('y1sigma', self.y1.sigma)
        print('y2mu', self.y2.mu)
        print('y2sigma', self.y2.sigma)
        print('mix', self.mix)

    def str(self):
        print('Gaussian y1.mu:%s, y1.sigma:%f, y2.mu:%f, y2.sigma:%f' % (self.y1.mu.astype(float),
                                                                         self.y1.sigma.astype(float),
                                                                         self.y2.mu.astype(float),
                                                                         self.y2.sigma.astype(float)))


df = pd.read_csv('2d-em.csv', header=None)
GMM = GaussianMixtureModel(df)
best_logLH = float('-inf')
best_GMM = None

for i in range(1, 10):
    try:
        GMM.EStep();
        GMM.MStep();
        # GMM.str()
        if GMM.logLH > best_logLH:
            best_logLH = GMM.logLH
            best_GMM = GMM
    except (ZeroDivisionError, ValueError, RuntimeWarning):
        pass
