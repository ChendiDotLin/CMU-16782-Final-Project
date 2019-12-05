import numpy as np
import pdb
import scipy

class Policy:
    def __init__(self, nbActions):
        self.nbActions = nbActions
    def decision(self):
        pass
    def getReward(self,reward):
        pass

class policyUCB(Policy):
    def __init__(self, nbActions=2):
        self.nbActions = nbActions
        self.estimateSum = np.zeros(self.nbActions)
        self.counter = np.zeros(self.nbActions)
        self.steps = 0
        self.alpha = 1
        
    def reset(self):
        self.estimateSum = np.zeros(self.nbActions)
        self.counter = np.zeros(self.nbActions)
        self.steps = 0

    def decision(self):
        #try each bandit atleast once
        if self.steps < self.nbActions:
            self.action = self.steps
        #choose the action according to UCB
        else: 
            self.action = np.argmax(self.estimateSum/self.counter + np.sqrt(self.alpha*np.log(self.steps)/(2*self.counter)))
        return self.action
    
    def getReward(self,reward):
        self.steps += 1
        self.counter[self.action] += 1
        self.estimateSum[self.action] += reward
        return

class policyDTS(Policy):
    def __init__(self, nbActions=2, C=1):
        self.nbActions = nbActions
        self.alphas = np.ones(self.nbActions) #initial alphsa and betas in the original paper of DTS is 2, but in Max's paper are 1, need to discuss.
        self.betas = np.ones(self.nbActions)
        self.C = C #threshold,measure of how the algorithm reacts to non stationarity in the environment. needs to be tuned.

    def reset(self):
        self.alphas = np.ones(self.nbActions)
        self.betas = np.ones(self.nbActions)

    def decision(self):
        #sample from the beta distributions. thetas should be of the shape of nbActions
        thetas = np.random.beta(self.alphas, self.betas)
        self.action = np.argmax(thetas)
        return self.action

    def getReward(self,reward):
        self.alphas[self.action] += reward
        self.betas[self.action] += 1 - reward
        
        if ((self.alphas[self.action] + self.betas[self.action]) > self.C):
            self.alphas[self.action] *= self.C/(self.C + 1)
            self.betas[self.action] *= self.C/(self.C + 1)
