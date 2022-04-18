import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli

import math

def KL_with_bernoulli_prior(probs):
  entropy = Bernoulli(probs).entropy().sum(-1)
  KL = - entropy - torch.log(torch.tensor(0.5))* probs.shape[1]
  return KL

class BernoulliEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, batch):
        hid = F.relu(self.proj1(batch))
        return torch.sigmoid(self.output(hid))

class BernoulliPriorDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, batch):
        hid = F.relu(self.proj1(batch))
        output = self.output(hid)
        return output
    
    def sample_distributions(self, n_samples):
        probs0 = torch.ones(n_samples, self.input_dim)*1/2
        m = Bernoulli(probs = probs0)
        z = m.sample()
        return torch.sigmoid(self(z))

    def sample_images(self, n_samples, argmax=False):
        probs = self.sample_distributions(n_samples)
        x = torch.bernoulli(probs)
        if argmax == False :
            return x
        return probs > 0.5


def KL_with_gaussian_prior(mu, log_sigma_squared):
    s = torch.exp(0.5*log_sigma_squared)
    KL = -0.5 * torch.sum(1 + torch.log(s*s) - mu*mu - s*s , axis = 1)
    return KL


class GaussianEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.proj1 = nn.Linear(input_dim, hidden_dim)

        self.proj1Mu = nn.Linear(hidden_dim,hidden_dim)
        self.proj2Mu = nn.Linear(hidden_dim, output_dim)

        self.proj1Sigma = nn.Linear(hidden_dim, hidden_dim)
        self.proj2Sigma = nn.Linear(hidden_dim, output_dim)

        
    def forward(self, batch):
        hid = self.proj1(batch)

        hidMu1 = self.proj1Mu(F.relu(hid))
        outMu = self.proj2Mu(hidMu1)

        hidSigma1 = self.proj1Sigma(F.relu(hid))
        outSigma = self.proj2Sigma(hidSigma1)

        return (outMu,outSigma)

class GaussianPriorDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.proj1Mu = nn.Linear(hidden_dim,output_dim)
        
    def forward(self, batch):
        hid = F.relu(self.proj1(batch))
        return self.proj1Mu(hid)
    
    def sample_distributions(self, n_samples):
        z = torch.empty(n_samples, self.input_dim) 
        z = z.normal_()
        return torch.sigmoid(self(z))

    # sample images
    def sample_images(self, n_samples, argmax=False):
        probs = self.sample_distributions(n_samples)
        if argmax == False:
            return torch.bernoulli(probs)
        elif argmax == True:
            return 1*(probs >= 0.5)




