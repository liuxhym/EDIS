import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEencoder(nn.Module):
    def __init__(self, x_dim, hidden_dim, latent_dim):
        super(VAEencoder, self).__init__()
        self.mu = nn.Sequential(nn.Linear(x_dim, hidden_dim), 
                                nn.ReLU(), 
                                nn.Linear(hidden_dim, latent_dim))
        self.sigma = nn.Sequential(nn.Linear(x_dim, hidden_dim), 
                                   nn.ReLU(), 
                                   nn.Linear(hidden_dim, latent_dim))
    
    def forward(self, xs):
        cur_mu, cur_sigma = self.mu(xs), self.sigma(xs)
        return cur_mu, cur_sigma


class VAEdecoder(nn.Module):
    def __init__(self, x_dim, hidden_dim, latent_dim) -> None:
        super(VAEdecoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), 
                                     nn.ReLU(), 
                                     nn.Linear(hidden_dim, x_dim)) 

    def forward(self, zs):
        return self.decoder(zs)

class VAE(nn.Module):
    def __init__(self, x_dim=1, hidden_dim=128, latent_dim=10):
        super(VAE, self).__init__()
        self.x_dim=x_dim
        self.hidden_dim=hidden_dim
        self.latent_dim=latent_dim
        self.encoder = VAEencoder(x_dim=x_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim)
        self.decoder = VAEdecoder(x_dim=x_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim)

    def forward(self, xs):
        mu, sigma = self.encoder(xs)
        part = torch.exp(sigma / 2)
        sample = torch.randn_like(part)
        sample = (mu + sample * part)
        output = self.decoder(sample)
        return output, mu, sigma
    
class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.kldv = nn.CrossEntropyLoss()
        
    def recon_loss(self, x, x_):
        los = nn.MSELoss(reduction='sum')
        return los(x, x_)

    def forward(self, x, x_, mu, sigma):
        recon = self.recon_loss(x, x_)
        kldv = self.kldv(x, x_)
        return recon + kldv 
