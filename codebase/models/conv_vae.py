# Copyright (c) 2018 Rui Shu
import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F
import numpy as np

class ConvVAE(nn.Module):
    def __init__(self, nn='conv', name='vae', z_dim=2, k=1):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        self.k = k
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
        nn = getattr(nns, nn)
        self.enc = nn.ConvEncoder(self.z_dim)
        self.dec = nn.ConvDecoder(self.z_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

        # # Mixture of Gaussians prior
        # self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
        #                                 / np.sqrt(self.k * self.z_dim))
        # # Uniform weighting
        # self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        phi = self.enc.encode(x)
        z_hat = ut.sample_gaussian(*phi)

        kl = ut.kl_normal(*phi, *self.z_prior).mean()
        # prior = ut.gaussian_parameters(self.z_pre, dim=1)
        #
        # q = self.enc.encode(x)
        # z_hat = ut.sample_gaussian(*q)
        #
        # kl = ut.log_normal(z_hat, *q) - ut.log_normal_mixture(z_hat, *prior)
        # kl = kl.mean()

        rec = -ut.log_bernoulli_with_logits(x, self.dec.decode(z_hat)).mean()
        nelbo = kl+rec
        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################
        batch = x.size()[0]

        """
        sample iw z's
        for z_i in sample:
            find p(z_i, all x)
            find q(z_i, x)
        average
        """

        phi_m, phi_v = self.enc.encode(x) # (batch, z_dim)
        phi_m, phi_v = ut.duplicate(phi_m, iw), ut.duplicate(phi_v, iw) # (batch*iw, z_dim)
        x_iw = ut.duplicate(x, iw)

        z_hat = ut.sample_gaussian(phi_m, phi_v) # (batch*iw, z_dim)
        log_q_zx = ut.log_normal(z_hat, phi_m, phi_v) # (batch*iw)
        log_p_z = ut.log_normal(z_hat, *self.z_prior) # (batch*iw)
        log_p_xz = ut.log_bernoulli_with_logits(x_iw, self.dec.decode(z_hat)) # (batch*iw)

        f = lambda x: x.reshape(iw, batch).transpose(1,0)
        log_p_xz, log_q_zx, log_p_z = f(log_p_xz), f(log_q_zx), f(log_p_z)
        iwae = ut.log_mean_exp(log_p_xz-log_q_zx+log_p_z, -1)
        iwae = iwae.mean(0)

        niwae = -iwae

        kl = ut.log_mean_exp(log_q_zx-log_p_z, -1)
        kl = kl.mean(0)

        rec = ut.log_mean_exp(log_p_xz, -1)
        rec = -rec.mean(0)

        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))
        # m, v = ut.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        # idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        # m, v = m[idx], v[idx]
        # return ut.sample_gaussian(m, v)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return self.compute_sigmoid_given(z)
