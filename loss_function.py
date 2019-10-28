import torch
import torch.distributions as distr
import torch.nn as nn


class Tacotron2Loss(nn.Module):
    def __init__(self, kld_weight=1.0):
        super(Tacotron2Loss, self).__init__()
        self.kld_weight = kld_weight

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _, lat_mu, lat_logvar = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        kld_loss = torch.tensor(0).type(mel_loss.type()).to(mel_loss.device)
        if lat_mu is not None:
            q = distr.Normal(lat_mu, torch.exp(lat_logvar / 2.))
            p = distr.Normal(torch.zeros_like(lat_mu), torch.ones_like(lat_logvar))
            kld_loss = distr.kl.kl_divergence(q, p).mean()
        loss = mel_loss + gate_loss + kld_loss * self.kld_weight
        return loss, mel_loss, gate_loss, kld_loss
