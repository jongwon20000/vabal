import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class vabal_vae(nn.Module):
    def __init__(self, input_chs, encode_size, fc_size, class_latent_size, num_class):
        super(vabal_vae, self).__init__()
        
        self.num_class = num_class
        self.class_latent_size = class_latent_size
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

        # pre-processor
        self.pre_bns = nn.ModuleList([nn.BatchNorm1d(encode_size) for input_ch in input_chs])
        self.pre_fcs = nn.ModuleList([nn.Linear(input_ch, encode_size) for input_ch in input_chs])
        
        # encoder
        self.enc_fc11 = nn.Linear(encode_size*len(self.pre_fcs), fc_size)
        self.enc_bn11 = nn.BatchNorm1d(fc_size)
        
        self.enc_fc12 = nn.Linear(fc_size, fc_size)
        self.enc_bn12 = nn.BatchNorm1d(fc_size)
        
        self.enc_fc13 = nn.Linear(fc_size, fc_size)
        self.enc_bn13 = nn.BatchNorm1d(fc_size)
        
        self.enc_fc21 = nn.Linear(fc_size, class_latent_size*num_class)
        self.enc_fc22 = nn.Linear(fc_size, class_latent_size*num_class)        
        
        # decoder
        self.dec_fc11 = nn.Linear(class_latent_size*num_class, fc_size)
        self.dec_bn11 = nn.BatchNorm1d(fc_size)
        
        self.dec_fc12 = nn.Linear(fc_size, fc_size)
        self.dec_bn12 = nn.BatchNorm1d(fc_size)
        
        self.dec_fc13 = nn.Linear(fc_size, fc_size)
        self.dec_bn13 = nn.BatchNorm1d(fc_size)
        
        self.dec_fc2 = nn.Linear(fc_size, encode_size*len(self.pre_fcs))
        self.dec_bn2 = nn.BatchNorm1d(encode_size*len(self.pre_fcs))
                
        # Non-linear windows
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def preprocess(self, feats):
        # GAP
        h1_list = [F.avg_pool2d(feat, kernel_size=(feat.size(2), feat.size(3))).view(feat.size(0), -1) for feat in feats]
        
        # batch normalization
        h2_list = [self.relu(self.pre_fcs[i](h1)) for (i, h1) in enumerate(h1_list)]
        
        # fc layer (encoding)
        h3_list = [self.pre_bns[i](h2) for (i, h2) in enumerate(h2_list)]
        
        # concatenating...
        h3 = torch.cat(h3_list, dim=1)
        
        return h3
        #return self.sigmoid(h3)
        
    def encode(self, x):
        h11 = self.enc_bn11(self.relu(self.enc_fc11(x)))
        h12 = self.enc_bn12(self.relu(self.enc_fc12(h11)))
        h13 = self.enc_bn13(self.relu(self.enc_fc13(h12)))
        return self.enc_fc21(h13), self.enc_fc22(h13)

    
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    
    def decode(self, z):
        h11 = self.dec_bn11(self.relu(self.dec_fc11(z)))
        h12 = self.dec_bn12(self.relu(self.dec_fc12(h11)))
        h13 = self.dec_bn13(self.relu(self.dec_fc13(h12)))
        
        return self.dec_fc2(h13)

    
    def get_latent_var(self, feats):
        x = self.preprocess(feats)
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z

    
    def forward(self, feats, classes):
        x = self.preprocess(feats)
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        masked_z = z.view(-1, self.num_class, self.class_latent_size).clone()
        for i in range(z.size(0)):
            masked_z[i,classes[i],:] = 0        
        res = self.decode(masked_z.view(-1, self.num_class*self.class_latent_size))
        return x, z, res, mu, logvar

    
    def loss_function(self, recon_x, z, x, mu, logvar, classes):
        
        w = torch.sum(z.view(-1, self.num_class, self.class_latent_size)**2, axis=2)
        Loss_cls = self.criterion(-w, classes)
        BCE = torch.mean((recon_x - x.detach())**2)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.mean(KLD_element).mul_(-0.5)
        
        return (BCE, KLD, Loss_cls)
    
    
    def estimate_likelihood(self, feats):
        
        x = self.preprocess(feats)
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        z_reshape = z.view(-1, self.num_class, self.class_latent_size)
        
        likelihood = torch.zeros([x.size(0), self.num_class], dtype=torch.float64)
        
        for i in range(self.num_class):
            z_masked = z_reshape.clone()
            
            z_masked[:, i, :] = 0

            z_masked = z_masked.view(-1, self.num_class*self.class_latent_size)            
            res_masked = self.decode(z_masked)
            
            BCE_masked = torch.mean((res_masked - x)**2, dim=1)
            KLD_element_masked = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            KLD_masked = torch.mean(KLD_element_masked, dim=1).mul_(-0.5)
            
            likelihood[:, i] = BCE_masked + KLD_masked
            
        return likelihood, torch.sum(z_reshape**2, axis=2)
    