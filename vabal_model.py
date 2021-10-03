import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from dataset_load import dataset_loader
from model_load import model_loader

from models.vae import vabal_vae

import os

class classification_modeler:
    
    
    def __init__(self, args, datapool_idx=[], sample_idx=[]):
        
        # training parameter loading...
        self.epoch = args.epoch
        self.vae_epoch = args.vae_epoch
        self.lr_change_epoch = args.lr_change_epoch
        self.vae_lr_change_epoch = args.vae_lr_change_epoch
        self.debug = args.debug
        self.sampling_num = args.sampling_num
        self.vae_lambda = args.vae_lambda
        
        # random seed arrange
        np.random.seed(args.rand_seed)
        torch.manual_seed(args.rand_seed)
        
        # dataset loading...
        (self.trainloader, self.valloader, self.testloader, self.classes) = dataset_loader(args.dataset_name, datapool_idx, sample_idx, args.batch_size, args.debug)
        
        # model loading...
        (net, input_chs) = model_loader(args.model_name)        
        net_vae = vabal_vae(input_chs, args.encode_size, args.fc_size, args.class_latent_size, len(self.classes))
        
        # CUDA & cudnn checking...
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = net.to(self.device)
        self.net_vae = net_vae.to(self.device)        
        if self.device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            self.net_vae = torch.nn.DataParallel(self.net_vae)
            cudnn.deterministic = True
            cudnn.benchmark = False
            
        # loss loading...
        self.criterion = nn.CrossEntropyLoss()
        
        # optimizer loading...
        self.init_optimizer = optim.SGD(net.parameters(), lr=args.init_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.last_optimizer = optim.SGD(net.parameters(), lr=args.last_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        
        self.init_vae_optimizer = optim.Adam(net_vae.parameters(), lr=args.vae_init_lr)
        self.last_vae_optimizer = optim.Adam(net_vae.parameters(), lr=args.vae_last_lr)
        
        
    def train(self, rands):
        self.train_net()
        if not rands:
            self.train_vae()
        
        
    def train_net(self):
        
        # Network initialization...
        self.net.train()
        optimizer = self.init_optimizer
        
        for i_epoch in range(self.epoch):
            
            # epoch initialization
            train_loss = 0
            correct = 0
            total = 0
            
            # lr change processing...
            if i_epoch == self.lr_change_epoch:
                optimizer = self.last_optimizer

            # batch iterations...
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                # set zero
                optimizer.zero_grad()
                
                # forward
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                
                # backword
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                # optimization
                optimizer.step()

                # logging...
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
            # Print Training Result
            if self.debug:
                print('%02d epoch finished >> Training Loss: %.3f | Acc: %.3f%% (%d/%d)' % (i_epoch+1, train_loss/(batch_idx+1), 100.*correct/total, correct, total))
                
        
    def train_vae(self):
        
        # Network initialization...
        self.net.eval()
        self.net_vae.train()
        optimizer = self.init_vae_optimizer
        
        for i_epoch in range(-1, self.vae_epoch):
            
            # epoch initialization
            train_BCE_loss = 0
            train_KLD_loss = 0
            train_class_loss = 0
            train_loss = 0
            
            # lr change processing...
            if i_epoch == self.vae_lr_change_epoch:
                optimizer = self.last_vae_optimizer
            
            # batch iterations... (for unlabelled data)
            for batch_idx, (inputs, _) in enumerate(self.valloader):
                # set zero
                optimizer.zero_grad()
                
                # forward (origin network)
                inputs = inputs.to(self.device)
                (feats, outputs) = self.net.module.extract_feature(inputs)
                _, predicted = outputs.max(1)
                
                # forward (vae module)
                x, z, recon_x, mu, logvar = self.net_vae(feats, predicted)
                
                # loss estimation
                (BCE_loss, KLD_loss, CLS_loss) = self.net_vae.module.loss_function(recon_x, z, x, mu, logvar, predicted)
                
                loss = BCE_loss + KLD_loss + self.vae_lambda*CLS_loss
                
                if i_epoch >= 0 :
                    # backward & optimization
                    loss.backward()                    
                    optimizer.step()

                # logging...
                train_BCE_loss += BCE_loss.item()
                train_KLD_loss += KLD_loss.item()
                train_class_loss += CLS_loss.item()
                train_loss += loss.item()
                
            # Print Training Result
            if self.debug:
                print('%02d epoch vae training finished >> VAE Training Loss: %.3f (BCE %.3f / KLD %.3f/ CLS %.3f)' % (i_epoch+1, train_loss/(batch_idx+1),  train_BCE_loss/(batch_idx+1),  train_KLD_loss/(batch_idx+1), train_class_loss/(batch_idx+1)))
        
        
    def val(self):
        
        # network initialization...
        self.net.eval()
        self.net_vae.eval()
        
        # logging initialization...
        num_classes = len(self.classes)
        prior_prob = torch.zeros(num_classes, dtype=torch.float64).to(self.device)
        labeling_error_prob_nom = torch.zeros(num_classes, dtype=torch.float64).to(self.device)
        labeling_error_prob_den = torch.zeros(num_classes, dtype=torch.float64).to(self.device)
        likelihood_prob = []
        
        if self.debug:
            correct = 0
            total = 0
            correct2 = 0
            total2 = 0
            labeling_error_prob_nom2 = torch.zeros(num_classes, dtype=torch.float64).to(self.device)
            labeling_error_prob_den2 = torch.zeros(num_classes, dtype=torch.float64).to(self.device)
        
        with torch.no_grad():
            # Likelihood estimation (On unlabelled pool)
            for batch_idx, (inputs, targets) in enumerate(self.valloader):
                # forward
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                (feats, _) = self.net.module.extract_feature(inputs)
                
                likelihood_stack = []                
                for sampling_idx in range(self.sampling_num):

                    # forward (vae module)
                    likelihood_temp, w = self.net_vae.module.estimate_likelihood(feats)
                    predicted_labels = torch.argmin(w, dim=1).float().to(self.device)
                    
                    if len(likelihood_stack) < 1:
                        #likelihood_stack = torch.exp(likelihood_temp)
                        likelihood_stack = likelihood_temp
                    else:
                        #likelihood_stack += torch.exp(likelihood_temp)
                        likelihood_stack += likelihood_temp

                    prior_prob += torch.histc(predicted_labels, bins=num_classes, min=0, max=num_classes)
                    
                    if self.debug:
                        total += predicted_labels.size(0)
                        correct += predicted_labels.eq(targets).sum().item()
                        
                        
                        labeling_error_prob_den2 += torch.histc(predicted_labels, bins=num_classes, min=0, \
                                                               max=num_classes).type(torch.cuda.FloatTensor) / 100.0
                        labeling_error_prob_nom2 += torch.histc(torch.where(predicted_labels.eq(targets), \
                                                                           predicted_labels, \
                                                                           torch.FloatTensor([-1]).to(self.device)), \
                                                               bins=num_classes, min=0, \
                                                               max=num_classes).type(torch.cuda.FloatTensor) / 100.0
                    
                likelihood_stack /= self.sampling_num                
                likelihood_prob += likelihood_stack.tolist()
                
            prior_prob /= torch.sum(prior_prob)
            prior_prob = prior_prob.cpu().numpy()
            
            # label noise estimation (On labelled pool)
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                # forward
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                (feats, _) = self.net.module.extract_feature(inputs)
                
                for sampling_idx in range(self.sampling_num):

                    # forward (vae module)
                    likelihood_temp, w = self.net_vae.module.estimate_likelihood(feats)
                    predicted_labels = torch.argmin(w, dim=1).float().to(self.device)

                    labeling_error_prob_den += torch.histc(predicted_labels, bins=num_classes, min=0, \
                                                           max=num_classes).type(torch.cuda.FloatTensor) / 100.0
                    labeling_error_prob_nom += torch.histc(torch.where(predicted_labels.eq(targets), \
                                                                       predicted_labels, \
                                                                       torch.FloatTensor([-1]).to(self.device)), \
                                                           bins=num_classes, min=0, \
                                                           max=num_classes).type(torch.cuda.FloatTensor) / 100.0
                    
                    if self.debug:                        
                        total2 += predicted_labels.size(0)
                        correct2 += predicted_labels.eq(targets).sum().item()
                                                                

            if self.debug:
                print('number of corrected samples: ')
                print(torch.sum(labeling_error_prob_nom))
                print('number of entire samples: ')
                print(torch.sum(labeling_error_prob_den))
                print('number of correct predictions')
                print('%d/%d'%(correct, total))
                print('number of correct predictions (training set)')
                print('%d/%d'%(correct2, total2))
                print('likelihood_prob 1: ')
                print(likelihood_prob[0])
                print('likelihood_prob 10: ')
                print(likelihood_prob[9])
                print('likelihood_prob 100: ')
                print(likelihood_prob[99])
            labeling_error_prob = labeling_error_prob_nom / labeling_error_prob_den
            labeling_error_prob = labeling_error_prob.cpu().numpy()
        
        
        likelihood_prob = np.array(likelihood_prob)
        
        scores = 1 - np.sum(likelihood_prob * np.reshape(prior_prob, [1,-1]), axis=1)
        
        if self.debug:
            print('prior_prob: ')
            print(prior_prob)
            print('labeling_error_prob_val: ')
            print((labeling_error_prob_nom2 / labeling_error_prob_den2).cpu().numpy())
            print('labeling_histogram_val: ')
            print(labeling_error_prob_den2)
            print('labeling_correct_histogram_val: ')
            print(labeling_error_prob_nom2)
            print('labeling_error_prob: ')
            print(labeling_error_prob)
            print('scores: ')
            print(scores)
                
        return scores
            
    def test(self):
        
        # network initialization...
        self.net.eval()
        
        # logging initialization...
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            # batch iterations...
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                # forward
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                
                # loss estimation (for logging)
                loss = self.criterion(outputs, targets)

                # logging...
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # Print Test Result
        print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        
        
    def save(self, ckpt_folder):
                
        # state building...
        state = {
            'net': self.net.state_dict()
        }
         
        # Save checkpoint...
        if not os.path.isdir(ckpt_folder):
            os.mkdir(ckpt_folder)
        torch.save(state, './'+ckpt_folder+'/model_ckpt.pth')
        
        return ckpt_folder
