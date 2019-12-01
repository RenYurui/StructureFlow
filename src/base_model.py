import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from .utils import weights_init, get_iteration


class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.samples_path = os.path.join(config.PATH, config.NAME, 'images')
        self.checkpoints_path = os.path.join(config.PATH, config.NAME, 'checkpoints')

    def save(self, which_epoch):
        """Save all the networks to the disk"""
        for net_name in self.net_name:
            if hasattr(self, net_name) and not(self.config.MODEL == 3 and 's_' in net_name):
                sub_net = getattr(self, net_name)
                save_filename = '%s_net_%s.pth' % (which_epoch, net_name)
                save_path = os.path.join(self.checkpoints_path, save_filename)
                torch.save(sub_net.state_dict(), save_path)


    def load(self, which_epoch):
        for net_name in self.net_name:
            if hasattr(self, net_name):
                sub_net = getattr(self, net_name)
                filename = '%s_net_%s.pth' % (which_epoch, net_name)
                model_name = os.path.join(self.checkpoints_path, filename)
                if not os.path.isfile(model_name):
                    print('checkpoint %s do not exist'%model_name)
                    continue                
                self.load_networks(model_name, sub_net, net_name)
                self.iterations = get_iteration(self.checkpoints_path, filename, net_name)
                print('Resume %s from iteration %s' % (net_name, which_epoch))

                sub_net_opt = getattr(self, net_name+'_opt')
                setattr(self, net_name+'_scheduler', self.get_scheduler(sub_net_opt))


    def load_networks(self, path, net, name):
        """Load all the networks from the disk"""
        try:
            net.load_state_dict(torch.load(path))
        except:
            pretrained_dict = torch.load(path)
            model_dict = net.state_dict()
            try:
                pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
                net.load_state_dict(pretrained_dict)
                print('Pretrained network %s has excessive layers; Only loading layers that are used' % name)
            except:
                print('Pretrained network %s has fewer layers; The following are not initialized:' % name)
                not_initialized = set()
                for k, v in pretrained_dict.items():
                    if v.size() == model_dict[k].size():
                        model_dict[k] = v

                for k, v in model_dict.items():
                    if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                        not_initialized.add(k)
                print(sorted(not_initialized))
                net.load_state_dict(model_dict)


    def init(self):
        for net_name in self.net_name:
            if hasattr(self, net_name):
                sub_net = getattr(self, net_name)
                sub_net.apply(weights_init(self.config.INIT_TYPE))


    def define_optimizer(self):
        for net_name in self.net_name:
            if hasattr(self, net_name):
                sub_net = getattr(self, net_name)
                optimizer = optim.Adam(sub_net.parameters(),lr=self.config.LR,betas=(self.config.BETA1, self.config.BETA2))
                scheduler = self.get_scheduler(optimizer)
                setattr(self, net_name+'_opt', optimizer)
                setattr(self, net_name+'_scheduler', scheduler)


    def get_scheduler(self, optimizer):
        if self.config.LR_POLICY == None or self.config.LR_POLICY == 'constant':
            scheduler = None 
        elif self.config.LR_POLICY == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=self.config.STEP_SIZE,
                                            gamma=self.config.GAMMA, last_epoch = self.iterations-1)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', self.config.LR_POLICY)
        return scheduler





   

      
