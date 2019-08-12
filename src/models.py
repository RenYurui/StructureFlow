import torch
import torch.nn as nn
from .base_model import BaseModel
from .network import StructureGen
from .network import MultiDiscriminator
from .network import FlowGen
from .loss import AdversarialLoss, PerceptualCorrectness, StyleLoss, PerceptualLoss


class StructureFlowModel(BaseModel):
    def __init__(self, config):
        super(StructureFlowModel, self).__init__('StructureFlow', config)
        self.config = config
        self.net_name = ['s_gen', 's_dis', 'f_gen', 'f_dis'] 

        self.structure_param = {'input_dim':3, 'dim':64, 'n_res':4, 'activ':'relu', 
                         'norm':'in', 'pad_type':'reflect', 'use_sn':True}
        self.flow_param = {'input_dim':3, 'dim':64, 'n_res':2, 'activ':'relu', 
                         'norm_conv':'ln', 'norm_flow':'in', 'pad_type':'reflect', 'use_sn':False}
        self.dis_param =  {'input_dim':3, 'dim':64, 'n_layers':3, 
                         'norm':'none', 'activ':'lrelu', 'pad_type':'reflect', 'use_sn':True}

        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=config.DIS_GAN_LOSS)
        correctness_loss = PerceptualCorrectness()
        vgg_style = StyleLoss()
        vgg_content = PerceptualLoss()
        self.use_correction_loss = True
        self.use_vgg_loss = True if self.config.MODEL == 3 else False

        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        self.add_module('correctness_loss', correctness_loss)
        self.add_module('vgg_style', vgg_style)
        self.add_module('vgg_content', vgg_content)

        self.build_model()

    def build_model(self):
        self.iterations = 0
        # structure model
        if self.config.MODEL == 1:
            self.s_gen = StructureGen(**self.structure_param)
            self.s_dis = MultiDiscriminator(**self.dis_param)   
        # flow model with true input smooth
        elif self.config.MODEL == 2:
            self.f_gen = FlowGen(**self.flow_param)
            self.f_dis = MultiDiscriminator(**self.dis_param)
        # flow model with fake input smooth
        elif self.config.MODEL == 3:
            self.s_gen = StructureGen(**self.structure_param)
            self.f_gen = FlowGen(**self.flow_param)
            self.f_dis = MultiDiscriminator(**self.dis_param)

        self.define_optimizer()
        self.init()


    def structure_forward(self, inputs, smooths, maps):
        smooths_input = smooths*(1-maps)
        outputs = self.s_gen(torch.cat((inputs, smooths_input, maps),dim=1))
        return outputs

    def flow_forward(self, inputs, stage_1, maps):
        outputs, flow = self.f_gen(torch.cat((inputs, stage_1, maps),dim=1))
        return outputs, flow

    def sample(self, inputs, smooths, gts, maps):
        with torch.no_grad(): 
            if self.config.MODEL == 1:
                outputs = self.structure_forward(inputs, smooths, maps)
                result =[inputs,smooths,gts,maps,outputs]
                flow = None
            elif self.config.MODEL == 2:
                outputs, flow = self.flow_forward(inputs, smooths, maps)
                result =[inputs,smooths,gts,maps,outputs]
                if flow is not None:
                    flow = [flow[:,0,:,:].unsqueeze(1)/30, flow[:,1,:,:].unsqueeze(1)/30]
 
            elif self.config.MODEL == 3:
                smooth_stage_1 = self.structure_forward(inputs, smooths, maps)
                outputs, flow = self.flow_forward(inputs, smooth_stage_1, maps)
                result =[inputs,smooths,gts,maps,smooth_stage_1,outputs]
                if flow is not None:
                    flow = [flow[:,0,:,:].unsqueeze(1)/30, flow[:,1,:,:].unsqueeze(1)/30]
        return result, flow       

    def update_structure(self, inputs, smooths, maps):
        self.iterations += 1

        self.s_gen.zero_grad()
        self.s_dis.zero_grad()
        outputs = self.structure_forward(inputs, smooths, maps)

        dis_loss = 0
        dis_fake_input = outputs.detach()
        dis_real_input = smooths
        fake_labels = self.s_dis(dis_fake_input)
        real_labels = self.s_dis(dis_real_input)
        for i in range(len(fake_labels)):
            dis_real_loss = self.adversarial_loss(real_labels[i], True, True)
            dis_fake_loss = self.adversarial_loss(fake_labels[i], False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2
        self.structure_adv_dis_loss = dis_loss/len(fake_labels)

        self.structure_adv_dis_loss.backward()
        self.s_dis_opt.step()
        if self.s_dis_scheduler is not None:
            self.s_dis_scheduler.step()


        dis_gen_loss = 0
        fake_labels = self.s_dis(outputs)
        for i in range(len(fake_labels)):
            dis_fake_loss = self.adversarial_loss(fake_labels[i], True, False)
            dis_gen_loss += dis_fake_loss
        self.structure_adv_gen_loss = dis_gen_loss/len(fake_labels) * self.config.STRUCTURE_ADV_GEN
        self.structure_l1_loss = self.l1_loss(outputs, smooths) * self.config.STRUCTURE_L1
        self.structure_gen_loss = self.structure_l1_loss + self.structure_adv_gen_loss

        self.structure_gen_loss.backward()
        self.s_gen_opt.step()
        if self.s_gen_scheduler is not None:
            self.s_gen_scheduler.step() 

        logs = [
            ("l_s_adv_dis", self.structure_adv_dis_loss.item()),
            ("l_s_l1", self.structure_l1_loss.item()),
            ("l_s_adv_gen", self.structure_adv_gen_loss.item()),
            ("l_s_gen", self.structure_gen_loss.item()),
        ]
        return logs  


    def update_flow(self, inputs, smooths, gts, maps, use_correction_loss, use_vgg_loss):
        self.iterations += 1

        self.f_dis.zero_grad()
        self.f_gen.zero_grad()
        outputs, flow_maps = self.flow_forward(inputs, smooths, maps)


        dis_loss = 0
        dis_fake_input = outputs.detach()
        dis_real_input = gts
        fake_labels = self.f_dis(dis_fake_input)
        real_labels = self.f_dis(dis_real_input)
        # self.flow_adv_dis_loss = (dis_real_loss + dis_fake_loss) / 2
        for i in range(len(fake_labels)):
            dis_real_loss = self.adversarial_loss(real_labels[i], True, True)
            dis_fake_loss = self.adversarial_loss(fake_labels[i], False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2
        self.flow_adv_dis_loss = dis_loss/len(fake_labels)        

        self.flow_adv_dis_loss.backward()
        self.f_dis_opt.step()
        if self.f_dis_scheduler is not None:
            self.f_dis_scheduler.step()


        dis_gen_loss = 0
        fake_labels = self.f_dis(outputs)
        for i in range(len(fake_labels)):
            dis_fake_loss = self.adversarial_loss(fake_labels[i], True, False)
            dis_gen_loss += dis_fake_loss  
        self.flow_adv_gen_loss = dis_gen_loss/len(fake_labels) * self.config.FLOW_ADV_GEN        
        self.flow_l1_loss = self.l1_loss(outputs, gts) * self.config.FLOW_L1
        self.flow_correctness_loss = self.correctness_loss(gts, inputs, flow_maps, maps)* \
                                     self.config.FLOW_CORRECTNESS if use_correction_loss else 0


        if use_vgg_loss:
            self.vgg_loss_style = self.vgg_style(outputs*maps, gts*maps)*self.config.VGG_STYLE  
            self.vgg_loss_content = self.vgg_content(outputs, gts)*self.config.VGG_CONTENT
            self.vgg_loss = self.vgg_loss_style + self.vgg_loss_content
        else:
            self.vgg_loss = 0

        self.flow_loss = self.flow_adv_gen_loss + self.flow_l1_loss + self.flow_correctness_loss + self.vgg_loss

        self.flow_loss.backward()
        self.f_gen_opt.step()

        if self.f_gen_scheduler is not None:
            self.f_gen_scheduler.step()  


        logs = [
            ("l_f_adv_dis", self.flow_adv_dis_loss.item()),
            ("l_f_adv_gen", self.flow_adv_gen_loss.item()),
            ("l_f_l1_gen", self.flow_l1_loss.item()),
            ("l_f_total_gen", self.flow_loss.item()),
        ]
        if use_correction_loss:
            logs = logs + [("l_f_correctness_gen", self.flow_correctness_loss.item())]
        if use_vgg_loss:
            logs = logs + [("l_f_vgg_style", self.vgg_loss_style.item())]            
            logs = logs + [("l_f_vgg_content", self.vgg_loss_content.item())]            
        return logs           
