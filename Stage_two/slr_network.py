import pdb
import copy

from numpy.core.numeric import zeros_like
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import modules.resnet as resnet
import random
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class MLP(nn.Module):
    def __init__(self, channels=512, out_classes=7):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(channels,channels)
        self.fc2 = nn.Linear(channels,out_classes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return F.softmax(self.fc2(self.relu(self.fc1(x))),dim=-1)

class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs

class SLRModel(nn.Module):
    def __init__(self, num_classes, c2d_small_type, c2d_type, conv_type, use_bn=False, hidden_size=1024, gloss_dict=None, 
                loss_weights=None, policy_kernel=1, tau=5, tau_type='linear', tau_decay=-0.43, warmup_epoches=5, policy='adaptive',
            weight_norm=True, share_classifier=True):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.conv2d_small = getattr(models, c2d_small_type)(pretrained=True)
        self.conv2d_small.fc = Identity()  # for regnet_x_800mf, resnet18
        self.conv2d_mid = getattr(models, c2d_small_type)(pretrained=True)
        self.conv2d_mid.fc = Identity()  # for regnet_x_800mf, resnet18
        self.conv2d_big = getattr(models, c2d_type)(pretrained=True)
        self.conv2d_big.fc = Identity() # for regnet_x_800mf, resnet18
        self.tau = tau
        self.tau_type = tau_type
        self.tau_decay = tau_decay
        self.epoch = 0
        self.warmup_epoches = warmup_epoches
        self.policy = policy
        if 'Flops' in  self.loss_weights.keys():
            self.ori_flops_weight = self.loss_weights['Flops']

        self.intervals_beginings = [[4,0],[4,1],[4,2],[4,3],[2,0],[2,1],[1,0]]
        self.policy_conv = MLP(512, len(self.intervals_beginings)*2+1)  # 1-4 1/4,5-6 1/2, 7 1
        self.policy_gru = nn.GRU(input_size=512, hidden_size=512, batch_first=True, num_layers=1, bidirectional=False) #672 regnet_x_800mf, 512 resnet18
        self.logit_predictor = MLP(1024, 2)
        self.conv1d = TemporalConv(input_size=512, #672 regnet_x_800mf, 512 resnet18
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.kernel_sizes = self.conv1d.kernel_size
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)
        self.classifier = nn.Linear(hidden_size, self.num_classes)
        self.conv1d_small = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.temporal_model_small = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size, num_layers=2, bidirectional=True)
        self.classifier_small = nn.Linear(hidden_size, self.num_classes)
        self.conv1d_mid = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.temporal_model_mid = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size, num_layers=2, bidirectional=True)
        self.classifier_mid= nn.Linear(hidden_size, self.num_classes)

        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
            self.classifier_mid = NormLinear(hidden_size, self.num_classes)
            self.conv1d_mid.fc = NormLinear(hidden_size, self.num_classes)
            self.classifier_small = NormLinear(hidden_size, self.num_classes)
            self.conv1d_small.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
            self.classifier_mid = nn.Linear(hidden_size, self.num_classes)
            self.conv1d_mid.fc = nn.Linear(hidden_size, self.num_classes)
            self.classifier_small = nn.Linear(hidden_size, self.num_classes)
            self.conv1d_small.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
            self.conv1d_mid.fc = self.classifier_mid
            self.conv1d_small.fc = self.classifier_small

    def masked_bn(self, model, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = model(x)        
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def hook_fn(self, grad):
        print('grad.size()')
        print(grad.size())
        print(grad)

    def train(self, mode=True):
        super(SLRModel, self).train(mode)
        if mode:
            if self.tau_type == 'linear':
                self.tau = self.tau * np.exp(self.tau_decay)
            elif self.tau_type == 'cos':
                self.tau = 0.01 + 0.5 * (self.tau - 0.01) * (1 + np.cos(np.pi * self.epoch / self.args.max_epoch))
            else:
                raise RuntimeError('no such tau type')
            print('current tau: ', self.tau)
            if 'Flops' in  self.loss_weights.keys() and self.epoch < self.warmup_epoches:
                self.loss_weights['Flops'] = self.epoch / self.warmup_epoches * self.ori_flops_weight
            self.epoch += 1

    def forward(self, x, x_mid, x_small,  len_x, label=None, label_lgt=None):
        self.conv2d_big.eval()
        self.conv2d_mid.eval()
        self.conv2d_small.eval()
        self.conv1d.eval()
        self.temporal_model.eval()
        self.classifier.eval()
        self.conv1d_small.eval()
        self.temporal_model_small.eval()
        self.classifier_small.eval()
        self.conv1d_mid.eval()
        self.temporal_model_mid.eval()
        self.classifier_mid.eval()
        if len(x.shape) == 4:   #batch size 1
            x_small = x_small.unsqueeze(0)
            x = x.unsqueeze(0)
        batch, temp, channel, height, width = x_small.shape
        x_small = x_small.reshape(batch * temp, channel, height, width)
        framewise_small = self.masked_bn(self.conv2d_small, x_small, len_x)
        framewise_small = framewise_small.reshape(batch, temp, -1).transpose(1, 2)  # output size B*C*T
        if self.policy == 'adaptive':
            action_input = framewise_small.detach()
            action_input = self.policy_gru(action_input.permute(0,2,1))[1].permute(1,2,0).squeeze(2) #bc
            action_input = self.policy_conv(action_input).clamp(min=1e-8)     #output prob with size B*C
            if self.training:
                action = F.gumbel_softmax(torch.log(action_input), self.tau, hard=True, dim=1)#   #B*8
            else:
                action = F.gumbel_softmax(torch.log(action_input), 1e-5, hard=True, dim=1).bool()   #B*8
                #action = F.one_hot(torch.argmax(action_input, dim=1), num_classes=2).permute(0,2,1).bool() #bt2 -> b2t
        elif self.policy == 'random':
            action = framewise_small.new(batch, 2, temp).zero_()
            randn = torch.rand((batch,temp))
            threshold = 0.5
            for i in range(batch):
                action[i,0,randn[i]>=threshold] = 1.0
                action[i,1,randn[i]<threshold] = 1.0
            if not self.training:
                action = action.bool()
        else:
            raise ValueError("Not supported policy, please choose from 'adaptive|random'")

        def pad_seq(framewise, data, action):
                framewise_new = framewise.new(framewise.size()).zero_()
                for j in range(data[0]):
                    framewise_new[:,:,j::data[0]] =  framewise[:,:,data[1]::data[0]]
                return framewise_new

        def extend_seq(framewise, data, len_x):
                out = framewise.new(framewise.size(0), len_x)
                def pad_data(tensor, length):
                    return torch.cat([tensor, tensor[:,-length:]],1) if length>0 else tensor
                for i in range(data[0]):
                    tar_l = out[:,i::data[0]].size(1)  #int(torch.ceil((len_x-i)/data[0]))
                    out[:,i::data[0]] =  pad_data(framewise[:,:tar_l], tar_l-framewise.size(1))
                return out

        def forward_pass(input_data, len_x):
            input_data, left_pad, right_pad= self.pad_video(input_data,len_x)
            conv1d_outputs = self.conv1d(input_data, len_x+left_pad+right_pad)
            # x: T, B, C
            temporal_data = conv1d_outputs['visual_feat']
            lgt = conv1d_outputs['feat_len']
            tm_outputs = self.temporal_model(temporal_data, lgt)
            outputs = self.classifier(tm_outputs['predictions'])
            return conv1d_outputs['conv_logits'], outputs, lgt

        def forward_pass_mid(input_data, len_x):
            input_data, left_pad, right_pad= self.pad_video(input_data,len_x)
            conv1d_outputs = self.conv1d_mid(input_data, len_x+left_pad+right_pad)
            # x: T, B, C
            temporal_data = conv1d_outputs['visual_feat']
            lgt = conv1d_outputs['feat_len']
            tm_outputs = self.temporal_model_mid(temporal_data, lgt)
            outputs = self.classifier_mid(tm_outputs['predictions'])
            return conv1d_outputs['conv_logits'], outputs, lgt
        if self.training:
            #get frame-wise features
            batch, temp, channel, height, width = x.shape
            framewise_big = self.masked_bn(self.conv2d_big, x.reshape(batch * temp, channel, height, width), len_x).reshape(batch, temp, -1).transpose(1, 2) # output size B*C*T  
            batch, temp, channel, height, width = x_mid.shape
            framewise_mid = self.masked_bn(self.conv2d_mid, x_mid.reshape(batch * temp, channel, height, width), len_x).reshape(batch, temp, -1).transpose(1, 2) # output size B*C*T  
            
            input_data = torch.stack([ pad_seq(framewise_big, data, action[:,i]) for i, data in enumerate(self.intervals_beginings)], 1) #bsct
            input_data_mid = torch.stack([ pad_seq(framewise_mid, data, action[:,i]) for i, data in enumerate(self.intervals_beginings)], 1) #bsct
            conv_logits = []
            sequence_logits = []
            for i in range(len(self.intervals_beginings)):
                conv_logit, sequence_logit, lgt = forward_pass(input_data[:,i], len_x)
                conv_logits.append(conv_logit)
                sequence_logits.append(sequence_logit)
            for i in range(len(self.intervals_beginings)):
                conv_logit, sequence_logit, lgt = forward_pass_mid(input_data_mid[:,i], len_x)
                conv_logits.append(conv_logit)
                sequence_logits.append(sequence_logit)
            conv_logits_train = torch.stack(conv_logits, 1)  #tsbc
            sequence_logits_train = torch.stack(sequence_logits, 1)

            framewise_small_pad, left_pad, right_pad = self.pad_video(framewise_small, len_x)
            conv1d_outputs_small = self.conv1d_small(framewise_small_pad, len_x + left_pad + right_pad)
            x_small = conv1d_outputs_small['visual_feat']
            output_vid_length_small = conv1d_outputs_small['feat_len']
            tm_outputs_small = self.temporal_model_small(x_small, output_vid_length_small)
            outputs_small = self.classifier_small(tm_outputs_small['predictions'])

            len_i = len(self.intervals_beginings)
            features_avg = torch.concat([input_data.mean(3), input_data_mid.mean(3), framewise_small.mean(2).unsqueeze(1)],1) #bsc
            features_avg = torch.concat([features_avg, features_avg[:,-1:].repeat(1, 2*len_i+1, 1)], 2)
            logits = F.softmax(12*F.sigmoid(self.logit_predictor(features_avg.detach())), 2).permute(2,1,0) #bs2->2sb

            conv_logits = torch.concat([conv_logits_train, conv1d_outputs_small['conv_logits'].unsqueeze(1)], 1)  #tsbc
            sequence_logits = torch.concat([sequence_logits_train, outputs_small.unsqueeze(1)], 1)
            conv_logits = torch.einsum('tsbc,bs->tbc', (logits[0:1].unsqueeze(-1)*conv_logits.detach()+conv1d_outputs_small['conv_logits'].unsqueeze(1).repeat(1,len_i*2+1,1,1).detach() * logits[1:2].unsqueeze(-1)), action) 
            sequence_logits = torch.einsum('tsbc,bs->tbc', (logits[0:1].unsqueeze(-1)*sequence_logits.detach()+outputs_small.unsqueeze(1).repeat(1,len_i*2+1,1,1).detach() * logits[1:2].unsqueeze(-1)), action) 
        else:
            framewise_small_pad, left_pad, right_pad = self.pad_video(framewise_small, len_x)
            conv1d_outputs_small = self.conv1d_small(framewise_small_pad, len_x + left_pad + right_pad)
            x_small = conv1d_outputs_small['visual_feat']
            output_vid_length_small = conv1d_outputs_small['feat_len']
            tm_outputs_small = self.temporal_model_small(x_small, output_vid_length_small)
            outputs_small = self.classifier_small(tm_outputs_small['predictions'])

            outputs = []
            lgt = framewise_small.new(batch).zero_()
            len_i = len(self.intervals_beginings)
            logits = framewise_small.new(batch, 2).zero_()
            for i in range(batch):
                j = 0 
                while(j< len_i*2):
                    if j< len_i and action[i,j]:
                        input_data = extend_seq(self.conv2d_big(x[i,self.intervals_beginings[j][1]:len_x[i]:self.intervals_beginings[j][0]]).transpose(0, 1),  self.intervals_beginings[j], len_x[i]).unsqueeze(0)
                        logits[i:i+1] = F.softmax(12*F.sigmoid(self.logit_predictor(torch.concat([input_data.mean(2), framewise_small[i:i+1].mean(2)], 1))), -1) #b2
                        input_data, left_pad, right_pad= self.pad_video(input_data,len_x[i:i+1])
                        conv1d_outputs = self.conv1d(input_data, len_x[i:i+1]+left_pad+right_pad)
                        # x: T, B, C
                        temporal_data = conv1d_outputs['visual_feat']
                        tmp_lgt = conv1d_outputs['feat_len']
                        tm_outputs = self.temporal_model(temporal_data, tmp_lgt)
                        outputs.append(self.classifier(tm_outputs['predictions']))
                        lgt[i] = tmp_lgt[0]
                        break
                    elif j>= len_i and j< len_i*2 and action[i,j]:
                        input_data = extend_seq(self.conv2d_mid(x_mid[i,self.intervals_beginings[j% len_i][1]:len_x[i]:self.intervals_beginings[j% len_i][0]]).transpose(0, 1),  self.intervals_beginings[j% len_i], len_x[i]).unsqueeze(0)
                        logits[i:i+1] = F.softmax(12*F.sigmoid(self.logit_predictor(torch.concat([input_data.mean(2), framewise_small[i:i+1].mean(2)], 1))), -1) #b2
                        input_data, left_pad, right_pad= self.pad_video(input_data,len_x[i:i+1])
                        conv1d_outputs = self.conv1d_mid(input_data, len_x[i:i+1]+left_pad+right_pad)
                        # x: T, B, C
                        temporal_data = conv1d_outputs['visual_feat']
                        tmp_lgt = conv1d_outputs['feat_len']
                        tm_outputs = self.temporal_model_mid(temporal_data, tmp_lgt)
                        outputs.append(self.classifier_mid(tm_outputs['predictions']))
                        lgt[i] = tmp_lgt[0]
                        break
                    j +=1
                if j== len_i*2:
                    outputs.append(outputs_small[:,i:i+1])
                    logits[i:i+1] = 0.5
                    lgt[i] = output_vid_length_small[i]
            def pad(tensor, length):
                return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])
            outputs = torch.concat([pad(tmp, int(max(lgt))) for tmp in outputs], 1)
        pred = None if self.training else self.decoder.decode(outputs*logits[:,0:1].unsqueeze(0)+outputs_small*logits[:,1:2].unsqueeze(0), lgt, batch_first=False, probs=False)
        
        return {
            "feat_len": lgt,
            "conv_logits": conv_logits if self.training else None,
            "sequence_logits": sequence_logits if self.training else None,
            "conv_logits_train": conv_logits_train if self.training else None,
            "sequence_logits_train": sequence_logits_train if self.training else None,
            "conv_logits_small": conv1d_outputs_small['conv_logits'],
            "sequence_logits_small": outputs_small,
            "recognized_sents": pred,
            "action": action, 
            "len_x": len_x,
            "framewise_small": framewise_small if self.training else None, 
            "framewise_big": framewise_big if self.training else None,
            "framewise_mid": framewise_mid if self.training else None,
        }

    # pad videos according to the kernel size and stride of temporal model
    def pad_video(self, video, len_x):
        left_pad = 0
        last_stride = 1
        total_stride = 1
        for _, ks in enumerate(self.kernel_sizes):
            if ks[0] == 'K':
                left_pad = left_pad * last_stride 
                left_pad += int((int(ks[1])-1)/2)
            elif ks[0] == 'P':
                last_stride = int(ks[1])
                total_stride = total_stride * last_stride
        right_pad = int(np.ceil(video.size(2) / total_stride)) * total_stride - video.size(2) + left_pad
        padded_video = torch.stack([torch.cat(
            (
                vid[:,0:1].repeat(1, left_pad),  #CT
                vid[:,:len_x[i]-1],         # len_x-1 frames
                vid[:,len_x[i]-1:len_x[i]].repeat(1, right_pad),
                vid[:,len_x[i]-1:]          # at least 1 frame
            )  , dim=1) for i,vid in enumerate(video)])
        return padded_video, left_pad, right_pad

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        total_loss = {}
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                total_loss['ConvCTC'] = weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()  
                loss += total_loss['ConvCTC']
            elif k == 'SeqCTC':
                total_loss['SeqCTC'] = weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean() 
                loss += total_loss['SeqCTC']
            elif k == 'Dist':
                total_loss['Dist'] = weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False) + weight * self.loss['distillation'](ret_dict["conv_logits_small"], ret_dict["sequence_logits_small"].detach(), use_blank=False) 
                loss += total_loss['Dist']
            elif k == 'Flops':
                total_loss['Flops'] = weight * self.loss['Flops'](ret_dict["action"])
                loss += total_loss['Flops']
            elif k=='Frame_dist':
                total_loss['Frame_dist'] = weight * self.loss['distillation_frame'](ret_dict["framewise_small"],
                                                           ret_dict["framewise_big"].detach(),
                                                           use_blank=False) +  weight * self.loss['distillation_frame'](ret_dict["framewise_mid"],
                                                           ret_dict["framewise_big"].detach(),
                                                           use_blank=False)
                loss += total_loss['Frame_dist']

        return loss, total_loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        self.loss['Flops'] = Flops_loss()
        self.loss['distillation_frame'] = SeqKD(T=8)
        return self.loss
    
class Flops_loss(nn.Module):
    def __init__(self, T=1):
        super(Flops_loss, self).__init__()
        self.flops_vector = torch.FloatTensor([0.43,0.43,0.43,0.43,0.68,0.68,1.18, 0.31, 0.31, 0.31, 0.31, 0.43, 0.43, 0.68, 0.18])

    def forward(self, action):
        # bs * s -> b
        return torch.mean(torch.einsum('bs,s->b',action, self.flops_vector.to(action.get_device())))
