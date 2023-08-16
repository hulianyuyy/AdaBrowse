import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import pdb
import sys
import cv2
import yaml
import torch
import random
import importlib
import faulthandler
import numpy as np
import torch.nn as nn
import shutil
import inspect
import time
from collections import OrderedDict

faulthandler.enable()
import utils
from seq_scripts import seq_train, seq_eval, seq_eval_separate, seq_feature_generation
from modules.sync_batchnorm import convert_model
from torch.cuda.amp import autocast as autocast


class Processor():
    def __init__(self, arg):
        self.arg = arg
        if os.path.exists(self.arg.work_dir):
            answer = input('Current dir exists, do you want to remove and refresh it? (y/n)\n')
            if answer in ['yes','y','ok','1']:
                print('Dir removed !')
                shutil.rmtree(self.arg.work_dir)
                os.makedirs(self.arg.work_dir)
            else:
                print('Dir Not removed !')
        else:
            os.makedirs(self.arg.work_dir)
        shutil.copy2(__file__, self.arg.work_dir)
        shutil.copy2('./configs/baseline.yaml', self.arg.work_dir)
        shutil.copy2('./seq_scripts.py', self.arg.work_dir) # added after iter 6
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        self.recoder_iter = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval, file_name='log_iterations')
        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.device = utils.GpuDataParallel()
        self.dataset = {}
        self.data_loader = {}
        self.gloss_dict = np.load(self.arg.dataset_info['dict_path'], allow_pickle=True).item()
        self.arg.model_args['num_classes'] = len(self.gloss_dict) + 1
        self.model, self.optimizer = self.loading()

    def start(self):
        if self.arg.phase == 'train':
            self.best_dev = 100.0
            best_epoch = 0
            total_time = 0
            epoch_time = 0
            self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            seq_model_list = []
            for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):
                save_model = epoch % self.arg.save_interval == 0
                eval_model = epoch % self.arg.eval_interval == 0
                epoch_time = time.time()
                # train end2end model
                seq_train(self.data_loader['train'], self.model, self.optimizer,
                          self.device, epoch, self.recoder_iter, loss_weights= self.arg.loss_weights)
                if eval_model:
                    dev_wer = seq_eval(self.arg, self.data_loader['dev'], self.model, self.device,
                                       'dev', epoch, self.arg.work_dir, self.recoder)
                    self.recoder.print_log("Dev WER: {:05.2f}".format(dev_wer))
                if dev_wer < self.best_dev:
                    self.best_dev = dev_wer
                    best_epoch = epoch
                    model_path = "{}best_model.pt".format(self.arg.work_dir)
                    self.save_model(epoch, model_path)
                    self.recoder.print_log('Save best model')
                self.recoder.print_log('Best_dev: {:05.2f}, Epoch : {}'.format(self.best_dev, best_epoch))
                if save_model:
                    model_path = "{}dev_{:05.2f}_epoch{}_model.pt".format(self.arg.work_dir, dev_wer, epoch)
                    seq_model_list.append(model_path)
                    print("seq_model_list", seq_model_list)
                    self.save_model(epoch, model_path)
                epoch_time = time.time() - epoch_time
                total_time += epoch_time
                self.recoder.print_log('Epoch {} costs {} mins {} seconds'.format(epoch, int(epoch_time)//60, int(epoch_time)%60))
            self.recoder.print_log('Training costs {} hours {} mins {} seconds'.format(int(total_time)//60//60, int(total_time)//60%60, int(total_time)%60))
        elif self.arg.phase == 'test':
            if self.arg.load_weights is None  and self.arg.load_checkpoints is None:
                self.recoder.print_log('Please appoint --weights.')
            self.recoder.print_log('Model:   {}.'.format(self.arg.model))
            self.recoder.print_log('Weights: {}.'.format(self.arg.load_weights))
            # train_wer = seq_eval(self.arg, self.data_loader["train_eval"], self.model, self.device,
            #                      "train", 6667, self.arg.work_dir, self.recoder)
            if self.arg.separate_eval:
                dev_wer = seq_eval_separate(self.arg, self.data_loader["dev"], self.model, self.device, "dev", 6667, self.arg.work_dir, self.recoder)
                test_wer = seq_eval_separate(self.arg, self.data_loader["test"], self.model, self.device,"test", 6667, self.arg.work_dir, self.recoder)
            else:
                dev_wer = seq_eval(self.arg, self.data_loader["dev"], self.model, self.device, "dev", 6667, self.arg.work_dir, self.recoder)
                test_wer = seq_eval(self.arg, self.data_loader["test"], self.model, self.device,"test", 6667, self.arg.work_dir, self.recoder)
            self.recoder.print_log('Evaluation Done.\n')
        elif self.arg.phase == "features":
            for mode in ["train", "dev", "test"]:
                seq_feature_generation(
                    self.data_loader[mode + "_eval" if mode == "train" else mode],
                    self.model, self.device, mode, self.arg.work_dir, self.recoder
                )

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def save_model(self, epoch, save_path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
            'rng_state': self.rng.save_rng_state() if self.arg.random_fix else None,
            'tau': self.model.tau, 
            'best_dev': self.best_dev,
        }, save_path)

    def adjust_lr(self, model):
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        for name, m in model.named_modules():
            if 'policy' in name : # only adjust policy_conv
                #if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.BatchNorm3d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.LSTM):
                if len(list(m.parameters()))>0:
                    ps = list(m.parameters())
                    lr5_weight.append(ps[0])
                    if len(ps) == 2:
                        lr10_bias.append(ps[1])
            elif len(list(m.parameters()))>0:
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
        return [{'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
             {'params': lr5_weight, 'lr_mult':  0.0001 , 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult':  0.0002 , 'decay_mult': 0, 
             'name': "lr10_bias"},
             ]
            
    def loading(self):
        self.device.set_device(self.arg.device)
        print("Loading model")
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            gloss_dict=self.gloss_dict,
            loss_weights=self.arg.loss_weights,
        )
        shutil.copy2(inspect.getfile(model_class), self.arg.work_dir)
        adjust_lr_params = self.adjust_lr(model)
        for group in adjust_lr_params:
            print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
                group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
        optimizer = utils.Optimizer(adjust_lr_params, self.arg.optimizer_args)

        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)
        else:
            self.load_init_weights(model, self.arg.conv2d_small_weight_path, self.arg.conv2d_mid_weight_path, self.arg.conv2d_big_weight_path)
        model = self.model_to_device(model)
        self.kernel_sizes = model.conv1d.kernel_size
        print("Loading model finished.")
        self.load_data()
        return model, optimizer

    def model_to_device(self, model):
        model = model.to(self.device.output_device)
        if len(self.device.gpu_list) > 1:
            model = nn.DataParallel(
                model,
                device_ids=self.device.gpu_list,
                output_device=self.device.output_device)
        model = convert_model(model)
        model.cuda()
        return model
    
    def load_init_weights(self, model, conv2d_small_weight_path, conv2d_mid_weight_path, conv2d_big_weight_path):
        state_dict_small = torch.load(conv2d_small_weight_path)['model_state_dict']
        #for tmp in list(state_dict_small.keys()):
        #    if 'conv1d' in tmp or 'temporal_model' in tmp or 'classifier' in tmp:
        #        del state_dict_small[tmp]
        state_dict_small = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict_small.items()])
        state_dict_small = OrderedDict([(k.replace('conv2d.', 'conv2d_small.'), v) for k, v in state_dict_small.items()])
        state_dict_small = OrderedDict([(k.replace('classifier.', 'classifier_small.'), v) for k, v in state_dict_small.items()])
        state_dict_small = OrderedDict([(k.replace('temporal_model.', 'temporal_model_small.'), v) for k, v in state_dict_small.items()])
        state_dict_small = OrderedDict([(k.replace('conv1d.', 'conv1d_small.'), v) for k, v in state_dict_small.items()])

        state_dict_mid = torch.load(conv2d_mid_weight_path)['model_state_dict']
        state_dict_mid = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict_mid.items()])
        state_dict_mid = OrderedDict([(k.replace('conv2d.', 'conv2d_mid.'), v) for k, v in state_dict_mid.items()])
        state_dict_mid = OrderedDict([(k.replace('classifier.', 'classifier_mid.'), v) for k, v in state_dict_mid.items()])
        state_dict_mid = OrderedDict([(k.replace('temporal_model.', 'temporal_model_mid.'), v) for k, v in state_dict_mid.items()])
        state_dict_mid = OrderedDict([(k.replace('conv1d.', 'conv1d_mid.'), v) for k, v in state_dict_mid.items()])

        state_dict_big = torch.load(conv2d_big_weight_path)['model_state_dict']
        state_dict_big = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict_big.items()])
        state_dict_big = OrderedDict([(k.replace('conv2d.', 'conv2d_big.'), v) for k, v in state_dict_big.items()])
        # K5 P2 K5 P2 to K5 K5
        #state_dict_big = OrderedDict([(k.replace('temporal_conv.4', 'temporal_conv.3'), v) for k, v in state_dict_big.items()])
        #state_dict_big = OrderedDict([(k.replace('temporal_conv.5', 'temporal_conv.4'), v) for k, v in state_dict_big.items()])

        state_dict = state_dict_small.copy()
        state_dict.update(state_dict_mid)
        state_dict.update(state_dict_big)
        model.load_state_dict(state_dict, strict = False)        # no policy conv weight
        print('Load init weight successfully')

    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path)
        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        weights = self.modified_weights(state_dict['model_state_dict'], False)
        # weights = self.modified_weights(state_dict['model_state_dict'])
        model.load_state_dict(weights, strict=True)

    @staticmethod
    def modified_weights(state_dict, modified=False):
        state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict

    def load_checkpoint_weights(self, model, optimizer):
        self.load_model_weights(model, self.arg.load_checkpoints)
        state_dict = torch.load(self.arg.load_checkpoints)

        if len(torch.cuda.get_rng_state_all()) == len(state_dict['rng_state']['cuda']) & self.arg.random_fix:
            print("Loading random seeds")
            if state_dict['rng_state']!=None:
                self.rng.set_rng_state(state_dict['rng_state'])
            else:
                print("Load null random seeds...")
        if "optimizer_state_dict" in state_dict.keys():
            print("Loading optimizer parameters...")
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            optimizer.to(self.device.output_device)
            #for k, v in optimizer.state_dict().items():
            #    if torch.is_tensor(v):
            #        optimizer.state_dict()[k] = v.cuda()
        if "scheduler_state_dict" in state_dict.keys():
            print("Loading scheduler parameters...")
            optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        if 'best_dev' in state_dict.keys(): 
            print("Loading Best_dev...")
            self.best_dev = state_dict['best_dev']

        self.arg.optimizer_args['start_epoch'] = state_dict["epoch"] + 1
        for _ in range(self.arg.optimizer_args['start_epoch']):
            model.train()
        #model.tau = state_dict["tau"]
        self.recoder.print_log(f"Resuming from checkpoint: epoch {self.arg.optimizer_args['start_epoch']}")

    def load_data(self):
        print("Loading data")
        self.feeder = import_class(self.arg.feeder)
        shutil.copy2(inspect.getfile(self.feeder), self.arg.work_dir)
        if self.arg.dataset == 'CSL':
            dataset_list = zip(["train", "dev"], [True, False])
        elif 'phoenix' in self.arg.dataset:
            dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False]) 
        elif self.arg.dataset == 'CSL-Daily':
            dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False])
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["prefix"] = self.arg.dataset_info['dataset_root']
            arg["mode"] = mode.split("_")[0]
            arg["transform_mode"] = train_flag
            self.dataset[mode] = self.feeder(gloss_dict=self.gloss_dict, kernel_size= self.kernel_sizes, dataset=self.arg.dataset, **arg)
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)
        print("Loading data finished.")
    def init_fn(self, worker_id):
        np.random.seed(int(self.arg.random_seed)+worker_id)
    def build_dataloader(self, dataset, mode, train_flag):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
            shuffle=train_flag,
            drop_last=train_flag,
            num_workers=self.arg.num_worker,  # if train_flag else 0
            collate_fn=self.feeder.collate_fn,
            pin_memory=True, # added later
            worker_init_fn=self.init_fn,
        )

def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod

if __name__ == '__main__':
    sparser = utils.get_parser()
    p = sparser.parse_args()
    # p.config = "baseline_iter.yaml"
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
    processor = Processor(args)
    utils.pack_code("./", args.work_dir)
    processor.start()
