import os
import pdb
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from evaluation.slr_eval.wer_calculation import evaluate
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

def seq_train(loader, model, optimizer, device, epoch_idx, recoder, loss_weights=None):
    model.train()
    loss_value = []
    total_loss_dict = {}    # dict of all types of loss
    action_num = 7
    action_count = [0 for i in range(action_num)]
    for k in loss_weights.keys(): 
        total_loss_dict[k] = 0
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    scaler = GradScaler()
    for batch_idx, data in enumerate(loader):
        vid = device.data_to_device(data[0])
        vid_mid = device.data_to_device(data[1])
        vid_small = device.data_to_device(data[2])
        vid_lgt = device.data_to_device(data[3])
        label = device.data_to_device(data[4])
        label_lgt = device.data_to_device(data[5])
        optimizer.zero_grad()
        with autocast():
            ret_dict = model(vid, vid_mid, vid_small, vid_lgt, label=label, label_lgt=label_lgt)
            loss, loss_dict = model.criterion_calculation(ret_dict, label, label_lgt)
        # update orders for label and label_lgt
        for i in range(vid.size(0)):
            for j in range(ret_dict['action'].size(1)):
                if ret_dict['action'][i,j]:
                    if j>=0 and j<4:
                        j=0
                        break
                    elif j>=4 and j<6:
                        j=1
                        break
                    elif j==6 :
                        j=2
                        break
                    if j>=7 and j<11:
                        j=3
                        break
                    elif j>=11 and j<13:
                        j=4
                        break
                    elif j==13 :
                        j=5
                        break
                    elif j==14 :
                        j=6
                        break
            action_count[j] = action_count[j] + 1
        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print('loss is nan')
            print(data[-1])
            continue
        scaler.scale(loss).backward()
        scaler.step(optimizer.optimizer)
        scaler.update()
        # nn.utils.clip_grad_norm_(model.rnn.parameters(), 5)
        loss_value.append(loss.item())
        for item, value in loss_dict.items():
            total_loss_dict[item] += value
        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.5f}  lr:{:.6f}'
                    .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0]))
            for item, value in total_loss_dict.items():
                recoder.print_log(f'\tMean {item} loss: {value/recoder.log_interval:.5f}')
            for j in range(action_num):
                recoder.print_log(f'\t ratio of action {j} : {action_count[j]/sum(action_count):.5f}')
            total_loss_dict = {}
            for k in loss_weights.keys(): 
                total_loss_dict[k] = 0
            action_count = [0 for i in range(action_num)]
        del ret_dict
        del loss
    optimizer.scheduler.step()
    recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
    return loss_value


def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder):
    model.eval()
    total_sent = []
    total_info = []
    action_num = 7
    action_count = [0 for i in range(action_num)]
    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_mid = device.data_to_device(data[1])
        vid_small = device.data_to_device(data[2])
        vid_lgt = device.data_to_device(data[3])
        label = device.data_to_device(data[4])
        label_lgt = device.data_to_device(data[5])
        with torch.no_grad():
            ret_dict = model(vid, vid_mid, vid_small, vid_lgt, label=label, label_lgt=label_lgt)

        for i in range(vid.size(0)):
            for j in range(ret_dict['action'].size(1)):
                if ret_dict['action'][i,j]:
                    if j>=0 and j<4:
                        j=0
                        break
                    elif j>=4 and j<6:
                        j=1
                        break
                    elif j==6 :
                        j=2
                        break
                    if j>=7 and j<11:
                        j=3
                        break
                    elif j>=11 and j<13:
                        j=4
                        break
                    elif j==13 :
                        j=5
                        break
                    elif j==14 :
                        j=6
                        break
            action_count[j] = action_count[j] + 1
        total_info += [file_name.split("|")[0] for file_name in data[-1]]
        total_sent += ret_dict['recognized_sents']
    try:
        write2file(work_dir + "output-hypothesis-{}.ctm".format(mode), total_info, total_sent)
        ret = evaluate(prefix=work_dir, mode=mode, output_file="output-hypothesis-{}.ctm".format(mode),
                       evaluate_dir=cfg.dataset_info['evaluation_dir'],
                       evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
                       output_dir="epoch_{}_result/".format(epoch), python_evaluate=cfg.python_evaluate)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        ret = "Percent Total Error       =  100.00%   (ERROR)"
        return ret
    finally:
        pass
    recoder.print_log("Epoch {}, {} {}".format(epoch, mode, ret))
    recoder.print_log("Epoch {}, {} {}".format(epoch, mode, ret),
                      '{}/{}.txt'.format(work_dir, mode))
    for j in range(action_num):
        recoder.print_log(f'\t ratio of action {j} : {action_count[j]/sum(action_count):.5f}')
    return ret

def seq_eval_separate(cfg, loader, model, device, mode, epoch, work_dir, recoder):
    model.eval()
    total_sent = []
    total_info = []
    action_num = 8
    action_count = [0 for i in range(action_num)]
    action_classes = 4
    action_sent = {}
    action_info = {}
    for i in range(action_classes):
        action_sent[i] = []
        action_info[i] = []
    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_small = device.data_to_device(data[1])
        vid_lgt = device.data_to_device(data[2])
        label = device.data_to_device(data[3])
        label_lgt = device.data_to_device(data[4])
        with torch.no_grad():
            ret_dict = model(vid, vid_small, vid_lgt, label=label, label_lgt=label_lgt)

        for i in range(vid.size(0)):
            for j in range(ret_dict['action'].size(1)):
                if ret_dict['action'][i,j]:
                    action_count[j] = action_count[j] + 1
        action_indice = np.argwhere(ret_dict['action'].detach().cpu().numpy()>0) #b2
        for i in range(vid.size(0)):
            if action_indice[i,1]>=0 and action_indice[i,1]<4:
                action_info[0] += [ data[-1][i].split("|")[0] ] 
                action_sent[0] += ret_dict['recognized_sents'][i:i+1]
            elif action_indice[i,1]>=4 and action_indice[i,1]<6:
                action_info[1] += [ data[-1][i].split("|")[0] ]
                action_sent[1] += ret_dict['recognized_sents'][i:i+1]
            elif action_indice[i,1] == 6:
                action_info[2] += [ data[-1][i].split("|")[0] ]
                action_sent[2] += ret_dict['recognized_sents'][i:i+1]
            elif action_indice[i,1] == 7:
                action_info[3] += [ data[-1][i].split("|")[0] ]
                action_sent[3] += ret_dict['recognized_sents'][i:i+1]
        total_info += [file_name.split("|")[0] for file_name in data[-1]]
        total_sent += ret_dict['recognized_sents']
    try:
        for i in range(action_classes):
            write2file(work_dir + "output-hypothesis-{}-action-{}.ctm".format(mode,i), action_info[i], action_sent[i])
        write2file(work_dir + "output-hypothesis-{}.ctm".format(mode), total_info, total_sent)
        for i in range(action_classes):
            ret = evaluate(prefix=work_dir, mode=mode, output_file="output-hypothesis-{}-action-{}.ctm".format(mode,i),
                       evaluate_dir=cfg.dataset_info['evaluation_dir'],
                       evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
                       output_dir="epoch_{}_result/".format(epoch),python_evaluate=True, evaluated_files=action_info[i])
            recoder.print_log("WER of action {}: {}".format(i, ret))
        ret = evaluate(prefix=work_dir, mode=mode, output_file="output-hypothesis-{}.ctm".format(mode),
                       evaluate_dir=cfg.dataset_info['evaluation_dir'],
                       evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
                       output_dir="epoch_{}_result/".format(epoch))
    except:
        print("Unexpected error:", sys.exc_info()[0])
        ret = "Percent Total Error       =  100.00%   (ERROR)"
        return ret
    finally:
        pass
    recoder.print_log("Epoch {}, {} {}".format(epoch, mode, ret))
    recoder.print_log("Epoch {}, {} {}".format(epoch, mode, ret),
                      '{}/{}.txt'.format(work_dir, mode))
    for j in range(ret_dict['action'].size(1)):
        recoder.print_log(f'\t ratio of action {j} : {action_count[j]/sum(action_count):.5f}')
    return ret

def seq_feature_generation(loader, model, device, mode, work_dir, recoder):
    model.eval()

    src_path = os.path.abspath(f"{work_dir}{mode}")
    tgt_path = os.path.abspath(f"./features/{mode}")

    if os.path.islink(tgt_path):
        curr_path = os.readlink(tgt_path)
        if work_dir[1:] in curr_path and os.path.isabs(curr_path):
            return
        else:
            os.unlink(tgt_path)
    else:
        if os.path.exists(src_path) and len(loader.dataset) == len(os.listdir(src_path)):
            os.symlink(src_path, tgt_path)
            return

    for batch_idx, data in tqdm(enumerate(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt)
        if not os.path.exists(src_path):
            os.makedirs(src_path)
        start = 0
        for sample_idx in range(len(vid)):
            end = start + data[3][sample_idx]
            filename = f"{src_path}/{data[-1][sample_idx].split('|')[0]}_features.npy"
            save_file = {
                "label": data[2][start:end],
                "features": ret_dict['framewise_features'][sample_idx][:, :vid_lgt[sample_idx]].T.cpu().detach(),
            }
            np.save(filename, save_file)
            start = end

        os.symlink(src_path, tgt_path)
        assert end == len(data[2])


def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
                                                 word_idx * 1.0 / 100,
                                                 (word_idx + 1) * 1.0 / 100,
                                                 word[0]))
