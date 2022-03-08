#Created by Adam Goldbraikh - Scalpel Lab Technion
# parts of the code were adapted from: https://github.com/sj-li/MS-TCN2?utm_source=catalyzex.com

from model import *
import sys
from torch import optim
import math
import pandas as pd
from termcolor import colored, cprint

from metrics import *
import wandb
from datetime import datetime
import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau



class Trainer:
    def __init__(self, dim, num_classes_list,tools_outs_path,tools_reps_path,hidden_dim=64,dropout=0.4,num_layers=3, offline_mode=True, task="gestures", device="cuda",
                 network='LSTM',debugging=False):

        self.model = MT_RNN_dp(network, input_dim=dim, hidden_dim=hidden_dim, num_classes_list=num_classes_list,
                            bidirectional=offline_mode, dropout=dropout,num_layers=num_layers)

        self.debugging =debugging
        self.network = network
        self.device = device
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.num_classes_list = num_classes_list
        self.task = task
        self.tools_outs_path = tools_outs_path
        self.tools_reps_path = tools_reps_path
        self.tau = 4
        self.lamda = 0.15

        self.model.load_state_dict(torch.load('/home/student/Adams/shell_code_surgical_data_science/final_model/03 16:10 TwoInToolsOut + standard norm + scheduler + OS-loss(0.15,4) models/split_0/epoch-40.model'))
        self.model.to(device)

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, eval_dict, args):

        number_of_seqs = len(batch_gen.list_of_train_examples)
        number_of_batches = math.ceil(number_of_seqs / batch_size)

        eval_results_list = []
        train_results_list = []
        print(args.dataset + " " + args.group + " " + args.dataset + " dataset " + "split: " + args.split)

        if args.upload is True:
            wandb.login()
            wandb.init(project=args.project,
                       group=args.group,
                       name="split: " + args.split,
                       reinit=True)
            delattr(args, 'split')
            wandb.config.update(args)

        self.model.train()
        self.model.to(self.device)
        eval_rate = eval_dict["eval_rate"]
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)


        # ReduceLROnPlateau scheduler
        mode='max'
        factor=0.5
        patience=5
        scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, verbose=True)

        for epoch in range(num_epochs):
            pbar = tqdm.tqdm(total=number_of_batches)
            epoch_loss = 0
            correct1 = 0
            total1 = 0

            while batch_gen.has_next():
                batch_input, batch_target_gestures, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target_gestures, mask = batch_input.to(self.device), batch_target_gestures.to(
                    self.device), mask.to(self.device)

                # Classification Cross-Entropy Loss
                optimizer.zero_grad()
                lengths = torch.sum(mask[:, 0, :], dim=1).to(dtype=torch.int64).to(device='cpu')
                predictions1 = self.model(batch_input, lengths)
                predictions1 = (predictions1[0] * mask)

                classification_loss = self.ce(predictions1.transpose(2, 1).contiguous().view(-1, self.num_classes_list[0]),
                                              batch_target_gestures.view(-1))

                # Over-Segmentation Loss
                T_MSE_loss = 0
                predictions1 = predictions1.permute(0,2,1)
                probs = torch.softmax(predictions1, dim=2)
                log_probs = torch.log(probs)
                diff_probs = log_probs[:,1:,:] - log_probs[:,:-1,:].detach()
                abs_probs = torch.abs(diff_probs)
                taus = torch.ones(abs_probs.size()) * self.tau
                tau_probs = torch.maximum(abs_probs, taus.to(self.device))
                square_probs = torch.square(tau_probs)

                for seq_idx, seq_probs in enumerate(square_probs):
                    seq_len = torch.where(mask[seq_idx].permute(1,0)[:,0] == 1)[0][-1] + 1
                    num_classes = square_probs.size(2)
                    seq_sum = torch.sum(seq_probs)
                    T_MSE_loss += (1/(seq_len*num_classes)) * seq_sum

                loss = classification_loss + self.lamda * T_MSE_loss
                predictions1 = predictions1.permute(0,2,1)

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                _, predicted1 = torch.max(predictions1.data, 1)
                for i in range(len(lengths)):
                    correct1 += (predicted1[i][:lengths[i]] == batch_target_gestures[i][
                                                               :lengths[i]]).float().sum().item()
                    total1 += lengths[i]
                
                pbar.update(1)

            batch_gen.reset()
            pbar.close()
            if not self.debugging:
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            print(colored(dt_string, 'green',
                          attrs=['bold']) + "  " + "[epoch %d]: train loss = %f,   train acc = %f" % (epoch + 1,
                                                                                                      epoch_loss / len(
                                                                                                          batch_gen.list_of_train_examples),
                                                                                                      float(
                                                                                                          correct1) / total1))
            train_results = {"epoch": epoch, "train loss": epoch_loss / len(batch_gen.list_of_train_examples),
                             "train acc": float(correct1) / total1}

            scheduler.step(float(correct1) / total1)

            if args.upload:
                wandb.log(train_results)

            train_results_list.append(train_results)

            if (epoch) % eval_rate == 0:
                print(colored("epoch: " + str(epoch + 1) + " model evaluation", 'red', attrs=['bold']))
                results = {"epoch": epoch}
                results.update(self.evaluate(eval_dict, batch_gen))
                eval_results_list.append(results)
                if args.upload is True:
                    wandb.log(results)
            
        return eval_results_list, train_results_list

    def evaluate(self, eval_dict, batch_gen):
        results = {}
        device = eval_dict["device"]
        features_path = eval_dict["features_path"]
        sample_rate = eval_dict["sample_rate"]
        actions_dict_gesures = eval_dict["actions_dict_gestures"]
        ground_truth_path_gestures = eval_dict["gt_path_gestures"]

        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            list_of_vids = batch_gen.list_of_valid_examples
            recognition1_list = []

            for seq in list_of_vids:
                seq_npy_file = seq.split('.')[0] + '.npy'
                # kinematic features
                kin_features = np.load(features_path + seq_npy_file)
                normalized_kin_features = batch_gen.normalize_features(kin_features)
                normalized_kin_features = batch_gen.calc_kinematics(normalized_kin_features)
                normalized_kin_features = normalized_kin_features[:, ::sample_rate]

                # tools predicted labels
                tools_outputs = np.load(self.tools_outs_path + seq_npy_file) # seq_len, 2, 4
                tools_reps = torch.Tensor(np.load(self.tools_reps_path + seq_npy_file))
                tools_probs = torch.softmax(torch.Tensor(tools_outputs), dim=2).view(-1,8)

                # With Representations:
                # tools_features = torch.cat([tools_probs, tools_reps], dim=1)
                # Without Representations:
                tools_features = tools_probs

                tools_features = tools_features.permute(1,0)
                tools_features = np.array(tools_features)

                # same seq len
                min_len = min(normalized_kin_features.shape[1], tools_features.shape[1])
                normalized_kin_features = normalized_kin_features[:,:min_len]
                tools_features = tools_features[:,:min_len]

                concatenated_input = np.concatenate([tools_features, normalized_kin_features], axis=0)

                input_x = torch.tensor(concatenated_input, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)

                predictions1 = self.model(input_x, torch.tensor([concatenated_input.shape[1]]))
                predictions1 = predictions1[0].unsqueeze_(0)
                predictions1 = torch.nn.Softmax(dim=2)(predictions1)

                _, predicted1 = torch.max(predictions1[-1].data, 1)
                predicted1 = predicted1.squeeze()


                recognition1 = []
                for i in range(len(predicted1)):
                    recognition1 = np.concatenate((recognition1, [list(actions_dict_gesures.keys())[
                                                                      list(actions_dict_gesures.values()).index(
                                                                          predicted1[i].item())]] * sample_rate))
                recognition1_list.append(recognition1)

            print("gestures results")
            results1, _ = metric_calculation(ground_truth_path=ground_truth_path_gestures,
                                             recognition_list=recognition1_list, list_of_videos=list_of_vids,
                                             suffix="gesture")
            results.update(results1)


            self.model.train()
            return results