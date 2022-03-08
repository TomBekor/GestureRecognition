#!/usr/bin/python2.7

import torch
import numpy as np
import random
import os
import pandas as pd
from scipy.stats import norm

class BatchGenerator(object):
    def __init__(self, num_classes_gestures, num_classes_tools,
                actions_dict_gestures, actions_dict_tools,
                features_path, tools_outs_path, tools_reps_path, split_num, folds_folder,
                gt_path_gestures=None, gt_path_tools_left=None,
                gt_path_tools_right=None, sample_rate=1,
                normalization="None", kinematics='raw', task="gestures"):
        self.task =task
        self.normalization = normalization
        self.kinematics = kinematics
        self.folds_folder = folds_folder
        self.split_num = split_num
        self.list_of_train_examples = list()
        self.list_of_valid_examples = list()
        self.index = 0
        self.num_classes_gestures = num_classes_gestures
        self.num_classes_tools = num_classes_tools
        self.actions_dict_gestures= actions_dict_gestures
        self.action_dict_tools = actions_dict_tools
        self.gt_path_gestures = gt_path_gestures
        self.gt_path_tools_left = gt_path_tools_left
        self.gt_path_tools_right = gt_path_tools_right
        self.features_path = features_path
        self.tools_outs_path = tools_outs_path
        self.tools_reps_path = tools_reps_path
        self.sample_rate = sample_rate
        self.read_data()
        self.normalization_params_read()

    def normalization_params_read(self):
        params = pd.read_csv(os.path.join(self.folds_folder, "std_params_fold_" + str(self.split_num) + ".csv"),index_col=0).values
        self.max = params[0, :]
        self.min = params[1, :]
        self.mean = params[2, :]
        self.std = params[3, :]

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_train_examples)

    def has_next(self):
        if self.index < len(self.list_of_train_examples):
            return True
        return False

    def read_data(self):
        self.list_of_train_examples =[]
        for file in os.listdir(self.folds_folder):
            filename = os.fsdecode(file)
            if filename.endswith(".txt") and "fold" in filename:
                if str(self.split_num) in filename:
                    file_ptr = open(os.path.join(self.folds_folder, filename), 'r')
                    self.list_of_valid_examples = file_ptr.read().split('\n')[:-1]
                    file_ptr.close()
                    random.shuffle(self.list_of_valid_examples)
                else:
                    file_ptr = open(os.path.join(self.folds_folder, filename), 'r')
                    self.list_of_train_examples = self.list_of_train_examples + file_ptr.read().split('\n')[:-1]
                    file_ptr.close()
                continue
            else:
                continue
        random.shuffle(self.list_of_train_examples)

    def parse_ground_truth(self,gt_source):
        contant =[]
        for line in gt_source:
            info = line.split()
            line_contant = [info[2]] * (int(info[1])-int(info[0]) +1)
            contant = contant + line_contant
        return contant

    def normalize_features(self, features):
        if self.normalization == "Min-max":
            numerator = features.T - self.min
            denominator = self.max-self.min
            features = (numerator / denominator).T
        elif self.normalization == "Standard":
            numerator = features.T - self.mean
            denominator = self.std
            features = (numerator / denominator).T
        elif self.normalization == "samplewise_SD":
            samplewise_meam = features.mean(axis=1)
            samplewise_std = features.std(axis=1)
            numerator = features.T - samplewise_meam
            denominator = samplewise_std
            features = (numerator / denominator).T
        return features
    
    def calc_kinematics(self, features):
        '''
        Gets kinematics features, finds its velocity / acceleration,
        aggregates the new features to the original ones and returns them.
        '''
        if self.kinematics == "velocity":
            velocity = np.diff(features, axis=1)
            features = np.concatenate((features[:,:-1], velocity), axis=0)
        elif self.kinematics == "acceleration":
            velocity = np.diff(features, axis=1)
            acceleration = np.diff(velocity, axis=1)
            features = np.concatenate((features[:,:-2], velocity[:,:-1], acceleration), axis=0)
        return features

    def next_batch(self, batch_size):
        batch = self.list_of_train_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target_gestures = []

        for seq in batch:
            seq_npy_file = seq.split('.')[0] + '.npy'

            # kinematic features
            kin_features = np.load(self.features_path + seq_npy_file)
            normalized_kin_features = self.normalize_features(kin_features)
            normalized_kin_features = self.calc_kinematics(normalized_kin_features)
            normalized_kin_features = normalized_kin_features[:, ::self.sample_rate]

            # load tools model outputs and vector representations
            tools_outputs = np.load(self.tools_outs_path + seq_npy_file) # seq_len, 2, 4
            tools_reps = torch.Tensor(np.load(self.tools_reps_path + seq_npy_file)) # seq_len, 1000
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

            batch_input.append(concatenated_input)

            file_ptr = open(self.gt_path_gestures + seq.split('.')[0] + '.txt', 'r')
            gt_source = file_ptr.read().split('\n')[:-1]
            content = self.parse_ground_truth(gt_source)
            classes_size = min(np.shape(kin_features)[1], len(content))

            classes = np.zeros(classes_size)
            for i in range(len(classes)):
                classes[i] = self.actions_dict_gestures[content[i]]
            batch_target_gestures.append(classes[::self.sample_rate])
        
        # seq masking
        length_of_sequences = list(map(len, batch_target_gestures))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes_gestures, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i][:,:batch_input_tensor.shape[2]])
            batch_target_tensor[i, :np.shape(batch_target_gestures[i])[0]] = torch.from_numpy(batch_target_gestures[i])
            mask[i, :, :np.shape(batch_target_gestures[i])[0]] = torch.ones(self.num_classes_gestures, np.shape(batch_target_gestures[i])[0])

        return batch_input_tensor, batch_target_tensor, mask