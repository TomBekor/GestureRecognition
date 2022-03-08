from enum import unique
from operator import index
import numpy as np
import cv2 as cv
import torch
import os
from tqdm import tqdm

class ToolBatchGenerator:

    def __init__(self, num_classes_tools, actions_dict_tools, videos_path,
                gt_path_tools_left, gt_path_tools_right,
                batch_size, sample_rate, folds_folder, split_num):

        self.num_classes_tools = num_classes_tools
        self.actions_dict_tools = actions_dict_tools
        self.videos_path = videos_path
        self.gt_path_tools_left = gt_path_tools_left
        self.gt_path_tools_right = gt_path_tools_right
        self.batch_size = batch_size
        self.data_size = 0
        self.index = 0
        self.all_clean_vids = []
        self.sample_rate = sample_rate

        self.folds_folder = folds_folder
        self.split_num = split_num

        print('\nLoading image data...')

        self.read_data()

        self.X_train_top, self.X_train_side, self.y_train_right, self.y_train_left, self.train_size = self.select_files(mode='train')
        self.X_valid_top, self.X_valid_side, self.y_valid_right, self.y_valid_left, self.valid_size = self.select_files(mode='valid')

        self.number_of_batches = self.train_size // self.batch_size

    def select_files(self, mode):
        '''
        for modes 'validation'/'train' reutrns:
        - top-frames paths list
        - side-frames paths list
        - right-hand tags
        - left-hand tags
        - data size
        '''
        if mode=='train':
            vid_list = self.train_files
        else:
            vid_list = self.valid_files
        vid_list = self.clean_vid_list(vid_list)
        self.all_clean_vids += list(vid_list)

        top_image_paths = []
        side_image_paths = []
        right_tags = []
        left_tags = []
        
        for vid_path in tqdm(vid_list):
            # Load labels
            vid_right_tags = self.generate_tags(vid_path, 'right')
            vid_left_tags = self.generate_tags(vid_path, 'left')
            
            # Load videos
            full_vid_path = os.path.join(self.videos_path, vid_path)
            top_videos = np.array(os.listdir(full_vid_path + '_top'))
            n_top_videos = top_videos.size
            side_videos = np.array(os.listdir(full_vid_path + '_side'))
            n_side_videos = side_videos.size

            # Equalify data
            min_vid_len = min(n_top_videos, n_side_videos)
            min_tags_len = min(len(vid_right_tags), len(vid_left_tags))
            min_len = min(min_vid_len, min_tags_len)
            vid_right_tags = vid_right_tags[:min_len]
            vid_left_tags = vid_left_tags[:min_len]
            top_videos = top_videos[:min_len]
            side_videos = side_videos[:min_len]

            # Add path suffix
            video_images_top = [os.path.join(full_vid_path + '_top', img) for img in top_videos]
            video_images_side = [os.path.join(full_vid_path + '_side', img) for img in side_videos]

            top_image_paths += video_images_top
            side_image_paths += video_images_side
            right_tags += vid_right_tags
            left_tags += vid_left_tags

        # Trim data and create Tensor dataset
        indexes_subset = self.select_indexes(left_tags, right_tags)[::self.sample_rate]
        top_image_paths = np.array(top_image_paths)[indexes_subset]
        side_image_paths = np.array(side_image_paths)[indexes_subset]
        right_tags = np.array(right_tags)[indexes_subset]
        left_tags = np.array(left_tags)[indexes_subset]
        data_size = indexes_subset.size
        
        # Shuffle data
        random_indexes = np.arange(data_size)
        np.random.shuffle(random_indexes)
        top_image_paths = top_image_paths[random_indexes]
        side_image_paths = side_image_paths[random_indexes]
        left_tags = left_tags[random_indexes]
        right_tags = right_tags[random_indexes]
        right_tags, left_tags = torch.Tensor(right_tags).type(torch.long), torch.Tensor(left_tags).type(torch.long)

        return top_image_paths, side_image_paths, right_tags, left_tags, data_size

    def read_data(self):
        self.train_files =[]
        for file in os.listdir(self.folds_folder):
            filename = os.fsdecode(file)
            if filename.endswith(".txt") and "fold" in filename:
                if str(self.split_num) in filename:
                    file_ptr = open(os.path.join(self.folds_folder, filename), 'r')
                    self.valid_files = file_ptr.read().split('\n')[:-1]
                    file_ptr.close()
                else:
                    file_ptr = open(os.path.join(self.folds_folder, filename), 'r')
                    self.train_files = self.train_files + file_ptr.read().split('\n')[:-1]
                    file_ptr.close()

    def clean_vid_list(self, vid_list):
        clean_list = [video_path.split('.')[0] for video_path in vid_list]
        return np.unique(clean_list)

    def reset(self):
        '''
        resets self.index and shuffles the data
        '''
        self.index = 0
        random_indexes = np.random.shuffle(np.arange(self.train_size))
        self.X_train_top = self.X_train_top[random_indexes]
        self.X_train_side = self.X_train_side[random_indexes]
        self.y_train_left = self.y_train_left[random_indexes]
        self.y_train_right = self.y_train_right[random_indexes]

    def has_next(self):
        if self.index < self.train_size:
            return True
        return False

    def process_image(self, image_path):
        '''
        Reads image from image_path in RGB format.
        '''    
        frame = cv.imread(image_path)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = np.moveaxis(frame, -1 , 0)
        return frame

    def parse_ground_truth(self, gt_source):
        content = []
        for line in gt_source:
            info = line.split()
            line_content = [int(info[2][1])] * (int(info[1])-int(info[0]) +1)
            content = content + line_content
        return content   

    def generate_tags(self, video_path, hand):
        if hand == 'right':
            gt_path = self.gt_path_tools_right
        else:
            gt_path = self.gt_path_tools_left
        file_ptr = open(gt_path + video_path + '.txt', 'r')
        gt_source = file_ptr.read().split('\n')[:-1]
        content = self.parse_ground_truth(gt_source)
        return content

    def select_indexes(self, arr_left, arr_right):
        '''
        Balances the dataset so there will be an equal
        number of tool classes.
        '''
        arr_left, arr_right = np.array(arr_left), np.array(arr_right)
        arr_left = arr_left[:min(arr_left.size, arr_right.size)]
        arr_right = arr_right[:min(arr_left.size, arr_right.size)]
        unique_left, counts_left = np.unique(arr_left, return_counts=True)
        unique_right, counts_right = np.unique(arr_right, return_counts=True)
        min_count_left = np.min(counts_left)
        min_count_right = np.min(counts_right)
        idxes = []
        for tag in unique_left:
            tag_indexes = np.where(arr_left == tag)[0]
            tag_indexes = np.random.choice(tag_indexes, min_count_left, replace=False)
            idxes += list(tag_indexes)
        for tag in unique_right:
            tag_indexes = np.where(arr_right == tag)[0]
            tag_indexes = np.random.choice(tag_indexes, min_count_right, replace=False)
            idxes += list(tag_indexes)
        return np.unique(idxes)

    def process_data(self, tops, sides, rights, lefts):
        '''
        Loads top and side images.
        args:
        tops - top image paths
        sides - side image paths
        '''
        if type(tops) != np.ndarray:
            tops = np.array([tops])
            sides = np.array([sides])
        batch_inputs_top = np.array([self.process_image(img_path) for img_path in tops])
        batch_inputs_side = np.array([self.process_image(img_path) for img_path in sides])
        batch_inputs_top = torch.from_numpy(batch_inputs_top).type(torch.float)
        batch_inputs_side = torch.from_numpy(batch_inputs_side).type(torch.float)
        return [batch_inputs_top, batch_inputs_side, rights, lefts]

    def next_batch(self):
        '''
        Returns the next batch, with self.batch_size samples.
        '''
        if not self.has_next():
            self.reset()
        batch_top_images = self.X_train_top[self.index: self.index + self.batch_size]
        batch_side_images = self.X_train_side[self.index: self.index + self.batch_size]
        batch_right_tags = self.y_train_right[self.index: self.index + self.batch_size]
        batch_left_tags = self.y_train_left[self.index: self.index + self.batch_size]
        self.index += self.batch_size
        return self.process_data(batch_top_images, batch_side_images, batch_right_tags, batch_left_tags)
