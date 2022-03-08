from builtins import float
import torch
from torch import nn, optim
from tools_batch_gen import ToolBatchGenerator
from tools_model import TwoInToolsOut
from tqdm import tqdm
import argparse, os
import numpy as np
from multiprocessing import Pool
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',choices=['APAS'], default="APAS")
parser.add_argument('--num_epochs', default = 1, type=int)
parser.add_argument('--batch_size',default = 4, type=int)
parser.add_argument('--valid_iters',default = 500, type=int)
args = parser.parse_args()

sample_rate = 6

mapping_tool_file = "/datashare/"+args.dataset+"/mapping_tools.txt"

folds_folder = "/datashare/"+args.dataset+"/folds"
videos_path = '/datashare/'+args.dataset+'/frames/'
gt_path_tools_left = "/datashare/"+args.dataset+"/transcriptions_tools_left_new/"
gt_path_tools_right = "/datashare/"+args.dataset+"/transcriptions_tools_right_new/"

epochs = args.num_epochs
batch_size = args.batch_size
valid_iters = args.valid_iters

num_classes_tools = 0
actions_dict_tools = dict()
if args.dataset == "APAS":
    file_ptr = open(mapping_tool_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    for a in actions:
        actions_dict_tools[a.split()[1]] = int(a.split()[0])
    num_classes_tools = len(actions_dict_tools)

seed = 1538574472
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

class ToolsTrainer:

    def __init__(self, num_classes_tools, actions_dict_tools,
                videos_path, gt_path_tools_left, 
                gt_path_tools_right, sample_rate, batch_size, 
                epochs, validate_iters,
                pretrained_path, folds_folder, split_num):
        
        # General
        self.num_classes_tools = num_classes_tools
        self.actions_dict_tools = actions_dict_tools
        self.videos_path = videos_path
        self.gt_path_tools_left = gt_path_tools_left
        self.gt_path_tools_right = gt_path_tools_right
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = sample_rate

        self.folds_folder = folds_folder
        self.split_num = split_num

        # Model initialization
        self.model = TwoInToolsOut(self.num_classes_tools)
        if pretrained_path:
            self.model.load_state_dict(torch.load(pretrained_path))
        self.model = self.model.to(self.device)

        # Training settings
        self.batch_size = batch_size
        self.epochs = epochs
        self.validate_iters = validate_iters
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_func = nn.CrossEntropyLoss()

        # Batch generator
        self.batch_gen = ToolBatchGenerator(
            num_classes_tools, actions_dict_tools,
            videos_path, gt_path_tools_left,
            gt_path_tools_right, batch_size,
            sample_rate, folds_folder, split_num)

    def train(self):
        '''
        TwoInToolsOut model training.
        '''
        for epoch in range(self.epochs):
            print(f'[Epoch {epoch+1}]:')
            pbar = tqdm(total=self.batch_gen.number_of_batches)
            tot_loss = 0
            iters = 0
            while self.batch_gen.has_next():
                iters += 1
                batch_data = self.batch_gen.next_batch()
                for idx, data in enumerate(batch_data):
                    batch_data[idx] = data.to(self.device)
                X_top, X_side, y_right, y_left = batch_data
                self.optimizer.zero_grad()
                outs_right, outs_left, _ = self.model(X_top, X_side)
                loss_right = self.loss_func(outs_right, y_right)
                loss_left = self.loss_func(outs_left, y_left)
                loss = loss_right + loss_left
                loss.backward()
                tot_loss += loss.item()
                # if iters % self.validate_iters == 0:
                #     valid_acc = self.evaluate(mode='validation')
                #     print(f'Loss: {tot_loss:.3f} | Validation accuracy: {(valid_acc*100):.3f}%')
                #     tot_loss = 0
                #     self.model.train()
                self.optimizer.step()
                pbar.update(1)
            pbar.close()
            print(f'Epoch loss: {tot_loss}')
            self.batch_gen.reset()

    def evaluate(self, mode='validation'):
        '''
        TwoInToolsOut model evalutation
        '''
        with torch.no_grad():
            self.model.eval()
            bg = self.batch_gen
            correct = 0
            if mode == 'validation':
                total = bg.valid_size
                X_t = bg.X_valid_top
                X_s = bg.X_valid_side
                y_r = bg.y_valid_right
                y_l = bg.y_valid_left
            else:
                raise TypeError
            print(f'Evaluating...')
            for x_top, x_side, y_right, y_left in tqdm(list(zip(X_t, X_s, y_r, y_l))):
                x_top, x_side, y_right, y_left = bg.process_data(x_top, x_side, y_right, y_left)
                x_top, x_side, y_right, y_left = x_top.to(self.device), x_side.to(self.device), y_right.to(self.device), y_left.to(self.device)
                outs_right, outs_left, _ = self.model(x_top, x_side)
                preds_right, preds_left = torch.argmax(outs_right, dim=1), torch.argmax(outs_left, dim=1)
                correct += float((preds_right == y_right).cpu()/2)
                correct += float((preds_left == y_left).cpu()/2)
        acc = correct / total
        return acc

    def _image_infer(self, frames_tup):
        '''
        infer on single frame
        args:
        frames_tup - (top_frame, side_frame)
        '''
        top_tensor = torch.Tensor(frames_tup[0]).type(torch.float).to(self.device).unsqueeze(0)
        side_tensor = torch.Tensor(frames_tup[1]).type(torch.float).to(self.device).unsqueeze(0)
        pred_right, pred_left, rep_vec = self.model(top_tensor, side_tensor)
        pred_right, pred_left, rep_vec = pred_right.squeeze().cpu(), pred_left.squeeze().cpu(), rep_vec.squeeze().cpu()
        pred_right, pred_left, rep_vec = np.array(pred_right), np.array(pred_left), np.array(rep_vec)
        preds = np.array([pred_right, pred_left])
        return preds, rep_vec

    def inference(self, labels_save_dir, reps_save_dir):
        '''
        Inference over the whole dataset.
        For each video, saves for each frame the model outputs and last layer values.
        args:
        labels_save_dir: directory path to save model outputs.
        reps_save_dir: directory path to save model last layer.
        '''
        clean_full_vid_paths = [os.path.join(self.videos_path, vid_name) for vid_name in self.batch_gen.all_clean_vids]
        os.makedirs(labels_save_dir, exist_ok=True)
        os.makedirs(reps_save_dir, exist_ok=True)
        self.model.eval()
        with torch.no_grad():
            for vid_path in tqdm(clean_full_vid_paths):
                top_frames_paths = sorted(glob(vid_path + '_top/*'))[::self.sample_rate]
                side_frames_paths = sorted(glob(vid_path + '_side/*'))[::self.sample_rate]
                min_len = min(len(top_frames_paths), len(side_frames_paths))
                top_frames_paths = top_frames_paths[:min_len]
                side_frames_paths = side_frames_paths[:min_len]
                
                with Pool() as p:
                    top_frames = p.map(self.batch_gen.process_image, top_frames_paths)
                
                with Pool() as p:
                    side_frames = p.map(self.batch_gen.process_image, side_frames_paths)

                vid_tup = list(map(self._image_infer, zip(top_frames, side_frames)))
                labels_arr = np.array([t[0] for t in vid_tup])
                reps_arr = np.array([t[1] for t in vid_tup])
                np.save(os.path.join(labels_save_dir, vid_path.split('/')[-1]), labels_arr)
                np.save(os.path.join(reps_save_dir, vid_path.split('/')[-1]), reps_arr)
def main():

    from_pretrained = True
    dataset_inference = True

    for split_num in range(0,1):

        print(f'\n\n# ----------- Split {split_num} ----------- #')

        model_save_path = f'/home/student/Adams/shell_code_surgical_data_science/models/TwoInToolsOut_split_{split_num}'
        pretrained_path = f'/home/student/Adams/shell_code_surgical_data_science/models/TwoInToolsOut_split_{split_num}' if from_pretrained else None

        trainer = ToolsTrainer(
            num_classes_tools, actions_dict_tools,
            videos_path, gt_path_tools_left,
            gt_path_tools_right, sample_rate, batch_size,
            epochs, valid_iters, pretrained_path=pretrained_path,
            folds_folder=folds_folder, split_num=split_num)

        if not pretrained_path:
            trainer.train()
            torch.save(trainer.model.state_dict(), model_save_path)

        # final_valid_acc = trainer.evaluate(mode='valid')
        # print(f'Split {split_num} final validation accuracy: {round(final_valid_acc*100, 2)}%\n\n')

        if dataset_inference:
            trainer.inference(labels_save_dir=f'ToolsPredictions_split_{split_num}',
                              reps_save_dir=f'ToolsReps_split_{split_num}')

if __name__=='__main__':
    main()