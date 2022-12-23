import os
import numpy as np
import torch
from MSPN import MSPNet
import datetime
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from utils import util_of_lpips, MyDataset
import random
os.environ['TORCH_HOME'] = '/home/Ling.cf/HHD/conda/miniconda3/torch-model'
'''
Trained with only pixel (deterministic) loss 
'''

class PixTraining():
    def __init__(self):
        self.dataset = 'Human3.6M'
        self.AddInfo = 'None'        # additional instructions
        self.train_data_path = '/mnt/DevFiles_NAS/Ling.cf/dataset/{}/train'.format(self.dataset)
        self.val_data_path = '/mnt/DevFiles_NAS/Ling.cf/dataset/{}/test2'.format(self.dataset)
        self.tag = 'MSPN_{}_{}.pt'.format(self.dataset, self.AddInfo)      # saving tag
        self.retrain = True           # continue training or not
        self.num_epoch = 40           # training epoch
        self.len_input = 10           # length of input sequence (real frames)
        self.PredSteps = 10           # Predict the future frames for the next few steps (predictive frames as input)

    def val_model(self, model):
        model.eval()
        all_ssim = []
        all_psnr = []
        all_lpips = []
        lpips_loss = util_of_lpips('alex')

        with torch.no_grad():
            for curdir, dirs, files in os.walk(self.val_data_path):
                if len(files) == 0:
                    continue
                files.sort()
                for file in files:
                    cur_path = os.path.join(curdir, file)
                    val_dataset = MyDataset(path=cur_path, len_input=self.len_input + self.PredSteps)
                    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, drop_last=True)
                    for data in val_dataloader:
                        inputs = torch.true_divide(data[:, :self.len_input], 255).to(0)
                        targets = torch.true_divide(data[:, self.len_input:], 255)
                        ssim_score = []
                        psnr_score = []
                        lpips_score = []
                        predictions, error = model(inputs, pred_steps=self.PredSteps, mode='test')
                        for t in range(self.PredSteps):
                            target = targets[:, t].to(0)
                            predict = predictions[t]
                            lpips = lpips_loss.calc_lpips(predict, target)
                            lpips_score.append(lpips.mean().item())
                            target = target.squeeze().cpu().numpy()
                            predict = predict.data.cpu().numpy().squeeze()
                            tmp_psnr = []
                            tmp_ssim = []
                            for j in range(target.shape[0]):
                                target_j = target[j, :]
                                predict_j = predict[j, :]
                                if len(target_j.shape) > 2:
                                    target_j = np.transpose(target_j, (1, 2, 0))
                                    predict_j = np.transpose(predict_j, (1, 2, 0))
                                (ssim, diff) = structural_similarity(target_j, predict_j, win_size=None, multichannel=True, data_range=1.0,full=True)
                                psnr = peak_signal_noise_ratio(target_j, predict_j, data_range=1.0)
                                tmp_psnr.append(psnr)
                                tmp_ssim.append(ssim)
                            psnr_score.append(np.mean(tmp_psnr))
                            ssim_score.append(np.mean(tmp_ssim))
                        all_ssim.append(ssim_score)
                        all_psnr.append(psnr_score)
                        all_lpips.append(lpips_score)
        all_ssim = np.array(all_ssim)
        mean_ssim = np.mean(all_ssim, axis=0)
        all_psnr = np.array(all_psnr)
        mean_psnr = np.mean(all_psnr, axis=0)
        all_lpips = np.array(all_lpips)
        mean_lpips = np.mean(all_lpips, axis=0)
        print('ssim: ', mean_ssim, '\n', 'psnr: ', mean_psnr, '\n', 'lpips: ', mean_lpips)
        return np.mean(mean_ssim), np.mean(mean_psnr), np.mean(mean_lpips)

    def start_training(self):
        print('start pix train')
        state_path = './models/pix_{}'.format(self.tag)                # path to save model state
        acc_path = './metric/Acc_{}'.format(self.tag)                  # path to save training metric

        # setup model and optimizer
        model = MSPNet(channels=(64, 128, 256, 512, 1024), layers=4, in_ch=3, hidden_dim=64, BN=False).to(0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.00001)

        # load model state, optimizer state, etc. if retrain
        if self.retrain:
            checkpoint = torch.load(state_path, map_location='cuda:{}'.format(0))
            model.load_state_dict(checkpoint['model_state'])
            cur_epoch = checkpoint['epoch']
            cur_loss = checkpoint['loss']
            accuracy = torch.load(acc_path)
            ssim_score = accuracy['ssim']
            psnr_score = accuracy['psnr']
            lpips_score = accuracy['lpips']
            print('Begine resume training')
        else:
            cur_epoch = 0
            cur_loss = 0
            ssim_score = []
            lpips_score = []
            psnr_score = []


        start_time = datetime.datetime.now()
        print('start time: ', start_time.strftime('%H:%M:%S'))

        for epoch in range(cur_epoch, self.num_epoch):
            model.train()
            for curdir, dirs, files in os.walk(self.train_data_path):
                if len(files) == 0:
                    continue
                random.shuffle(files)
                for file in files:
                    cur_path = os.path.join(curdir, file)
                    train_dataset = MyDataset(path=cur_path, len_input=self.len_input+self.PredSteps)
                    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=3, shuffle=True, drop_last=True)
                    for inputs in train_dataloader:
                        optimizer.zero_grad()
                        inputs = torch.true_divide(inputs, 255).to(0)
                        # print(inputs.size())
                        predictions, error = model(inputs, pred_steps=self.PredSteps, mode='train')
                        error.backward()
                        optimizer.step()

            # validation
            ssim, psnr, lpips = self.val_model(model)
            ssim_score.append(ssim)
            psnr_score.append(psnr)
            lpips_score.append(lpips)
            print('ssim score', ssim, 'psnr score', psnr,'lpips', lpips,
                  'epoch: ', epoch, 'time: ', datetime.datetime.now().strftime('%H:%M:%S'))

            if ssim+psnr/100-lpips > cur_loss:
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'loss': (ssim + psnr / 100 - lpips)
                }, state_path)
                cur_loss = (ssim+psnr/100-lpips)
                torch.save({'ssim': ssim_score, 'psnr': psnr_score, 'lpips': lpips_score}, acc_path)

if __name__ == "__main__":
    rank = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(rank)
    pix_training = PixTraining()
    pix_training.start_training()