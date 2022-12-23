import torch
import os
from MSPN import MSPNet
from utils import Loss, MyDataset, util_of_lpips
from Disrciminator import discriminator
import datetime
import random
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
device = torch.device('cuda:0')

class AdvTraining():
    def __init__(self):
        self.dataset = 'Human3.6M'
        self.train_data_path = '/mnt/DevFiles_NAS/Ling.cf/dataset/{}/train'.format(self.dataset)
        self.val_data_path = '/mnt/DevFiles_NAS/Ling.cf/dataset/{}/test'.format(self.dataset)
        self.tag = 'MSPN_{}_2.pt'.format(self.dataset)
        self.retrain = True
        self.pix_pretrain = False
        self.num_epoch = 60
        self.len_input = 10
        self.PredSteps = 10

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
                                (ssim, diff) = structural_similarity(target_j, predict_j, win_size=None,
                                                                     multichannel=True, data_range=1.0, full=True)
                                psnr = peak_signal_noise_ratio(target_j, predict_j, data_range=1.0)
                                tmp_psnr.append(psnr)
                                tmp_ssim.append(ssim)
                            psnr_score.append(np.mean(tmp_psnr))
                            ssim_score.append(np.mean(tmp_ssim))
                        # print(ssim_score, psnr_score, lpips_score)
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
        print('adversarial training')
        state_path = './models/adv_{}'.format(self.tag)  # storage path for model state
        acc_path = './metric/adv_Acc_{}'.format(self.tag)  # storage path for all precision (every epoch)
        pix_state_path = './models/pix_{}'.format(self.tag)

        # setup model and optimizer
        g_model = MSPNet(channels=(64, 128, 256, 512, 1024), layers=4, in_ch=3, hidden_dim=64, BN=False).to(device)
        d_model = discriminator(channels=(64, 64, 128, 256, 256, 512), in_channel=3, size=(4,4)).to(device)
        g_optimizer = torch.optim.Adam(g_model.parameters(), lr=0.00001, weight_decay=0.0001)
        d_optimizer = torch.optim.Adam(d_model.parameters(), lr=1/10**7, weight_decay=0.0001)

        # load model state, optimizer state, etc. if retrain
        if self.retrain:
            checkpoint = torch.load(state_path, map_location='cuda:{}'.format(0))
            g_model.load_state_dict(checkpoint['g_model'])
            # d_seq_model.load_state_dict(checkpoint['d_seq_model'])
            d_model.load_state_dict(checkpoint['d_img_model'])
            cur_epoch = checkpoint['epoch']
            cur_loss = checkpoint['loss']
            accuracy = torch.load(acc_path)
            ssim_score = accuracy['ssim']
            psnr_score = accuracy['psnr']
            lpips_score = accuracy['lpips']
            print('Begine resume training')
        elif self.pix_pretrain:
            checkpoint = torch.load(pix_state_path, map_location='cuda:{}'.format(0))
            g_model.load_state_dict(checkpoint['model_state'])
            cur_epoch = 0
            cur_loss = 0
            ssim_score = []
            lpips_score = []
            psnr_score = []
            print('Begine resume pixel training')
        else:
            cur_epoch = 0
            cur_loss = 0
            ssim_score = []
            lpips_score = []
            psnr_score = []

        loss = Loss()
        start_time = datetime.datetime.now()
        print('start time: ', start_time.strftime('%H:%M:%S'))

        for epoch in range(cur_epoch, self.num_epoch):
            for curdir, dirs, files in os.walk(self.train_data_path):
                if len(files)==0:
                    continue
                for file in files:
                    cur_path = os.path.join(curdir, file)
                    train_dataset = MyDataset(path=cur_path, len_input=self.len_input+self.PredSteps)
                    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True,
                                                                   num_workers=1, drop_last=True)
                    train_dataloader = list(train_dataloader)
                    data_length = len(train_dataloader)
                    idx = 0

                    d_fake_score = [0]
                    d_real_score = [0]

                    while idx < data_length-1:

                        # train discriminator
                        # print('train discriminator', idx)
                        while idx < data_length-1 and np.mean(d_fake_score) >= (
                                np.mean(d_real_score) - np.abs(np.mean(d_real_score)/ 50)):
                            d_model.train()
                            g_model.eval()
                            idx += 1
                            inputs = torch.true_divide(train_dataloader[idx], 255).to(device)
                            real_data = inputs[:, self.len_input//2:].squeeze()
                            logits_real = d_model(real_data)
                            # pred_steps = random.randint(3, 5)
                            fake_imgs, error = g_model(inputs, pred_steps=self.PredSteps, mode='adv_train')
                            fake_data = torch.cat(fake_imgs[self.len_input//2:], dim=0)
                            logits_fake = d_model(fake_data.detach())
                            d_total_error, d_fake, d_real = loss.d_loss(logits_real, logits_fake)  # 判别器的 loss
                            # print(d_real, d_fake)
                            d_optimizer.zero_grad()
                            d_total_error.backward()
                            d_optimizer.step()

                            d_fake_score.append(d_fake)
                            d_real_score.append(d_real)
                            if len(d_fake_score) > 3:
                                del d_fake_score[0]
                                del d_real_score[0]

                            if len(d_fake_score) == 3 and np.mean(d_fake_score) < (
                                    np.mean(d_real_score) - np.abs(np.mean(d_real_score) / 100)):
                                break

                        # train generator
                        # print('train generator', idx)
                        while idx < data_length-1:
                            d_model.eval()
                            g_model.train()
                            idx += 1
                            inputs = torch.true_divide(train_dataloader[idx], 255).to(device)
                            real_data = inputs[:, self.len_input//2:].squeeze()
                            # pred_steps = random.randint(3, 5)
                            fake_imgs, error = g_model(inputs, pred_steps=self.PredSteps, mode='adv_train')
                            fake_data = torch.cat(fake_imgs[self.len_input//2:], dim=0)
                            logits_real = d_model(real_data)
                            gen_logits_fake = d_model(fake_data)
                            g_loss, d_fake, d_real = loss.g_loss(gen_logits_fake, logits_real)
                            total_loss = error + 100*g_loss
                            # print(total_loss)
                            g_optimizer.zero_grad()
                            total_loss.backward()
                            g_optimizer.step()

                            d_fake_score.append(d_fake)
                            d_real_score.append(d_real)
                            if len(d_fake_score) > 3:
                                del d_fake_score[0]
                                del d_real_score[0]
                            if len(d_fake_score) == 3 and np.mean(d_fake_score) >= (
                                    np.mean(d_real_score) - np.abs(np.mean(d_real_score)/ 50) ):
                                break

                        # print('fake score:', np.mean(d_fake_score), 'real score:', np.mean(d_real_score))

            # validation
            ssim, psnr, lpips = self.val_model(g_model)
            ssim_score.append(ssim)
            psnr_score.append(psnr)
            lpips_score.append(lpips)
            print('ssim score', ssim, 'psnr score', psnr, 'lpips', lpips,
                  'epoch: ', epoch, 'time: ', datetime.datetime.now().strftime('%H:%M:%S'))

            # the accuracy obtained by adversarial training is more volatile
            # sometimes lower accuracy doesn't mean worse results, so we divide cur_loss by a factor for comparison
            if ssim + psnr / 100 - lpips > (cur_loss / 1.02):
                torch.save({
                    'epoch': epoch,
                    'g_model': g_model.state_dict(),
                    # 'd_seq_model' : d_seq_model.state_dict(),
                    'd_img_model': d_model.state_dict(),
                    'loss': (ssim + psnr / 100 - lpips)
                }, state_path)
                cur_loss = max((ssim + psnr / 100 - lpips), cur_loss)

            torch.save({'ssim': ssim_score, 'psnr': psnr_score, 'lpips': lpips_score}, acc_path)


if __name__ == '__main__':
    rank = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(rank)
    adv_train = AdvTraining()
    adv_train.start_training()