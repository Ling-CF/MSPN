import os
import numpy as np
import torch
from MSPN import MSPNet
import cv2
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from utils import util_of_lpips, MyDataset




def show_muti_img(imgs, targets, index):
    """showing predicted images"""
    pre_imgs = []
    tar_imgs = []
    # print(len(imgs), targets[0].size())
    for i in range(0, len(imgs), 1):
        pre = imgs[i]
        pre = pre.cpu().numpy().squeeze()
        target = targets[:, i]
        target = target.cpu().squeeze().numpy()
        if len(target.shape) > 2:
            pre = np.transpose(pre, (1, 2, 0))
            target = np.transpose(target, (1, 2, 0))
        pre_imgs.append(pre)
        tar_imgs.append(target)
    stack_img = np.hstack(pre_imgs)
    stack_tar = np.hstack(tar_imgs)
    cv2.imshow('stack_img', stack_img)
    cv2.imshow('tar_img', stack_tar)
    d = cv2.waitKey()
    if d == 27:
        # Enter 'Esc' to close the window
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()
        print(index)
        save_img = input('save image? Enter y if yes.')
        if save_img == 'y':
            name = input('Enter save name: ')
            save_name = './predictions/{}_pred.png'.format(name)
            cv2.imwrite(save_name, stack_img*255)
            tar_name = './predictions/{}_real.png'.format(name)
            cv2.imwrite(tar_name, stack_tar*255)



def test_model():
    # MNIST channels=(64, 128, 256, 512), layers=3, in_ch=1, hidden_dim=32
    # Others channels=(64, 128, 256, 512, 1024), layers=4, in_ch=3, hidden_dim=64
    model=MSPNet(channels=(64, 128, 256, 512), layers=3, in_ch=1, hidden_dim=32, BN=False)
    test_data_path='/mnt/DevFiles_NAS/Ling.cf/dataset/{}/test'.format('MNIST')       # the path where test data is stored
    state_path='./models/pix_MSPN_MNIST.pt'                                      # the path where model state is stored
    PredSteps=10               # the number of time steps in the long term prediction
    show_img=True             # bool: visualize image or not, if true, setting the batch size to 1
    LenInput=10               # length of input sequence
    interval = 1
    batch_size = 32

    if show_img:
        batch_size = 1

    model = model.to(0)
    checkpoint = torch.load(state_path, map_location='cuda:{}'.format(0))
    model.load_state_dict(checkpoint['model_state']) # pixel training
    # model.load_state_dict(checkpoint['g_model'])       # adversarial training
    model.eval()
    all_ssim = []
    all_psnr = []
    all_lpips = []
    all_mse = []
    lpips_loss = util_of_lpips('alex')
    with torch.no_grad():
        for curdir, dirs, files in os.walk(test_data_path):
            if len(files) == 0:
                continue
            files.sort()
            print(files)
            for file in files:
                cur_path = os.path.join(curdir, file)
                test_dataset = MyDataset(path=cur_path, len_input=LenInput + PredSteps, interval=interval, begin=0)
                test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, drop_last=False)

                for data in test_dataloader:
                    inputs = torch.true_divide(data[:, :LenInput], 255).to(0)
                    targets = torch.true_divide(data[:, LenInput:], 255).to(0)
                    ssim_score = []
                    psnr_score = []
                    lpips_score = []
                    mse_score = []
                    # print(inputs.size())
                    predictions, errors = model(inputs, pred_steps=PredSteps, mode='test')
                    # print(len(predictions), targets.size())
                    if show_img:
                        show_muti_img(predictions, targets, (file))
                    for t in range(PredSteps):
                        target = targets[:, t]
                        predict = predictions[t]

                        lpips = lpips_loss.calc_lpips(predict, target)
                        lpips_score.append(lpips.mean().item())

                        target = target.cpu().squeeze().numpy()
                        predict = predict.data.cpu().numpy().squeeze()
                        if len(target.shape) > 2:
                            target = np.transpose(target, (1, 2, 0))
                            predict = np.transpose(predict, (1, 2, 0))
                        (ssim, diff) = structural_similarity(target, predict, win_size=None, multichannel=True, data_range=1.0,
                                                              full=True)
                        psnr = peak_signal_noise_ratio(target, predict, data_range=1.0)
                        mse = np.sum(np.square(target - predict)) / batch_size

                        psnr_score.append(psnr)
                        ssim_score.append(ssim)
                        mse_score.append(mse)

                    all_ssim.append(ssim_score)
                    all_psnr.append(psnr_score)
                    all_lpips.append(lpips_score)
                    all_mse.append(mse_score)


    all_ssim = np.array(all_ssim)
    mean_ssim = np.mean(all_ssim, axis=0)
    all_psnr = np.array(all_psnr)
    mean_psnr = np.mean(all_psnr, axis=0)
    all_lpips = np.array(all_lpips)
    mean_lpips = np.mean(all_lpips, axis=0) * 100
    all_mse = np.array(all_mse)
    mean_mse = np.mean(all_mse, axis=0)
    print('mean ssim: ', '\n', mean_ssim, '\n', '0-10: ', np.mean(mean_ssim[:10]), '\n', '5-10: ', np.mean(mean_ssim[5:]))
    print("mean psnr: ", '\n', mean_psnr, '\n', '0-10: ', np.mean(mean_psnr[:10]), '\n', '5-10: ', np.mean(mean_psnr[5:]))
    print("mean lpips: ", '\n', mean_lpips, '\n', '0-10: ', np.mean(mean_lpips[:10]), '\n', '5-10: ',np.mean(mean_lpips[5:]))
    print("mean mse: ", '\n', mean_mse, '\n', '0-10: ', np.mean(mean_mse[:10]), '\n', '5-10: ',
          np.mean(mean_mse[5:]))



if __name__ == '__main__':
    rank = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(rank)
    print('start testing')
    test_model()
