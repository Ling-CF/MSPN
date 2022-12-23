# muti-scale predictive Network
import torch
import torch.nn as nn
from torch.nn import functional as F
from ConvLSTM_Module import ConvLSTMCell

class MSPNet(nn.Module):
    def __init__(self, channels, layers, in_ch, hidden_dim, BN=False):
        super(MSPNet, self).__init__()
        '''
        channels  : channels for encoder-decoder network
        layers    : network levels
        in_ch     : input channel, 3 if RGB image else 1 (gray image)  
        hidden_dim: dimension for hidden state or cell state of LSTM network
        BN        : using Batch Normalization or not
        '''
        self.n_layers = layers
        self.hidden_dim = hidden_dim
        self.in_ch = in_ch
        assert len(channels) >= layers

        # E: local prediction error; L: lower prediction error; H: higher prediction
        for l in range(1, layers - 1):
            ELHcell = ConvLSTMCell(input_dim=in_ch * 5, hidden_dim=self.hidden_dim, EnDe_channels=channels[:-l], higher_channel=channels[:-l][-2], out_dim=in_ch, BN=BN)
            setattr(self, 'ELHcell{}'.format(l), ELHcell)

        # for the highest level, no prediction from higher level
        self.ELcell = ConvLSTMCell(input_dim=in_ch * 4, hidden_dim=self.hidden_dim, EnDe_channels=channels[:-(layers-1)], higher_channel=0, out_dim=in_ch, BN=BN)

        # for the lowest level, no prediction error from lower level
        self.EHcell = ConvLSTMCell(input_dim=in_ch * 3, hidden_dim=self.hidden_dim, EnDe_channels=channels, higher_channel=channels[-2], out_dim=in_ch, BN=BN)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)



    def forward(self, input, pred_steps, mode):  # input_size:(b, t, c, w, h)
        assert mode in ['train', 'test', 'adv_train']

        pred_seq = [None] * self.n_layers                # for storing predictions of each network level
        Hidden_seq = [None] * self.n_layers              # for storing hidden state
        Error_seq = [None] * self.n_layers               # for storing prediction errors


        w, h = input.size(-2), input.size(-1)
        batch_size = input.size(0)
        time_steps = input.size(1)

        # initializing
        for l in range(self.n_layers):
            Error_seq[l] = torch.zeros(batch_size, self.in_ch*2, w, h).to(input.device)

            pred_seq[l] = torch.zeros(batch_size, self.in_ch, w, h).to(input.device)
            Hidden_seq[l] = (torch.zeros(batch_size, self.hidden_dim, w, h).to(input.device), torch.zeros(batch_size, self.hidden_dim, w, h).to(input.device))
            w = w // 2
            h = h // 2

        lambda_l = 1
        total_loss = 0
        predictions = []

        if mode == 'train' or mode == 'adv_train':
            real_t = time_steps-pred_steps # Number of steps to use real image as input
        elif mode == 'test':
            real_t = time_steps

        for t in range(real_t+pred_steps):

            higher_h = None
            for l in reversed(range(self.n_layers)):
                # computation would be different at the lowest, the highest and the other levels
                if l == 0:
                    cell = self.EHcell
                    Error = Error_seq[l]
                    Hidden_state = Hidden_seq[l]
                    Higher_pred = self.upsample(pred_seq[l+1])
                    temp = torch.cat([Error, Higher_pred], dim=1)
                    pred, next_hid_state, current_h = cell(temp, Hidden_state,  higher_h=higher_h)
                elif l == self.n_layers-1:
                    cell = self.ELcell
                    Error = Error_seq[l]
                    Hidden_state = Hidden_seq[l]
                    Lower_error = F.interpolate(Error_seq[l-1], scale_factor=0.5)
                    temp = torch.cat([Error, Lower_error], dim=1)
                    pred, next_hid_state, current_h = cell (temp, Hidden_state,  higher_h=higher_h)
                    higher_h = current_h
                else:
                    cell = getattr(self, 'ELHcell{}'.format(l))
                    Error = Error_seq[l]
                    Hidden_state = Hidden_seq[l]
                    Lower_error = F.interpolate(Error_seq[l - 1], scale_factor=0.5)
                    Higher_pred = self.upsample(pred_seq[l + 1])
                    temp = torch.cat([Error, Lower_error, Higher_pred], dim=1)
                    pred, next_hid_state, current_h = cell(temp, Hidden_state,  higher_h=higher_h)
                    higher_h = current_h
                pred_seq[l] = pred
                Hidden_seq[l] = next_hid_state
                if l == 0 and t >= real_t and mode == 'test':
                    predictions.append(pred)
                if mode == 'adv_train' and l == 0:
                    predictions.append(pred)
            if t < real_t:
                cur_target = input[:, t]
            else:
                cur_target = pred_seq[0]
            w, h = cur_target.size()[-2:]
            for l in range(self.n_layers):
                cur_frame_l = F.interpolate(cur_target, size=(w//(2**l), h//(2**l)))
                cur_pred = pred_seq[l]
                pos = F.relu(cur_frame_l - cur_pred)
                neg = F.relu(cur_pred - cur_frame_l)
                E = torch.cat([pos, neg], 1)
                Error_seq[l] = E


            if mode == 'train' or mode == 'adv_train':
                cur_target = input[:, t]
                w, h = cur_target.size()[-2:]
                for l in range(self.n_layers):
                    cur_pred = pred_seq[l]
                    cur_target_l = F.interpolate(cur_target, size=(w//(2**l), h//(2**l)))
                    loss = torch.sum((cur_pred-cur_target_l)**2)
                    if t == 0:
                        lambda_t = 0.0
                    else:
                        lambda_t = 1.0
                    total_loss += loss * lambda_l * lambda_t



        return predictions, total_loss

