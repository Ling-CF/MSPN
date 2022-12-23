import torch.nn as nn
import torch
from NetBlock import EnDeNet


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, EnDe_channels, higher_channel, out_dim, BN=False):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        EnDe_channels: (int, int, ...)
            input channels for the encoder-decoder network
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.en_decoder = EnDeNet(
            channels=EnDe_channels, in_ch=self.input_dim+self.hidden_dim, out_ch=4*self.hidden_dim, higher_channel=higher_channel, BN=BN
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels=out_dim, kernel_size=(5,5), padding=2),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, input_tensor, cur_state, higher_h):
        '''

        :param input_tensor:
            input frames or feature maps, size = (b, c, h, w)
        :param cur_state:
            including hidden state and cell state
        '''
        h_cur, c_cur = cur_state # h_cur: current hidden state; c_cur: current cell state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined, current_h = self.en_decoder(combined,  higher_h=higher_h)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined, self.hidden_dim, dim=1)

        # split into 4 tensor(meet with the program of out_channels=4 * self.hidden_dim in line 24)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return self.out_conv(h_next), (h_next, c_next), current_h

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

