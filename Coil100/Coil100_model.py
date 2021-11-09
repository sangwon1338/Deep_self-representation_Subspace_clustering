import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
def init_weight(m):
    if type(m)==(nn.Conv2d or nn.ConvTranspose2d):
        init.xavier_uniform(m.weight)
        m.bias.data.fill_(0)


class Coil100_25(nn.Module):
    def __init__(self, test=False):
        super(Coil100_25, self).__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(1, 1, 5, stride=2, padding=2),
            nn.ReLU(True),

        )

        self.decoder_conv1 = nn.ConvTranspose2d(1, 1, 5, stride=2, padding=2)
        self.Relu = nn.ReLU(True)



        self.Coef = nn.Parameter(1.0e-8 * torch.ones((7200, 7200)))
        self.softmax = nn.Softmax(dim=1)

        self.attention_softmax = nn.Sequential(
            nn.Conv2d(1, 1, 5, stride=2, padding=2),
            nn.ReLU(True),

        )

        self.test = test
        self.shape = []



    def forward(self, x):
        self.shape.append(x.shape)




        if self.test == True:
            encoder_x = self.encoder(x)
            softmax_x = self.attention_softmax(x)
            z_conv, z_ssc = self.self_expressive(encoder_x, softmax_x)
            x = torch.reshape(z_ssc, encoder_x.shape)
            x = self.decoder_conv1(x, output_size=self.shape[0])
            x = self.Relu(x)
            return x, z_conv, z_ssc, self.Coef

        else:
            x = self.encoder(x)
            x = self.decoder_conv1(x, output_size=self.shape[0])
            x = self.Relu(x)

            return x

    def self_expressive(self, x, att_x):
        z_conv = x.view(x.shape[0], -1)
        z_conv_att = att_x.view(att_x.shape[0], -1)
        z_conv_att = self.softmax(z_conv_att)
        z_conv = z_conv * z_conv_att
        z_ssc = torch.matmul(self.Coef, z_conv)
        return z_conv, z_ssc


class Coil100_50(nn.Module):
    def __init__(self, test=False):
        super(Coil100_50, self).__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(1, 2, 5, stride=2, padding=2),
            nn.ReLU(True),

        )

        self.decoder_conv1 = nn.ConvTranspose2d(2, 1, 5, stride=2, padding=2)
        self.Relu = nn.ReLU(True)

        self.Coef = nn.Parameter(1.0e-8 * torch.ones((7200, 7200)))
        self.softmax = nn.Softmax(dim=1)
        self.test = test

        self.attention_softmax = nn.Sequential(
            nn.Conv2d(1, 2, 5, stride=2, padding=2),
            nn.ReLU(True),

        )

    def forward(self, x):
        x_shape = x.shape

        if self.test == True:
            encoder_x = self.encoder(x)
            softmax_x = self.attention_softmax(x)

            z_conv, z_ssc = self.self_expressive(encoder_x, softmax_x)
            x = torch.reshape(z_ssc, encoder_x.shape)
            x = self.decoder_conv1(x, output_size=x_shape)
            x = self.Relu(x)
            return x, z_conv, z_ssc, self.Coef

        else:
            x = self.encoder(x)
            x = self.decoder_conv1(x, output_size=x_shape)
            x = self.Relu(x)

            return x

    def self_expressive(self, x, att_x):
        z_conv = x.view(x.shape[0], -1)
        z_conv_att = att_x.view(att_x.shape[0], -1)
        z_conv_att = self.softmax(z_conv_att)
        z_conv = z_conv * z_conv_att
#        z_conv = F.normalize(z_conv, dim=1, p=2)
        z_ssc = torch.matmul(self.Coef, z_conv)
        return z_conv, z_ssc


class Coil100_75(nn.Module):
    def __init__(self, test=False):
        super(Coil100_75, self).__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(1, 3, 5, stride=2, padding=2),
            nn.ReLU(True),

        )

        self.decoder_conv1 = nn.ConvTranspose2d(3, 1, 5, stride=2, padding=2)
        self.Relu = nn.ReLU(True)



        self.Coef = nn.Parameter(1.0e-8 * torch.ones((7200, 7200)))
        self.softmax = nn.Softmax(dim=1)

        self.attention_softmax = nn.Sequential(
            nn.Conv2d(1, 3, 5, stride=2, padding=2),
            nn.ReLU(True),

        )

        self.test = test
        self.shape = []



    def forward(self, x):
        self.shape.append(x.shape)




        if self.test == True:
            encoder_x = self.encoder(x)
            softmax_x = self.attention_softmax(x)
            z_conv, z_ssc = self.self_expressive(encoder_x, softmax_x)
            x = torch.reshape(z_ssc, encoder_x.shape)
            x = self.decoder_conv1(x, output_size=self.shape[0])
            x = self.Relu(x)
            return x, z_conv, z_ssc, self.Coef

        else:
            x = self.encoder(x)
            x = self.decoder_conv1(x, output_size=self.shape[0])
            x = self.Relu(x)

            return x

    def self_expressive(self, x, att_x):
        z_conv = x.view(x.shape[0], -1)
        z_conv_att = att_x.view(att_x.shape[0], -1)
        z_conv_att = self.softmax(z_conv_att)
        z_conv = z_conv * z_conv_att
        z_ssc = torch.matmul(self.Coef, z_conv)
        return z_conv, z_ssc