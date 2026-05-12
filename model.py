import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class background_spatial(nn.Module):
    def __init__(self):
        super(background_spatial, self).__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=5, kernel_size=(5,1,1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=5, out_channels=5, kernel_size=(5, 1, 1), stride=(2, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=5, out_channels=10, kernel_size=(5, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=10, out_channels=10, kernel_size=(5, 1, 1), stride=(2, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=10, out_channels=20, kernel_size=(19, 2, 2)),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels=20, out_channels=20, kernel_size=(17, 2, 2)),
            nn.LeakyReLU(),
                                    )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv3d(x)
        x = self.flatten(x)
        return x


class background_autoencoder(nn.Module):
    def __init__(self, d, row_d):
        super(background_autoencoder, self).__init__()
        self.input_feature = d
        self.row_feature = row_d
        self.hidden_encoder_1 = nn.Sequential(nn.Linear(self.input_feature, 100), nn.ReLU(),)
        self.hidden_encoder_2 = nn.Sequential(nn.Linear(100, 60), nn.ReLU())
        self.hidden_encoder_3 = nn.Sequential(nn.Linear(60, 20), nn.ReLU())
        self.hidden = nn.Sequential(nn.Linear(20, 20), nn.ReLU())
        self.hidden_decoder_1 = nn.Sequential(nn.Linear(20, 60), nn.ReLU())
        self.hidden_decoder_2 = nn.Sequential(nn.Linear(60, 100), nn.ReLU())
        self.hidden_decoder_3 = nn.Linear(100, self.row_feature)


    def forward(self, x):
        x = self.hidden_encoder_1(x)
        feature_1 = x.clone()
        x = self.hidden_encoder_2(x)
        feature_2 = x.clone()
        x = self.hidden_encoder_3(x)
        feature_3 = x.clone()
        x = self.hidden(x)
        x = (x + feature_3) / 2
        x = self.hidden_decoder_1(x)
        x = (x + feature_2) / 2
        x = self.hidden_decoder_2(x)
        x = (x + feature_1) / 2
        x = self.hidden_decoder_3(x)
        return x


class sae_background_updemension(nn.Module):
    def __init__(self, d):
        super(sae_background_updemension, self).__init__()
        self.input_feature = d
        self.linear = nn.Linear(self.input_feature, 200)

    def forward(self, x):
        x = self.linear(x)
        return x


class sae_background_encoder(nn.Module):
    def __init__(self):
        super(sae_background_encoder, self).__init__()
        self.hidden = nn.Sequential(nn.ReLU(),
                                    nn.Linear(200, 100), nn.ReLU(),
                                    nn.Linear(100, 40), nn.ReLU(),
                                    nn.Linear(40, 20), nn.ReLU(),
                                    nn.Linear(20, 20), nn.ReLU(),
                                    nn.Linear(20, 40), nn.ReLU(),
                                    nn.Linear(40, 100), nn.ReLU(),
                                    nn.Linear(100, 200)
                                    )

    def forward(self, x):
        x = self.hidden(x)
        return x


class sae_background_decoder(nn.Module):
    def __init__(self, row_d):
        super(sae_background_decoder, self).__init__()
        self.row_feature = row_d
        self.hidden = nn.Sequential(nn.ReLU(),
                                    nn.Linear(200, self.row_feature)
                                    )

    def forward(self, x):
        x = self.hidden(x)
        return x

class sae_background(nn.Module):
    def __init__(self, d, row_d):
        super(sae_background, self).__init__()
        self.spatial = background_spatial()
        self.ae = background_autoencoder(d,row_d)

    def forward(self, x):
        x = self.spatial(x)
        x = self.ae(x)
        return x


class target_spatial(nn.Module):
    def __init__(self):
        super(target_spatial, self).__init__()
        self.conv3d = nn.Sequential(
                                    nn.Conv3d(in_channels=1, out_channels=5, kernel_size=(7, 1, 1)),
                                    nn.ReLU(),
                                    nn.Conv3d(in_channels=5, out_channels=5, kernel_size=(7, 1, 1), stride=(2, 1, 1)),
                                    nn.ReLU(),
                                    nn.Conv3d(in_channels=5, out_channels=10, kernel_size=(21, 2, 2)),
                                    nn.LeakyReLU(),
                                    nn.Conv3d(in_channels=10, out_channels=10, kernel_size=(17, 2, 2)),
                                    nn.LeakyReLU(),
                                    )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv3d(x)
        x = self.flatten(x)
        return x


class target_autoencoder(nn.Module):
    def __init__(self, d, row_d):
        super(target_autoencoder, self).__init__()
        self.input_feature = d
        self.row_feature = row_d
        self.hidden_encoder_1 = nn.Sequential(nn.Linear(self.input_feature, 100), nn.ReLU(),)
        self.hidden_encoder_2 = nn.Sequential(nn.Linear(100, 60), nn.ReLU())
        self.hidden_encoder_3 = nn.Sequential(nn.Linear(60, 20), nn.ReLU())
        self.hidden = nn.Sequential(nn.Linear(20, 20), nn.ReLU())
        self.hidden_decoder_1 = nn.Sequential(nn.Linear(20, 60), nn.ReLU())
        self.hidden_decoder_2 = nn.Sequential(nn.Linear(60, 100), nn.ReLU())
        self.hidden_decoder_3 = nn.Linear(100, self.row_feature)

    def forward(self, x):
        # x = self.hidden(x)
        x = self.hidden_encoder_1(x)
        feature_1 = x.clone()
        x = self.hidden_encoder_2(x)
        feature_2 = x.clone()
        x = self.hidden_encoder_3(x)
        feature_3 = x.clone()
        x = self.hidden(x)
        x = (x + feature_3) / 2
        x = self.hidden_decoder_1(x)
        x = (x + feature_2) / 2
        x = self.hidden_decoder_2(x)
        x = (x + feature_1) / 2
        x = self.hidden_decoder_3(x)
        return x


class sae_target(nn.Module):
    def __init__(self, d, row_d):
        super(sae_target, self).__init__()
        self.spatial = target_spatial()
        self.ae = target_autoencoder(d,row_d)
        # self.ae = sae_target_encoder(d)

    def forward(self, x):
        x = self.spatial(x)
        x = self.ae(x)
        return x