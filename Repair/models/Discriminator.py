import torch.nn as nn
import functools
def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=2, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d


        kw = 4
        padw = 1
        sequence = [
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),True),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(ndf * 1, ndf * 2,
                                    kernel_size=kw, stride=2, padding=padw, bias=use_bias), True),

            nn.LeakyReLU(0.2, True),


            spectral_norm(nn.Conv2d(ndf * 2, 1,
                                    kernel_size=kw, stride=2, padding=padw, bias=use_bias), True)

        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class NLayerDiscriminator2(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=6, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator2, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),True),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(ndf * 1, ndf * 2,
                                    kernel_size=kw, stride=2, padding=padw, bias=use_bias), True),

            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4,
                                    kernel_size=kw, stride=2, padding=padw, bias=use_bias), True),

            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 4,
                                    kernel_size=kw, stride=2, padding=padw, bias=use_bias), True),

            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 4,
                                    kernel_size=kw, stride=2, padding=padw, bias=use_bias), True),

            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 4,
                                    kernel_size=kw, stride=2, padding=padw, bias=use_bias), True),

            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8,
                                    kernel_size=kw, stride=2, padding=padw, bias=use_bias), True),

            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf * 8, 1,
                                    kernel_size=kw, stride=2, padding=padw, bias=use_bias), True)

        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)