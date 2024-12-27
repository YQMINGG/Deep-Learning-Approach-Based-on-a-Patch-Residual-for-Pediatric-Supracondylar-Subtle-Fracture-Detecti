"""
original code from facebook research:
https://github.com/facebookresearch/ConvNeXt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans: int = 26, num_classes: int = None, depths: list = None,
                 dims: list = None, drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.):
        super().__init__()
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                             LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)

        # 对应stage2-stage4前的3个downsample
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                                             nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        # 构建每个stage中堆叠的block
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer

        self.apply(self._init_weights)


        # 处理每层的feature_map
        self.relu = nn.ReLU(inplace=True)
        # 第一层
        self.score_dsn1 = nn.Conv2d(96, 64, 3, 1, 1)
        self.bn_1 = nn.BatchNorm2d(64)
        self.score_dsn1_p = nn.AdaptiveMaxPool2d(36)

        self.score_dsn1_1 = nn.Conv2d(64, 32, 3, 1, 1)
        self.bn_1_1 = nn.BatchNorm2d(32)
        self.score_dsn1_p_p = nn.AdaptiveMaxPool2d(18)

        self.score_dsn1_1_1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn_1_1_1 = nn.BatchNorm2d(32)
        self.score_dsn1_p_p_p = nn.AdaptiveMaxPool2d(9)
        self.score_dsn1_1_1_1 = nn.Conv2d(32, 1, 1)

        # 第二层
        self.score_dsn2 = nn.Conv2d(192, 128, 3, 1, 1)
        self.bn_2 = nn.BatchNorm2d(128)
        self.score_dsn1_p = nn.AdaptiveMaxPool2d(18)

        self.score_dsn2_2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.bn_2_2 = nn.BatchNorm2d(64)
        self.score_dsn2_p_p = nn.AdaptiveMaxPool2d(9)

        self.score_dsn2_2_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn_2_2_2 = nn.BatchNorm2d(64)
        self.score_dsn2_2_2_2 = nn.Conv2d(64, 1, 1)

        # 第三层
        self.score_dsn3 = nn.Conv2d(384, 256, 3, 1, 1)
        self.bn_3 = nn.BatchNorm2d(256)
        self.score_dsn3_p = nn.AdaptiveMaxPool2d(9)

        self.score_dsn3_3 = nn.Conv2d(256, 128, 3, 1, 1)
        self.bn_3_3 = nn.BatchNorm2d(128)

        self.score_dsn3_3_3 = nn.Conv2d(128, 64, 3, 1, 1)
        self.score_dsn3_3_3_3 = nn.Conv2d(64, 1, 1)

        # 第四层
        self.score_dsn4 = nn.Conv2d(768, 512, 3, 1, 1)
        self.bn_4 = nn.BatchNorm2d(512)
        self.score_dsn4_4 = nn.Conv2d(512, 64, 3, 1, 1)
        self.score_dsn4_4_4 = nn.Conv2d(64, 1, 1)

        self.out_conv = nn.Conv2d(4, 1, 1)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat_list = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feat_list.append(x)

        return feat_list   # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_list   = self.forward_features(x)
        feat1, feat2, feat3, feat4 = feat_list[0],feat_list[1],feat_list[2],feat_list[3]

        #通道卷为1
        score_dsn1 = self.score_dsn1_1_1_1(
            self.score_dsn1_p_p_p(self.relu(self.bn_1_1_1(self.score_dsn1_1_1(self.score_dsn1_p_p(self.relu(self.bn_1_1(
                self.score_dsn1_1(self.score_dsn1_p(self.relu(self.bn_1(self.score_dsn1(feat1)))))))))))))

        score_dsn2 = self.score_dsn2_2_2_2(self.relu(self.bn_2_2_2(self.score_dsn2_2_2(
            self.relu(self.score_dsn2_p_p(self.bn_2_2(self.score_dsn2_2(self.score_dsn1_p(self.relu(self.bn_2(self.score_dsn2(feat2))))))))))))

        score_dsn3 = self.score_dsn3_3_3_3(self.score_dsn3_3_3(self.relu(self.bn_3_3(
            self.score_dsn3_3(self.score_dsn3_p(self.relu((self.bn_3(self.score_dsn3(feat3))))))))))

        score_dsn4 = self.score_dsn4_4_4(self.score_dsn4_4(self.relu(self.bn_4(self.score_dsn4(feat4)))))
        # x = self.classifier(x)
        bboxes = torch.cat([score_dsn1, score_dsn2, score_dsn3, score_dsn4], dim=1)
        out_bbox = self.out_conv(bboxes)
        # out_bbox    = self.sig(out_bbox)
        return out_bbox



def convnext_tiny():
    # https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
    model = ConvNeXt(depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     )
    return model
# if __name__ == "__main__":
#
#     input_one = torch.ones(1, 26, 288, 288)
#     model = convnext_tiny(2)
#     out_put = model(input_one)
#     print(out_put.shape)

def convnext_small(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes)
    return model

def convnext_base(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024],
                     num_classes=num_classes)
    return model

def convnext_large(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth
    # https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[192, 384, 768, 1536],
                     num_classes=num_classes)
    return model


def convnext_xlarge(num_classes: int):
    # https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth
    model = ConvNeXt(depths=[3, 3, 27, 3],
                     dims=[256, 512, 1024, 2048],
                     num_classes=num_classes)
    return model