import torch
from torch import nn
from torchvision import models

RESNETS = ("resnet18", "resnet34", "resnet50", "resnet101")
LAYERS = ("layer1", "layer2", "layer3", "layer4")


def _set_stride(layer, stride):
    # шаг меняем только в первом блоке стадии: и в основной ветке, и в shortcut (downsample)
    block = layer[0]
    conv = block.conv2 if hasattr(block, "conv3") else block.conv1  # Bottleneck / BasicBlock
    conv.stride = stride
    if block.downsample is not None:
        block.downsample[0].stride = stride


class ResNetBackbone(nn.Module):
#  Готовый ResNet под строки текста.  Вид: (B, 1, H, W) -> (B, C, 1, ceil(W / 4))
#  Ширина сжимается ровно в 4 раза (stem), в layer2..layer4 шаг (2, 1) - сжимается только высота,
#  в конце высота схлопывается в 1 (AdaptiveAvgPool), поэтому работает с любой H (32, 64, ...).

    def __init__(self, name="resnet34", last_layer="layer3", pretrained=True):
        super().__init__()
        if name not in RESNETS:
            raise ValueError(f"name должен быть одним из {RESNETS}")
        if last_layer not in LAYERS[1:]:
            raise ValueError(f"last_layer должен быть одним из {LAYERS[1:]}")
        self._frozen = False

        # pretrained=True скачает веса ImageNet при первом запуске
        net = getattr(models, name)(weights="DEFAULT" if pretrained else None)

        # серая картинка: первый слой на 1 канал, веса = сумма весов по RGB (так предобучение сохраняется)
        old = net.conv1
        conv1 = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                          stride=old.stride, padding=old.padding, bias=False)
        if pretrained:
            with torch.no_grad():
                conv1.weight.copy_(old.weight.sum(dim=1, keepdim=True))

        used = LAYERS[: LAYERS.index(last_layer) + 1]
        for layer_name in used[1:]:
            _set_stride(getattr(net, layer_name), (2, 1))

        self.stem = nn.Sequential(conv1, net.bn1, net.relu, net.maxpool)   # /4 по высоте и по ширине
        self.layers = nn.Sequential(*[getattr(net, n) for n in used])
        self.pool = nn.AdaptiveAvgPool2d((1, None))

        last_block = self.layers[-1][-1]
        bn = last_block.bn3 if hasattr(last_block, "bn3") else last_block.bn2
        self.out_channels = bn.num_features    # подставлять в BiLSTM как input_size

    def forward(self, x):
        x = self.stem(x)
        x = self.layers(x)
        return self.pool(x)

    @staticmethod
    def output_lengths(image_widths):
        # conv1 (7x7, шаг 2) и maxpool (3x3, шаг 2) дают ceil(W / 2) каждый -> ceil(W / 4)
        return (image_widths + 3) // 4

    def freeze(self):
        # заморозить веса и держать BatchNorm в eval (иначе статистики плывут на маленьких батчах)
        self._frozen = True
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def unfreeze(self):
        # разморозить и вернуть train-режим
        self._frozen = False
        for p in self.parameters():
            p.requires_grad = True
        self.train()

    def train(self, mode=True):
        # model.train() не должен включать BatchNorm обратно, пока backbone заморожен
        super().train(mode and not self._frozen)
        return self
