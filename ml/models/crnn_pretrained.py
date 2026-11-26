import torch
from torch import nn

from .backbone import ResNetBackbone
from .rnn import BiLSTM


class CRNNPretrained(nn.Module):
#  ResNet + BiLSTM + Linear.  Вид: (B, 1, H, W) -> (T, B, num_classes), где T = ceil(W / 4), это log-вероятности для CTC loss
#  Выход такой же, как у CRNN из crnn.py, поэтому train.py работает без изменений.

    def __init__(self, num_classes, backbone="resnet34", last_layer="layer3",
                 hidden_size=256, pretrained=True):
        super().__init__()
        # параметры архитектуры для checkpoint (при загрузке весов создавать с pretrained=False)
        self.config = dict(num_classes=num_classes, backbone=backbone,
                           last_layer=last_layer, hidden_size=hidden_size)
        self.backbone = ResNetBackbone(backbone, last_layer, pretrained)
        self.rnn = BiLSTM(self.backbone.out_channels, hidden_size)
        self.fc = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x):
        x = self.backbone(x)                # (B, C, 1, T)
        x = x.squeeze(2)                    # (B, C, T)
        x = x.permute(0, 2, 1)              # (B, T, C)
        x = self.rnn(x)                     # (B, T, 2 * hidden)
        x = self.fc(x)                      # (B, T, num_classes)
        x = torch.log_softmax(x, dim=2)
        return x.permute(1, 0, 2)           # (T, B, num_classes)

    @staticmethod
    def output_lengths(image_widths):
        return ResNetBackbone.output_lengths(image_widths)

    def freeze_backbone(self):
        self.backbone.freeze()

    def unfreeze_backbone(self):
        self.backbone.unfreeze()
