import pytest
import torch
from torch import nn

from ml.models.backbone import ResNetBackbone
from ml.models.crnn_pretrained import CRNNPretrained

NUM_CLASSES = 80  # blank + символы


@pytest.mark.parametrize("name,last_layer,channels", [
    ("resnet18", "layer3", 256),
    ("resnet34", "layer4", 512),
    ("resnet50", "layer3", 1024),   
])
@pytest.mark.parametrize("height", [32, 64])
def test_backbone_shape(name, last_layer, channels, height):
    model = ResNetBackbone(name, last_layer, pretrained=False).eval()
    with torch.no_grad():
        y = model(torch.randn(2, 1, height, 130))
    assert model.out_channels == channels
    assert y.shape == (2, channels, 1, 33)          


@pytest.mark.parametrize("width", [128, 130, 131, 257, 1024])
def test_time_steps_match_output_lengths(width):
    model = CRNNPretrained(NUM_CLASSES, "resnet18", pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, 64, width))
    assert out.shape == (out.shape[0], 2, NUM_CLASSES)
    assert out.shape[0] == int(model.output_lengths(torch.tensor(width)))
    assert torch.allclose(out.exp().sum(-1), torch.ones(out.shape[0], 2), atol=1e-4)  


def test_works_with_ctc_loss():
    model = CRNNPretrained(NUM_CLASSES, "resnet18", pretrained=False)
    images = torch.randn(2, 1, 64, 128)
    image_lengths = torch.tensor([128, 90])
    targets = torch.randint(1, NUM_CLASSES, (2, 10))
    target_lengths = torch.tensor([10, 7])

    log_probs = model(images)
    input_lengths = model.output_lengths(image_lengths)
    loss = nn.CTCLoss(blank=0, zero_infinity=True)(log_probs, targets, input_lengths, target_lengths)
    assert torch.isfinite(loss)
    loss.backward()


def test_freeze_unfreeze():
    model = CRNNPretrained(NUM_CLASSES, "resnet18", pretrained=False)
    model.train()
    model.freeze_backbone()
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.rnn.parameters())

    model.train()                                    
    assert not model.backbone.training
    assert model.rnn.training

    before = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    model(torch.randn(2, 1, 64, 128))
    after = model.backbone.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)   

    model.unfreeze_backbone()
    assert all(p.requires_grad for p in model.backbone.parameters())
    assert model.backbone.training
