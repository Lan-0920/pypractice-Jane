import torch
from banknote_classifier.models import VGG16


def test_vgg16_instantiation():
    """VGG16 應能成功建立，且為 torch.nn.Module 的子類別"""
    model = VGG16()
    assert isinstance(model, torch.nn.Module)


def test_vgg16_output_shape():
    """VGG16 在 CPU 上的 forward pass 輸出 shape 應符合預期"""
    model = VGG16()
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape[0] == 1   # batch size
    assert output.shape[1] == 9   # num_classes


def test_vgg16_output_is_tensor():
    """VGG16 的輸出應為 torch.Tensor"""
    model = VGG16()
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert isinstance(output, torch.Tensor)