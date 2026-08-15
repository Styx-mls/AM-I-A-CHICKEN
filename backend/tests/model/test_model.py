import torch

from model import ChickenCNN


def test_model_accepts_single_image():
    model = ChickenCNN(num_classes=3)

    input_tensor = torch.randn(
        1,
        3,
        128,
        128,
    )

    output = model(input_tensor)

    assert output.shape == (1, 3)


def test_model_accepts_batch():
    model = ChickenCNN(num_classes=3)

    input_tensor = torch.randn(
        4,
        3,
        128,
        128,
    )

    output = model(input_tensor)

    assert output.shape == (4, 3)


def test_model_has_trainable_parameters():
    model = ChickenCNN(num_classes=3)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    assert parameter_count > 0


def test_model_output_contains_finite_values():
    model = ChickenCNN(num_classes=3)

    input_tensor = torch.randn(
        2,
        3,
        128,
        128,
    )

    output = model(input_tensor)

    assert torch.isfinite(output).all()