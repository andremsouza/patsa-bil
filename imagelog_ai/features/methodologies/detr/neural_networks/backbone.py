# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from imagelog_ai.features.methodologies.detr.utils.misc import (
    NestedTensor,
    is_main_process,
)

from imagelog_ai.features.methodologies.detr.neural_networks.position_encoding import (
    build_position_encoding,
)


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        """
        Forward pass for the neural network layer.
        This method performs the forward pass of the neural network layer by applying
        reshaping operations to the weights, biases, running variance, and running mean
        to make the computation fuser-friendly. It then calculates the scale and bias
        for the input tensor and returns the transformed tensor.
        Args:
            x (torch.Tensor): Input tensor to the layer.
        Returns:
            torch.Tensor: Transformed tensor after applying the scale and bias.
        """

        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    """
    A base class for backbone networks used in neural networks.
    Args:
        backbone (nn.Module): The backbone network.
        train_backbone (bool): If True, allows training of the backbone layers.
        num_channels (int): The number of channels in the output feature maps.
        return_interm_layers (bool): If True, returns intermediate layers.
    Attributes:
        body (IntermediateLayerGetter): A module that returns the specified layers
            from the backbone.
        num_channels (int): The number of channels in the output feature maps.
    Methods:
        forward(tensor_list: NestedTensor) -> Dict[str, NestedTensor]:
            Forward pass of the backbone network. Takes a NestedTensor as input and returns
            a dictionary of NestedTensors containing the output feature maps and
            their corresponding masks.
    """

    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        """Forward pass for the backbone network.
        Args:
            tensor_list (NestedTensor): A NestedTensor object containing the input tensors/masks.
        Returns:
            Dict[str, NestedTensor]: A dictionary where keys are layer names and values are
                NestedTensor objects containing the output tensors and corresponding masks.
        """

        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(),
            norm_layer=FrozenBatchNorm2d,
        )
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    """A custom neural network module that joins a backbone and a position embedding.
    Args:
        backbone (nn.Module): The backbone neural network module.
        position_embedding (nn.Module): The position embedding module.
    Methods:
        forward(tensor_list: NestedTensor) -> Tuple[List[NestedTensor], List[Tensor]]:
            Processes the input tensor_list through the backbone and position embedding modules.
            Args:
                tensor_list (NestedTensor): The input tensor list.
            Returns:
                Tuple[List[NestedTensor], List[Tensor]]: A tuple containing the output from
                the backbone and the position embeddings.
    """

    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        """
        Forward pass for the backbone network.
        Args:
            tensor_list (NestedTensor): Input tensor wrapped in a NestedTensor object.
        Returns:
            Tuple[List[NestedTensor], List[Tensor]]: A tuple containing:
                - A list of NestedTensor objects after processing through the backbone network.
                - A list of position encoded tensors corresponding to the processed NestedTensors.
        """

        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for _, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    """
    Builds the backbone model for a neural network.
    Args:
        args: An object containing the following attributes:
            - lr_backbone (float): Learning rate for the backbone.
                If greater than 0, the backbone will be trained.
            - masks (bool): If True, return intermediate layers.
            - backbone (str): The type of backbone to use.
            - dilation (bool): If True, apply dilation in the backbone.
    Returns:
        model: A model object that combines the backbone and position encoding,
            with the number of channels set to the backbone's number of channels.
    """

    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(
        args.backbone, train_backbone, return_interm_layers, args.dilation
    )
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
