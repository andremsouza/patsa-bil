# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from imagelog_ai.features.methodologies.detr.utils.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        """
        Computes the positional encoding for the input tensor.
        Args:
            tensor_list (NestedTensor): A NestedTensor object containing:
                - tensors (torch.Tensor): The input tensor of shape
                    (batch_size, channels, height, width).
                - mask (torch.Tensor): A mask tensor of shape (batch_size, height, width)
                    indicating the valid positions.
        Returns:
            torch.Tensor: The positional encoding tensor of shape
                (batch_size, num_pos_feats*2, height, width).
        """

        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """PositionEmbeddingLearned is a neural network module that generates learned positional
        embeddings for input tensors.
    Attributes:
        row_embed (nn.Embedding): Embedding layer for row positions.
        col_embed (nn.Embedding): Embedding layer for column positions.
    Methods:
        __init__(num_pos_feats=256):
            Initializes the PositionEmbeddingLearned module with the specified number of
                positional features.
        reset_parameters():
            Initializes the weights of the embedding layers with a uniform distribution.
        forward(tensor_list: NestedTensor):
            Generates positional embeddings for the input tensor and returns them.
            Args:
                tensor_list (NestedTensor): A NestedTensor object containing the input tensor.
            Returns:
                torch.Tensor: A tensor containing the positional embeddings.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        """Resets the parameters of the position encoding layers by initializing
        the weights of row_embed and col_embed with a uniform distribution.
        This method uses PyTorch's nn.init.uniform_ to fill the weights of
        row_embed and col_embed with values drawn from a uniform distribution
        over the interval [0, 1).
        Returns:
            None
        """

        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        """Computes the positional encoding for the given tensor list.
        Args:
            tensor_list (NestedTensor): A NestedTensor object containing:
                - tensors (torch.Tensor): The input tensor of shape
                    (batch_size, channels, height, width).
        Returns:
            torch.Tensor: The positional encoding tensor of shape
                          (batch_size, 2*embedding_dim, height, width), where embedding_dim is
                          the dimension of the embeddings.
        """
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1, 1)
        )
        return pos


def build_position_encoding(args):
    """Build the position encoding based on the specified arguments.
    Parameters:
    args (Namespace): A namespace object containing the following attributes:
        - hidden_dim (int): The dimension of the hidden layer.
        - position_embedding (str): The type of position embedding to use.
          It can be "v2" or "sine" for sine-based position embedding,
          or "v3" or "learned" for learned position embedding.
    Returns:
    PositionEmbedding: An instance of the position embedding class based on the specified type.
    Raises:
    ValueError: If the specified position_embedding type is not supported.
    """

    n_steps = args.hidden_dim // 2
    if args.position_embedding in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(n_steps, normalize=True)
    elif args.position_embedding in ("v3", "learned"):
        position_embedding = PositionEmbeddingLearned(n_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
