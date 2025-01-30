# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):
    """
    Transformer model for sequence-to-sequence tasks.
    Args:
        d_model (int): The number of expected features in the input (default=512).
        nhead (int): The number of heads in the multiheadattention models (default=8).
        num_encoder_layers (int): The number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers (int): The number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward (int): The dimension of the feedforward network model (default=2048).
        dropout (float): The dropout value (default=0.1).
        activation (str): The activation function of the intermediate layer, relu or gelu
            (default="relu").
        normalize_before (bool): Whether to apply layer normalization before each sub-layer
            (default=False).
        return_intermediate_dec (bool): Whether to return intermediate decoder layers
            (default=False).
    Attributes:
        encoder (TransformerEncoder): The encoder part of the transformer.
        decoder (TransformerDecoder): The decoder part of the transformer.
        d_model (int): The number of expected features in the input.
        nhead (int): The number of heads in the multiheadattention models.
    Methods:
        _reset_parameters(): Initializes the parameters of the model.
        forward(src, mask, query_embed, pos_embed): Defines the computation performed at evry call.
            Args:
                src (Tensor): The sequence to the encoder (required).
                mask (Tensor): The mask for the src sequence (required).
                query_embed (Tensor): The query embedding (required).
                pos_embed (Tensor): The positional encoding (required).
            Returns:
                Tuple[Tensor, Tensor]: The output of the decoder and the memory from the encoder.
    """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        """
        Forward pass for the transformer model.
        Args:
            src (torch.Tensor): The input tensor with shape
                (batch_size, channels, height, width).
            mask (torch.Tensor): The mask tensor with shape
                (batch_size, height * width).
            query_embed (torch.Tensor): The query embeddings with shape
                (num_queries, embedding_dim).
            pos_embed (torch.Tensor): The positional embeddings with shape
                (batch_size, channels, height, width).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - hs (torch.Tensor): The output tensor from the decoder with shape
                    (num_queries, batch_size, embedding_dim).
                - memory (torch.Tensor): The output tensor from the encoder with shape
                    (batch_size, channels, height, width).
        """

        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):
    """TransformerEncoder is a stack of N encoder layers.
    Args:
        encoder_layer (nn.Module): An instance of the encoder layer to be used.
        num_layers (int): The number of sub-encoder-layers in the encoder.
        norm (Optional[nn.Module]): The normalization layer to be applied after the
            last encoder layer (default=None).
    Methods:
        forward(src, mask=None, src_key_padding_mask=None, pos=None):
            Passes the input through the encoder layers in turn.
            Args:
                src (Tensor): The sequence to the encoder (required).
                mask (Optional[Tensor]): The mask for the src sequence (default=None).
                src_key_padding_mask (Optional[Tensor]): The mask for the src keys per batch
                    (default=None).
                pos (Optional[Tensor]): The positional encodings (default=None).
            Returns:
                Tensor: The encoded output.
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        """Forward pass of the transformer model.
        Args:
            src (Tensor): The input tensor of shape (sequence_length, batch_size, embedding_dim).
            mask (Optional[Tensor], optional): The mask tensor of shape
                (sequence_length, sequence_length) to prevent attention to certain positions.
                Defaults to None.
            src_key_padding_mask (Optional[Tensor], optional): The mask tensor of shape
                (batch_size, sequence_length) to prevent attention to padding tokens.
                Defaults to None.
            pos (Optional[Tensor], optional): The positional encoding tensor of shape
                (sequence_length, batch_size, embedding_dim). Defaults to None.
        Returns:
            Tensor: The output tensor of shape (sequence_length, batch_size, embedding_dim)
                after passing through the transformer layers and normalization (if applicable).
        """
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    """Transformer Decoder module.
    Args:
        decoder_layer (nn.Module): An instance of a decoder layer.
        num_layers (int): The number of decoder layers.
        norm (Optional[nn.Module]): Normalization layer to apply after each decoder layer.
            Default is None.
        return_intermediate (bool): Whether to return intermediate outputs from each layer.
            Default is False.
    Methods:
        forward(tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
            memory_key_padding_mask=None, pos=None, query_pos=None
        ):
            Forward pass through the Transformer Decoder.
            Args:
                tgt (Tensor): Target sequence.
                memory (Tensor): Memory sequence from the encoder.
                tgt_mask (Optional[Tensor]): Mask for the target sequence. Default is None.
                memory_mask (Optional[Tensor]): Mask for the memory sequence. Default is None.
                tgt_key_padding_mask (Optional[Tensor]): Padding mask for the target sequence.
                    Default is None.
                memory_key_padding_mask (Optional[Tensor]): Padding mask for the memory sequence.
                    Default is None.
                pos (Optional[Tensor]): Positional encoding for the memory sequence.
                    Default is None.
                query_pos (Optional[Tensor]): Positional encoding for the target sequence.
                    Default is None.
            Returns:
                Tensor: The output of the Transformer Decoder. If return_intermediate is True,
                    returns a stacked tensor of intermediate outputs.
    """

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        """Forward pass for the transformer decoder.
        Args:
            tgt (Tensor): The target sequence.
            memory (Tensor): The memory sequence from the encoder.
            tgt_mask (Optional[Tensor], optional): Mask for the target sequence.
                Defaults to None.
            memory_mask (Optional[Tensor], optional): Mask for the memory sequence.
                Defaults to None.
            tgt_key_padding_mask (Optional[Tensor], optional): Padding mask for the target keys.
                Defaults to None.
            memory_key_padding_mask (Optional[Tensor], optional): Padding mask for the memory keys.
                Defaults to None.
            pos (Optional[Tensor], optional): Positional encoding for the memory sequence.
                Defaults to None.
            query_pos (Optional[Tensor], optional): Positional encoding for the target sequence.
                Defaults to None.
        Returns:
            Tensor: The output of the transformer decoder. If `self.return_intermediate` is True,
                returns a stacked tensor of intermediate outputs. Otherwise, returns the final
                output with an additional dimension.
        """
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer is made up of self-attention and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
        d_model (int): the number of expected features in the input (required).
        nhead (int): the number of heads in the multiheadattention models (required).
        dim_feedforward (int): the dimension of the feedforward network model (default=2048).
        dropout (float): the dropout value (default=0.1).
        activation (str): the activation function of intermediate layer, relu or gelu
            (default=relu).
        normalize_before (bool): whether to apply layer normalization before or after the
            attention and feedforward operations (default=False).
    Methods:
        with_pos_embed(tensor, pos):
            Add positional embedding to the input tensor.
                pos (Optional[Tensor]): The positional embedding tensor.
                    If None, the input tensor is returned as is.
                Tensor: The input tensor with the positional embedding added, or the
                    input tensor itself if pos is None.
        forward_post(src, src_mask=None, src_key_padding_mask=None, pos=None):
            Forward pass when normalization is applied after attention and feedforward operations.
                src (Tensor): The input tensor.
                src_mask (Optional[Tensor]): The mask for the src sequence (optional).
                src_key_padding_mask (Optional[Tensor]): The mask for the src keys per batch
                    (optional).
                pos (Optional[Tensor]): The positional encoding (optional).
                Tensor: The output tensor.
        forward_pre(src, src_mask=None, src_key_padding_mask=None, pos=None):
            Forward pass when normalization is applied before attention and feedforward operations.
                src (Tensor): The input tensor.
                src_mask (Optional[Tensor]): The mask for the src sequence (optional).
                src_key_padding_mask (Optional[Tensor]): The mask for the src keys per batch
                    (optional).
                pos (Optional[Tensor]): The positional encoding (optional).
                Tensor: The output tensor.
        forward(src, src_mask=None, src_key_padding_mask=None, pos=None):
            Forward pass for the encoder layer.
                src (Tensor): The input tensor.
                src_mask (Optional[Tensor]): The mask for the src sequence (optional).
                src_key_padding_mask (Optional[Tensor]): The mask for the src keys per batch
                    (optional).
                pos (Optional[Tensor]): The positional encoding (optional).
                Tensor: The output tensor.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """Add positional embedding to the input tensor.
        Args:
            tensor (Tensor): The input tensor to which the positional embedding will be added.
            pos (Optional[Tensor]): The positional embedding tensor.
                If None, the input tensor is returned as is.
        Returns:
            Tensor: The input tensor with the positional embedding added,
                or the input tensor itself if pos is None.
        """

        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        """Forward pass for the transformer layer with post-normalization.
        Args:
            src (Tensor): The input tensor of shape (sequence_length, batch_size, embedding_dim).
            src_mask (Optional[Tensor], optional): The mask for the src sequence to prevent
                attention to certain positions. Default is None.
            src_key_padding_mask (Optional[Tensor], optional): The mask for the src keys per batch
             to prevent attention to padding tokens. Default is None.
            pos (Optional[Tensor], optional): The positional encoding tensor of shape
                (sequence_length, batch_size, embedding_dim). Default is None.
        Returns:
            Tensor: The output tensor of the same shape as `src`.
        """
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        """Perform the forward pass for the pre-normalization transformer layer.
        Args:
            src (Tensor): The input tensor.
            src_mask (Optional[Tensor], optional): The source mask tensor. Defaults to None.
            src_key_padding_mask (Optional[Tensor], optional): The source key padding mask tensor.
                Defaults to None.
            pos (Optional[Tensor], optional): The positional encoding tensor. Defaults to None.
        Returns:
            Tensor: The output tensor after applying the transformer layer.
        """
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        """Forward pass for the transformer model.
        Args:
            src (Tensor): The input source tensor.
            src_mask (Optional[Tensor], optional): The source mask tensor. Default is None.
            src_key_padding_mask (Optional[Tensor], optional): The source key padding mask tensor.
                Default is None.
            pos (Optional[Tensor], optional): The positional encoding tensor. Default is None.
        Returns:
            Tensor: The output tensor after applying the transformer model.
        """
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    """TransformerDecoderLayer is a single layer of the transformer decoder.
    Args:
        d_model (int): The number of expected features in the input (required).
        nhead (int): The number of heads in the multiheadattention models (required).
        dim_feedforward (int): The dimension of the feedforward network model (default=2048).
        dropout (float): The dropout value (default=0.1).
        activation (str): The activation function of the intermediate layer,
            either "relu" or "gelu" (default="relu").
        normalize_before (bool): If True, normalization is done before each sub-layer,
            otherwise after (default=False).
    Attributes:
        self_attn (nn.MultiheadAttention): Self-attention layer.
        multihead_attn (nn.MultiheadAttention): Multi-head attention layer.
        linear1 (nn.Linear): First linear layer of the feedforward network.
        dropout (nn.Dropout): Dropout layer.
        linear2 (nn.Linear): Second linear layer of the feedforward network.
        norm1 (nn.LayerNorm): Layer normalization after self-attention.
        norm2 (nn.LayerNorm): Layer normalization after multi-head attention.
        norm3 (nn.LayerNorm): Layer normalization after feedforward network.
        dropout1 (nn.Dropout): Dropout layer after self-attention.
        dropout2 (nn.Dropout): Dropout layer after multi-head attention.
        dropout3 (nn.Dropout): Dropout layer after feedforward network.
        activation (Callable): Activation function.
        normalize_before (bool): Whether to normalize before each sub-layer.
    Methods:
        with_pos_embed(tensor, pos):
            Adds positional embedding to the tensor if pos is not None.
        forward_post(tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
            memory_key_padding_mask=None, pos=None, query_pos=None
        ):
            Forward pass when normalization is done after each sub-layer.
        forward_pre(tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
            memory_key_padding_mask=None, pos=None, query_pos=None
        ):
            Forward pass when normalization is done before each sub-layer.
        forward(tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
            memory_key_padding_mask=None, pos=None, query_pos=None
        ):
            Forward pass that selects between forward_pre and forward_post based
                on normalize_before.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """Add positional embedding to the input tensor.
        Args:
            tensor (Tensor): The input tensor to which the positional embedding will be added.
            pos (Optional[Tensor]): The positional embedding tensor.
                If None, the input tensor is returned as is.
        Returns:
            Tensor: The resulting tensor after adding the positional embedding,
                or the original tensor if pos is None.
        """

        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        """Forward pass for the transformer decoder layer with post-normalization.
        Args:
            tgt (Tensor): The target sequence (decoder input).
            memory (Tensor): The memory sequence (encoder output).
            tgt_mask (Optional[Tensor], optional): Mask for the target sequence.
                Defaults to None.
            memory_mask (Optional[Tensor], optional): Mask for the memory sequence.
                Defaults to None.
            tgt_key_padding_mask (Optional[Tensor], optional): Padding mask for the target sequence
                Defaults to None.
            memory_key_padding_mask (Optional[Tensor], optional): Padding mask for the
                memory sequence. Defaults to None.
            pos (Optional[Tensor], optional): Positional encoding for the memory sequence.
                Defaults to None.
            query_pos (Optional[Tensor], optional): Positional encoding for the target sequence.
                Defaults to None.
        Returns:
            Tensor: The output of the transformer decoder layer.
        """
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        """Forward pass for the pre-processing step of the transformer decoder layer.
        Args:
            tgt (Tensor): The target sequence.
            memory (Tensor): The sequence from the encoder.
            tgt_mask (Optional[Tensor], optional): Mask for the target sequence.
                Defaults to None.
            memory_mask (Optional[Tensor], optional): Mask for the memory sequence.
                Defaults to None.
            tgt_key_padding_mask (Optional[Tensor], optional): Padding mask for the target keys.
                Defaults to None.
            memory_key_padding_mask (Optional[Tensor], optional): Padding mask for the memory keys.
                Defaults to None.
            pos (Optional[Tensor], optional): Positional encoding for the memory sequence.
                Defaults to None.
            query_pos (Optional[Tensor], optional): Positional encoding for the target sequence.
                Defaults to None.
        Returns:
            Tensor: The processed target sequence.
        """
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        """Forward pass for the transformer model.
        Args:
            tgt (Tensor): The target sequence.
            memory (Tensor): The sequence from the encoder.
            tgt_mask (Optional[Tensor], optional): The mask for the target sequence.
                Defaults to None.
            memory_mask (Optional[Tensor], optional): The mask for the memory sequence.
                Defaults to None.
            tgt_key_padding_mask (Optional[Tensor], optional): The padding mask for the target keys
                Defaults to None.
            memory_key_padding_mask (Optional[Tensor], optional): The padding mask for the
                memory keys. Defaults to None.
            pos (Optional[Tensor], optional): The positional encoding for the memory sequence.
                Defaults to None.
            query_pos (Optional[Tensor], optional): The positional encoding for the target sequence
                Defaults to None.
        Returns:
            Tensor: The output of the transformer model.
        """
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


def _get_clones(module, num):
    return nn.ModuleList([copy.deepcopy(module) for i in range(num)])


def build_transformer(args):
    """Builds a Transformer model based on the provided arguments.
    Args:
        args: An object containing the following attributes:
            hidden_dim (int): The dimension of the model.
            dropout (float): The dropout rate.
            nheads (int): The number of attention heads.
            dim_feedforward (int): The dimension of the feedforward network.
            enc_layers (int): The number of encoder layers.
            dec_layers (int): The number of decoder layers.
            pre_norm (bool): Whether to apply normalization before the attention and feedforward
                layers.
    Returns:
        Transformer: An instance of the Transformer model configured with the specified parameters.
    """

    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
