"""This module implements the Dinov2 model for semantic segmentation."""

import torch
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput


class LinearClassifier(torch.nn.Module):
    """A linear classifier implemented using a 2D convolutional layer.

    Args:
        in_channels (int): Number of input channels.
        tokenW (int, optional): Width of the token. Default is 32.
        tokenH (int, optional): Height of the token. Default is 32.
        num_labels (int, optional): Number of output labels. Default is 1.

    Attributes:
        in_channels (int): Number of input channels.
        width (int): Width of the token.
        height (int): Height of the token.
        classifier (torch.nn.Conv2d): Convolutional layer for classification.

    Methods:
        forward(embeddings):
            Forward pass of the classifier.
            Args:
                embeddings (torch.Tensor): Input embeddings of shape
                    (batch_size, height * width * in_channels).
            Returns:
                torch.Tensor: Output of the classifier.
    """

    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
        super().__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1, 1))

    def forward(self, embeddings):
        """Forward pass for the neural network.

        Args:
            embeddings (torch.Tensor): Input tensor of shape
                (batch_size, height * width * in_channels).

        Returns:
            torch.Tensor: Output tensor after applying the classifier.
        """
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

        return self.classifier(embeddings)


class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
    """Dinov2ForSemanticSegmentation is a model for performing semantic segmentation using
        the Dinov2 architecture.

    Args:
        config (Config): Configuration object containing model parameters.

    Attributes:
        dinov2 (Dinov2Model): The Dinov2 model used to extract features from input images.
        classifier (LinearClassifier): A linear classifier that maps patch embeddings to
            segmentation logits.

    Methods:
        forward(pixel_values, output_hidden_states=False, output_attentions=False, labels=None):
            Performs a forward pass of the model.

            Args:
                pixel_values (torch.Tensor): Input images as a tensor of shape
                    (batch_size, num_channels, height, width).
                output_hidden_states (bool, optional): Whether to return the hidden states.
                    Defaults to False.
                output_attentions (bool, optional): Whether to return the attention weights.
                    Defaults to False.
                labels (torch.Tensor, optional): Ground truth labels for the input images.
                    Defaults to None.

            Returns:
                SemanticSegmenterOutput: An object containing the loss (if labels are provided),
                    logits, hidden states, and attentions.
    """

    def __init__(self, config):
        super().__init__(config)

        self.dinov2 = Dinov2Model(config)
        self.classifier = LinearClassifier(
            config.hidden_size, 32, 32, config.num_labels
        )

    def forward(
        self,
        pixel_values,
        output_hidden_states=False,
        output_attentions=False,
        labels=None,
    ):
        """Perform a forward pass through the model.

        Args:
            pixel_values (torch.Tensor): Input pixel values.
            output_hidden_states (bool, optional): Whether to output hidden states.
                Defaults to False.
            output_attentions (bool, optional): Whether to output attentions.
                Defaults to False.
            labels (torch.Tensor, optional): Ground truth labels for computing the loss.
                Defaults to None.

        Returns:
            SemanticSegmenterOutput: The output of the semantic segmentation model, including loss,
                logits, hidden states, and attentions.
        """
        # if labels is not a list, convert it to a list
        if labels is not None and not isinstance(labels, list):
            labels = [labels]
        # use frozen features
        outputs = self.dinov2(
            pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        # get the patch embeddings - so we exclude the CLS token
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]

        # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(patch_embeddings)
        logits = torch.nn.functional.interpolate(
            logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False
        )

        loss = None
        if labels is not None:
            gt_masks = [labels[i]["gt_mask"] for i in range(len(labels))]
            gt_masks = torch.stack(gt_masks)
            gt_masks = torch.squeeze(gt_masks, dim=1)
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, gt_masks)

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
