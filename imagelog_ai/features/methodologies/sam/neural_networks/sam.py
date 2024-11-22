"""Implementation of the Segment-Anything Model (SAM)"""

# %% [markdown]
# ## Imports

# %%
# Change directory to the root of the project
if __name__ == "__main__":
    import os
    import sys

    import dotenv
    import matplotlib.pyplot as plt
    import numpy as np

    os.chdir(os.getcwd().split("imagelog_ai")[0])
    print(f"cwd: {os.getcwd()}")
    dotenv.load_dotenv()
    sys.path.append(os.getenv("PACKAGEPATH", ""))

    from imagelog_ai.features.methodologies.sam.datasets.dataset import SamDataset

# %%
from typing import Optional

from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import v2 as T

# %% [markdown]
# ## Constants

# %%

# %% [markdown]
# ## Classes

# %%


class Sam(nn.Module):
    """
    Implementation of the SAM (Segment Anything Mask) model.

    Args:
        checkpoint_fname (str, optional): Filename of the pre-trained checkpoint.
            Defaults to "sam_vit_l_0b3195.pth".
        type (str, optional): Type of the SAM model.
            Defaults to "vit_l".
        freeze_image_encoder (bool, optional): Whether to freeze the image encoder.
            Defaults to False.
        freeze_prompt_encoder (bool, optional): Whether to freeze the prompt encoder.
            Defaults to False.
        freeze_mask_decoder (bool, optional): Whether to freeze the mask decoder.
            Defaults to False.

    Attributes:
        checkpoint_fname (str): Filename of the pre-trained checkpoint.
        type (str): Type of the SAM model.
        freeze_image_encoder (bool): Whether the image encoder is frozen.
        freeze_prompt_encoder (bool): Whether the prompt encoder is frozen.
        freeze_mask_decoder (bool): Whether the mask decoder is frozen.
        model (nn.Module): The SAM model.
    """

    def __init__(
        self,
        checkpoint_fname: str = "sam_vit_l_0b3195.pth",
        sam_type: str = "vit_l",
        freeze_image_encoder: bool = False,
        freeze_prompt_encoder: bool = False,
        freeze_mask_decoder: bool = False,
    ):
        super().__init__()
        self.checkpoint_fname: str = checkpoint_fname
        self.sam_type: str = sam_type
        self.freeze_image_encoder: bool = freeze_image_encoder
        self.freeze_prompt_encoder: bool = freeze_prompt_encoder
        self.freeze_mask_decoder: bool = freeze_mask_decoder
        self.setup()

    def setup(self):
        """Set up the SAM model by initializing the model and freezing the specified components."""
        self.model: nn.Module = sam_model_registry[self.sam_type](
            checkpoint=self.checkpoint_fname
        )
        self.model.train()
        if self.freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if self.freeze_mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor,
        masks: Optional[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Forward pass of the SAM model.

        Args:
            images (torch.Tensor): Input images.
            bboxes (torch.Tensor): List of bounding boxes.
            masks (Optional[torch.Tensor]): List of masks.

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: Tuple containing the predicted masks
                and IOU predictions.
        """
        input_images = torch.stack([self.model.preprocess(x) for x in images], dim=0)
        _, _, H, W = input_images.shape
        transform = ResizeLongestSide(self.model.image_encoder.img_size)
        assert (
            H == W == self.model.image_encoder.img_size
        ), "Image size mismatch. Expected 1024x1024. Got {H}x{W}."
        image_embeddings = self.model.image_encoder(input_images)
        # If masks are not given, create a list of None values
        # Resize boxes
        if bboxes is None:
            # bboxes = torch.tensor([[0, 0, H, W]]).unsqueeze(0)
            bboxes = [None] * len(images)
        else:
            bboxes = transform.apply_boxes_torch(
                bboxes, original_size=images.shape[-2:]
            )
            if len(bboxes.shape) == 2:
                bboxes = bboxes.unsqueeze(0)
        if masks is None:
            masks = [None] * len(bboxes)
        else:
            # Resize masks to self.model.image_encoder.img_size // 4
            masks = ResizeLongestSide(
                self.model.image_encoder.img_size // 4
            ).apply_image_torch(masks)
        pred_masks: list[torch.Tensor] = []
        ious: list[torch.Tensor] = []
        for embedding, bbox, mask in zip(image_embeddings, bboxes, masks):
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=bbox,
                masks=mask,
            )
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            masks = self.model.postprocess_masks(
                low_res_masks,
                input_size=(H, W),
                original_size=images.shape[-2:],
            )
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)
        return pred_masks, ious

    def get_predictor(self):
        """Get the predictor for the SAM model.

        Returns:
            SamPredictor: The predictor for the SAM model.
        """
        return SamPredictor(self.model)


class FocalLoss(nn.Module):
    """
    Focal Loss implementation.

    Args:
        weight (Tensor, optional): A manual rescaling weight given to each class.
        size_average (bool, optional):
            If set to True, the losses are averaged over each loss element in the batch.
            If set to False, the losses are summed for each loss element in the batch.

    Attributes:
        None

    Methods:
        forward(inputs, targets, alpha=0.8, gamma=2, smooth=1): Computes the focal loss.

    """

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        """
        Compute the focal loss.

        Args:
            inputs (Tensor): The input tensor.
            targets (Tensor): The target tensor.
            alpha (float, optional): The balancing parameter for positive class.
            gamma (float, optional): The focusing parameter for modulating the loss.
            smooth (float, optional): The smoothing parameter to avoid division by zero.

        Returns:
            Tensor: The computed focal loss.

        """
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        bce = F.binary_cross_entropy(inputs, targets, reduction="none")
        bce_exp = torch.exp(-bce)
        focal_loss = alpha * (1 - bce_exp) ** gamma * bce
        focal_loss = focal_loss.mean()

        return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss implementation for binary segmentation tasks.
    """

    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Compute the Dice Loss between the predicted inputs and the target labels.

        Args:
            inputs (torch.Tensor): Predicted inputs from the neural network.
            targets (torch.Tensor): Target labels.
            smooth (float, optional): Smoothing factor to avoid division by zero.

        Returns:
            torch.Tensor: Computed Dice Loss.
        """
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


# %% [markdown]
# ## Main (for testing)


# %%


if __name__ == "__main__":

    def transform_func(x):
        """
        Transforms the input tensor by selecting the first three channels.

        Args:
            x (torch.Tensor): Input tensor of shape (C, H, W).

        Returns:
            torch.Tensor: Transformed tensor of shape (3, H, W).
        """
        # Select channels
        x = x[:3, :, :]
        return x

    def target_transform_func(x):
        """
        Applies transformations to the target data.

        Args:
            x (dict): A dictionary containing the target data.

        Returns:
            dict: The transformed target data.
        """
        # Select channel
        x["boxes"] = x["boxes"]
        x["masks"] = x["masks"]
        # Resize masks to 256x256
        x["masks"] = T.Resize((256, 256))(x["masks"])
        # Binarize masks
        # x["masks"] = (x["masks"] > 0).float()
        # Repeat channels
        # repeat_transform = RepeatChannels(3)
        # x["masks"] = repeat_transform.forward(x["masks"])
        # x["boxes"] = x["boxes"].repeat(3, 1)
        return x

    # Load dataset with sample data
    project_name: str = "SAMFineTuning"
    preprocess_name: str = "lstudio"
    list_datasource_names: list[str] = [
        "WellD",
    ]
    class_list: list[str] = [
        "camada condutiva",
        "fratura condutiva",
        "fratura induzida",
        "fratura parcial",
        "vug",
    ]
    others_class_list: list[str] = ["outros"]
    transform = transform_func
    target_transform = target_transform_func

    dataset = SamDataset(
        project_name=project_name,
        preprocess_name=preprocess_name,
        list_datasource_names=list_datasource_names,
        class_list=class_list,
        others_class_list=others_class_list,
        transform=transform,
        target_transform=target_transform,
        target_boxes=True,
        target_labels=True,
        target_masks=True,
        boxes_location="masks",
        masks_location="labels",
    )

# %%
if __name__ == "__main__":
    print(f"Dataset length: {len(dataset)}")
    img, labels = dataset[np.random.randint(0, len(dataset))]
    print(f"Sample image shape: {img.shape}")
    print(f"Sample mask shape: {labels['masks'].shape}")
    print(f"Samples boxes shape: {labels['boxes'].shape}")
    # Plot sample image with box, and image with mask
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(T.ToPILImage()(img))
    axs[0].add_patch(
        plt.Rectangle(
            (labels["boxes"][0][0], labels["boxes"][0][1]),
            labels["boxes"][0][2] - labels["boxes"][0][0],
            labels["boxes"][0][3] - labels["boxes"][0][1],
            edgecolor="r",
            facecolor="none",
        )
    )
    axs[1].imshow(labels["masks"][0], cmap="Oranges")
    plt.show()

# %%
if __name__ == "__main__":
    # Instantiate SAM model
    sam = Sam(
        checkpoint_fname="data/PretrainedModels/sam_vit_l_0b3195.pth",
        sam_type="vit_l",
        freeze_image_encoder=True,
        freeze_prompt_encoder=True,
        freeze_mask_decoder=False,
    )
    # print(sam)

    # Try forward pass with random data
    images = torch.randn(1, 3, 1024, 1024)
    bboxes = torch.tensor([[0, 0, 500, 500]]).unsqueeze(0)
    # ! Important: masks should be exactly 4x smaller than the input image
    # ! As defined in the SAM model
    masks = torch.randn(1, 1024 // 4, 1024 // 4).unsqueeze(0)
    print(f"Sample image shape: {images.shape}")
    print(f"Sample boxes shape: {bboxes.shape}")
    print(f"Sample boxes: {bboxes}")
    print(f"Sample mask shape: {masks.shape}")
    pred_masks, ious = sam(images, bboxes, None)
    print(f"Predicted masks: {len(pred_masks)}")
    print(f"Predicted mask shape: {pred_masks[0].shape}")
    print(f"IOU predictions: {ious}")
    # Plot sample image with box, and image with mask
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(T.ToPILImage()(images[0]))
    axs[0].add_patch(
        plt.Rectangle(
            (bboxes[0][0][0], bboxes[0][0][1]),
            bboxes[0][0][2] - bboxes[0][0][0],
            bboxes[0][0][3] - bboxes[0][0][1],
            edgecolor="r",
            facecolor="none",
        )
    )
    axs[1].imshow(masks[0][0], cmap="Oranges")
    axs[2].imshow(pred_masks[0].detach().numpy()[0], cmap="Oranges")
    plt.show()

# %%
if __name__ == "__main__":
    # Test forward with samples from dataset
    img, labels = dataset[np.random.randint(0, len(dataset))]
    img = img.unsqueeze(0)
    bboxes = labels["boxes"].unsqueeze(0)
    masks = labels["masks"].unsqueeze(0)
    print(f"Sample image shape: {img.shape}")
    print(f"Sample boxes shape: {bboxes.shape}")
    print(f"Sample boxes: {bboxes}")
    print(f"Sample mask shape: {masks.shape}")
    pred_masks, ious = sam(img, bboxes, None)
    print(f"Predicted masks: {len(pred_masks)}")
    print(f"Predicted mask shape: {pred_masks[0].shape}")
    print(f"IOU predictions: {ious}")
    # Plot sample image with box, and image with mask
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(T.ToPILImage()(img[0]))
    axs[0].add_patch(
        plt.Rectangle(
            (bboxes[0][0][0], bboxes[0][0][1]),
            bboxes[0][0][2] - bboxes[0][0][0],
            bboxes[0][0][3] - bboxes[0][0][1],
            edgecolor="r",
            facecolor="none",
        )
    )
    axs[1].imshow(masks[0][0], cmap="Oranges")
    axs[2].imshow(pred_masks[0].detach().numpy()[0], cmap="Oranges")
    plt.show()


# %%
