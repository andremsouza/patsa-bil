"""This module contains the SAM model training script."""

# %% [markdown]
# ## Imports

# %%
# Change directory to the root of the project
if __name__ == "__main__":
    import os
    import sys

    import dotenv

    os.chdir(os.getcwd().split("/patsa-bil")[0] + "/patsa-bil")
    print(f"cwd: {os.getcwd()}")
    dotenv.load_dotenv()
    sys.path.append(os.getenv("PACKAGEPATH", ""))

# %%
import os


import lightning.pytorch as pl
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from skimage.color import label2rgb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch

from imagelog_ai.features.methodologies.sam.datasets.dataset import SamDataset
from imagelog_ai.features.methodologies.sam.modules.lit_module import SamModule
from imagelog_ai.utils.io_functions import json_load

# %% [markdown]
# ## Constants

# %%
RANDOM_STATE: int = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.set_float32_matmul_precision("high")

# %% [markdown]
# ## Main (Experimentation)

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
        # Invert intensities
        x = x.max() - x
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
        x["boxes"] = x["boxes"].float()
        x["masks"] = x["masks"].float()
        return x

    # Load dataset with sample data
    project_name = "SAM"
    project_settings = json_load(f"experiment_configs/{project_name}.json")

    dataset = SamDataset(
        project_name=project_name,
        preprocess_name=project_settings["preprocess_name"],
        list_datasource_names=project_settings["list_datasource_names"],
        class_list=project_settings["class_list"],
        others_class_list=project_settings["others_class_list"],
        background_class=project_settings["background_class"],
        transform=transform_func,
        target_transform=target_transform_func,
        target_boxes=True,
        target_labels=True,
        target_masks=True,
        boxes_location="masks",
        masks_location="labels",
    )
    # Get dataset.dataframe to separate train and val
    dataframe = dataset.dataframe
    train_dataframe, val_dataframe = train_test_split(
        dataframe, test_size=0.2, random_state=RANDOM_STATE
    )
    train_dataset = SamDataset(
        project_name=project_name,
        preprocess_name=project_settings["preprocess_name"],
        list_datasource_names=project_settings["list_datasource_names"],
        class_list=project_settings["class_list"],
        others_class_list=project_settings["others_class_list"],
        background_class=project_settings["background_class"],
        transform=transform_func,
        target_transform=target_transform_func,
        target_boxes=True,
        target_labels=True,
        target_masks=True,
        boxes_location="masks",
        masks_location="labels",
    )
    train_dataset.dataframe = train_dataframe
    val_dataset = SamDataset(
        project_name=project_name,
        preprocess_name=project_settings["preprocess_name"],
        list_datasource_names=project_settings["list_datasource_names"],
        class_list=project_settings["class_list"],
        others_class_list=project_settings["others_class_list"],
        background_class=project_settings["background_class"],
        transform=transform_func,
        target_transform=target_transform_func,
        target_boxes=True,
        target_labels=True,
        target_masks=True,
        boxes_location="masks",
        masks_location="labels",
    )
    val_dataset.dataframe = val_dataframe

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=project_settings["batch_size"],
        shuffle=True,
        num_workers=os.cpu_count() or 0,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=project_settings["batch_size"],
        shuffle=True,
        num_workers=os.cpu_count() or 0,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=project_settings["batch_size"],
        shuffle=False,
        num_workers=os.cpu_count() or 0,
    )

    # Create model
    model = SamModule(
        early_stopping_patience=2,
        lr_scheduler_patience=1,
        learning_rate=1e-5,
        weight_decay=1e-4,
        checkpoint_fname="data/PretrainedModels/sam_vit_l_0b3195.pth",
        sam_type="vit_l",
        freeze_image_encoder=True,
        freeze_prompt_encoder=True,
        freeze_mask_decoder=False,
    )
    # Load from checkpoint
    # ! Comment these lines to train the model
    # model = SamModule.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
    #     checkpoint_path="models/sam-epoch=11-Loss/Val=0.23.ckpt",
    #     early_stopping_patience=4,
    #     lr_scheduler_patience=2,
    #     learning_rate=1e-4,
    #     weight_decay=1e-4,
    #     checkpoint_fname="data/PretrainedModels/sam_vit_l_0b3195.pth",
    #     sam_type="vit_l",
    #     freeze_image_encoder=True,
    #     freeze_prompt_encoder=True,
    #     freeze_mask_decoder=False,
    # )

    # Create trainer
    trainer = pl.Trainer(
        logger=[
            pl.loggers.CSVLogger("logs", name="sam-finetuning"),
            pl.loggers.MLFlowLogger(
                experiment_name="sam-finetuning", tracking_uri="http://localhost:5000"
            ),
        ],
        max_epochs=100,
    )

# %%
if __name__ == "__main__":
    # Test predict with SamAutomaticMaskGenerator
    batch = dataset[1]
    img, labels = batch
    try:
        batch_bboxes = labels["boxes"].unsqueeze(0)
    except AttributeError as exc:
        print(f"AttributeError: {exc}")
        batch_bboxes = [None] * len(img)
    try:
        batch_masks = labels["masks"].unsqueeze(0)
    except AttributeError as exc:
        print(f"AttributeError: {exc}")
        batch_masks = [None] * len(img)
    predictor = SamAutomaticMaskGenerator(
        model.network.model,
        stability_score_thresh=0.9,
        min_mask_region_area=5,
    )
    masks = []
    images, labels = batch
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    image_original_shape = images.shape[-2:]
    sam_transform = ResizeLongestSide(model.network.model.image_encoder.img_size)
    box = None
    mask = None
    for image, box, mask in zip(images, batch_bboxes, batch_masks):
        image_resized = sam_transform.apply_image(
            image.detach().numpy().transpose(1, 2, 0)
        )
        prediction = predictor.generate(image_resized.astype(np.uint8))
        prediction.sort(key=lambda x: x["area"], reverse=True)
        sam_mask = np.zeros(image_resized.shape, dtype=np.uint8)
        for pred_idx, pred in enumerate(prediction):
            region_mask = pred["segmentation"].astype(np.float32)
            sam_mask[region_mask > 0] = pred_idx + 1
        # Resize mask to original image size
        resize_transform = ResizeLongestSide(max(image_original_shape))
        sam_mask = resize_transform.apply_image(sam_mask)
        masks.append(sam_mask)
    # Plot sample image with box, and image with mask
    print(f"SamAutomaticMaskGenerator masks: {len(masks)}")
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(img[0], cmap="Oranges")
    if box is not None:
        axs[0].add_patch(
            plt.Rectangle(
                (batch_bboxes[0][0][0], batch_bboxes[0][0][1]),
                batch_bboxes[0][0][2] - batch_bboxes[0][0][0],
                batch_bboxes[0][0][3] - batch_bboxes[0][0][1],
                edgecolor="r",
                facecolor="none",
            )
        )
    axs[1].imshow(masks[0][:, :, 0] > 0, cmap="Oranges")
    axs[2].imshow(
        label2rgb(
            label=masks[0][:, :, 0], image=img.numpy().transpose(1, 2, 0), bg_label=0
        )
    )
    plt.show()

# %%
if __name__ == "__main__":
    # Test predict with SamPredictor
    batch = dataset[1]
    img, labels = batch
    try:
        batch_bboxes = labels["boxes"].unsqueeze(0)
    except AttributeError as exc:
        print(f"AttributeError: {exc}")
        batch_bboxes = [None] * len(img)
    try:
        batch_masks = labels["masks"].unsqueeze(0)
    except AttributeError as exc:
        print(f"AttributeError: {exc}")
        batch_masks = [None] * len(img)
    predictor = SamPredictor(model.network.model)
    masks = []
    images, labels = batch
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    image_original_shape = images.shape[-2:]
    for image, box, mask in zip(images, batch_bboxes, batch_masks):
        np_image = image.detach().numpy().transpose(1, 2, 0)
        np_mask = mask.detach().numpy() if mask is not None else None
        np_box = box.detach().numpy() if box is not None else None
        predictor.set_image(np_image)
        pred_masks, pred_qualities, pred_low_res_masks = predictor.predict(
            box=np_box,
            multimask_output=False,
        )
        masks.append(pred_masks)
    # Plot sample image with box, and image with mask
    print(f"SamPredictor masks: {len(masks)}")
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(img[0], cmap="Oranges")
    if box is not None:
        axs[0].add_patch(
            plt.Rectangle(
                (batch_bboxes[0][0][0], batch_bboxes[0][0][1]),
                batch_bboxes[0][0][2] - batch_bboxes[0][0][0],
                batch_bboxes[0][0][3] - batch_bboxes[0][0][1],
                edgecolor="r",
                facecolor="none",
            )
        )
    if mask is not None:
        axs[1].imshow(batch_masks[0][0], cmap="Oranges")
    else:
        # Plot empty
        axs[1].imshow(np.zeros(img[0].shape))
    axs[2].imshow(
        label2rgb(label=masks[0][0], image=img.numpy().transpose(1, 2, 0), bg_label=0)
    )
    plt.show()

# %%
if __name__ == "__main__":
    # Run a validation epoch and collect metrics
    metrics = trainer.validate(model, val_dataloader)
    print(f"Validation metrics: {metrics}")

# %%
if __name__ == "__main__":
    # # Set MLFlow experiment
    mlflow.pytorch.autolog()
    # Train model
    with mlflow.start_run():
        trainer.fit(model, train_dataloader, val_dataloader)
        # Save model to MLFlow
        mlflow.pytorch.log_model(model, "models")
    # Load from checkpoint
    # ! Comment these lines to train the model
    # model = SamModule.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
    #     checkpoint_path="models/sam-epoch=07-Loss_Val=0.43.ckpt",
    #     early_stopping_patience=2,
    #     lr_scheduler_patience=1,
    #     learning_rate=1e-5,
    #     weight_decay=1e-4,
    #     checkpoint_fname="data/PretrainedModels/sam_vit_l_0b3195.pth",
    #     sam_type="vit_l",
    #     freeze_image_encoder=True,
    #     freeze_prompt_encoder=True,
    #     freeze_mask_decoder=False,
    # )

# %%
if __name__ == "__main__":
    # Test predict with SamAutomaticMaskGenerator
    batch = dataset[1]
    img, labels = batch
    try:
        batch_bboxes = labels["boxes"].unsqueeze(0)
    except AttributeError as exc:
        print(f"AttributeError: {exc}")
        batch_bboxes = [None] * len(img)
    try:
        batch_masks = labels["masks"].unsqueeze(0)
    except AttributeError as exc:
        print(f"AttributeError: {exc}")
        batch_masks = [None] * len(img)
    predictor = SamAutomaticMaskGenerator(
        model.network.model,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.85,
    )
    masks = []
    images, labels = batch
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    image_original_shape = images.shape[-2:]
    sam_transform = ResizeLongestSide(model.network.model.image_encoder.img_size)
    for image, box, mask in zip(images, batch_bboxes, batch_masks):
        image_resized = sam_transform.apply_image(
            image.detach().numpy().transpose(1, 2, 0)
        )
        prediction = predictor.generate(image_resized.astype(np.uint8))
        prediction.sort(key=lambda x: x["area"], reverse=True)
        sam_mask = np.zeros(image_resized.shape, dtype=np.uint8)
        for pred_idx, pred in enumerate(prediction):
            region_mask = pred["segmentation"].astype(np.float32)
            sam_mask[region_mask > 0] = pred_idx + 1
        # Resize mask to original image size
        resize_transform = ResizeLongestSide(max(image_original_shape))
        sam_mask = resize_transform.apply_image(sam_mask)
        masks.append(sam_mask)
    # Plot sample image with box, and image with mask
    print(f"SamAutomaticMaskGenerator masks: {len(masks)}")
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(img[0], cmap="Oranges")
    if box is not None:
        axs[0].add_patch(
            plt.Rectangle(
                (batch_bboxes[0][0][0], batch_bboxes[0][0][1]),
                batch_bboxes[0][0][2] - batch_bboxes[0][0][0],
                batch_bboxes[0][0][3] - batch_bboxes[0][0][1],
                edgecolor="r",
                facecolor="none",
            )
        )
    axs[1].imshow(masks[0][:, :, 0] > 0, cmap="Oranges")
    axs[2].imshow(
        label2rgb(
            label=masks[0][:, :, 0], image=img.numpy().transpose(1, 2, 0), bg_label=0
        )
    )
    plt.show()

# %%
if __name__ == "__main__":
    # Test predict with SamPredictor
    batch = dataset[1]
    img, labels = batch
    try:
        batch_bboxes = labels["boxes"].unsqueeze(0)
    except AttributeError as exc:
        print(f"AttributeError: {exc}")
        batch_bboxes = [None] * len(img)
    try:
        batch_masks = labels["masks"].unsqueeze(0)
    except AttributeError as exc:
        print(f"AttributeError: {exc}")
        batch_masks = [None] * len(img)
    predictor = SamPredictor(model.network.model)
    masks = []
    images, labels = batch
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    image_original_shape = images.shape[-2:]
    for image, box, mask in zip(images, batch_bboxes, batch_masks):
        np_image = image.detach().numpy().transpose(1, 2, 0)
        np_mask = mask.detach().numpy() if mask is not None else None
        np_box = box.detach().numpy() if box is not None else None
        predictor.set_image(np_image)
        pred_masks, pred_qualities, pred_low_res_masks = predictor.predict(
            box=np_box,
            multimask_output=False,
        )
        masks.append(pred_masks)
    # Plot sample image with box, and image with mask
    print(f"SamPredictor masks: {len(masks)}")
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(img[0], cmap="Oranges")
    if box is not None:
        axs[0].add_patch(
            plt.Rectangle(
                (batch_bboxes[0][0][0], batch_bboxes[0][0][1]),
                batch_bboxes[0][0][2] - batch_bboxes[0][0][0],
                batch_bboxes[0][0][3] - batch_bboxes[0][0][1],
                edgecolor="r",
                facecolor="none",
            )
        )
    if mask is not None:
        axs[1].imshow(batch_masks[0][0], cmap="Oranges")
    else:
        # Plot empty
        axs[1].imshow(np.zeros(img[0].shape))
    axs[2].imshow(
        label2rgb(label=masks[0][0], image=img.numpy().transpose(1, 2, 0), bg_label=0)
    )
    plt.show()

# %%
if __name__ == "__main__":
    # Run a validation epoch and collect metrics
    metrics = trainer.validate(model, val_dataloader)
    print(f"Validation metrics: {metrics}")

# %%
