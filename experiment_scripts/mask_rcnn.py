"""This module contains the Mask R-CNN model training script."""

# %% [markdown]
# # Constants and Imports

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
import copy
import os

import dotenv
import lightning.pytorch as pl
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
import torchvision.transforms.v2 as T
from torchvision.tv_tensors import Mask
import tqdm

from imagelog_ai.features.methodologies.mask_rcnn.datasets.dataset import (
    MaskRCNNDataset,
)
from imagelog_ai.features.methodologies.mask_rcnn.modules.lit_module import (
    LitMaskRCNN,
)
from imagelog_ai.utils.io_functions import json_load


# %%
# Load environment variables
dotenv.load_dotenv(".env", override=True)

# %%
np.random.seed(0)
torch.manual_seed(0)

OUTPUT_DIR: str = "test_figures"
OVERRIDE: bool = False

pil_transform = T.ToPILImage()
tensor_transform = T.ToImage()


def tensor_to_numpy_transform(x: torch.Tensor) -> np.ndarray:
    return x.numpy().transpose((1, 2, 0))


def collate_fn(batch):
    x = torch.stack([item[0] for item in batch])
    y = [item[1] for item in batch]
    return x, y


# %% [markdown]
# # Data Loading and Preprocessing

# %% [markdown]
# ## Mask R-CNN Dataset


# %%
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
    # Apply torchvision transform
    x = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT.transforms()(x)
    return x


def target_transform_func(x):
    """
    Applies transformations to the target data.

    Args:
        x (dict): A dictionary containing the target data.

    Returns:
        dict: The transformed target data.
    """
    # Binarize labels
    x["labels"] = torch.where(x["labels"] > 0, 1, 0)
    return x


project_name = "MaskRCNN"
project_settings = json_load(f"experiment_configs/{project_name}.json")

# test dataset
dataset = MaskRCNNDataset(
    project_name,
    project_settings["preprocess_name"],
    project_settings["list_datasource_names"],
    project_settings["class_list"],
    project_settings["others_class_list"],
    project_settings["background_class"],
    transform=transform_func,
    target_transform=target_transform_func,
    target_boxes=True,
    target_labels=True,
    target_masks=True,
    boxes_location="masks",
    masks_location="labels",
)
# Filter dataset rows with "component" == 0
dataset.dataframe = dataset.dataframe[dataset.dataframe["component"] != 0].reset_index(
    drop=True
)
print(dataset.dataframe)

# %%
# Separate dataset into train and val (80% train, 20% val)
idxs_train, idxs_test = train_test_split(
    np.arange(len(dataset)), test_size=0.2, random_state=0
)
train_dataset = copy.deepcopy(dataset)
train_dataset.dataframe_maskrcnn = dataset.dataframe_maskrcnn.copy(deep=True)
train_dataset.dataframe_maskrcnn = train_dataset.dataframe_maskrcnn.iloc[
    idxs_train
].reset_index(drop=True)
val_dataset = copy.deepcopy(dataset)
val_dataset.dataframe_maskrcnn = dataset.dataframe_maskrcnn.copy(deep=True)
val_dataset.dataframe_maskrcnn = val_dataset.dataframe_maskrcnn.iloc[
    idxs_test
].reset_index(drop=True)

# %%
# Iterate on all dataset samples and check if the labels are valid
for idx in tqdm.trange(len(dataset)):
    image, labels = dataset[idx]
    # Check if the boxes are valid
    if labels["boxes"].shape[0] == 0:
        print(f"Image {idx} has no boxes.")
    else:
        # Check if the boxes are valid
        for box in labels["boxes"]:
            if box[2] <= box[0] or box[3] <= box[1]:
                print(f"Image {idx} has invalid boxes.")
    # Check if the masks are valid
    if labels["masks"].shape[0] == 0:
        print(f"Image {idx} has no masks.")
    else:
        # Check if the masks are valid
        if labels["masks"].shape[1] == 0:
            print(f"Image {idx} has invalid masks.")

# %%
dataset_images = dataset.dataframe["image_file"].unique()
image_path = dataset_images[0]
print(image_path)
# Get all indexes from the dataset with the same image_path
indexes = dataset.dataframe[dataset.dataframe["image_file"] == image_path].index
print(indexes)

# %% [markdown]
# # Model Instantiation

# %%
# Load pre-trained model
model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
print(model)

# %%
# Zero-shot forward pass (test)
model.eval()
image, labels = dataset[0]
image = image.unsqueeze(0)
output = model(image)

# %%
# Plot output
ZERO_SHOT_THRESHOLD = 0.3
model_output = output[0]
print(model_output.keys())
print(model_output["boxes"].shape)
print(model_output["labels"].shape)
print(model_output["masks"].shape)
print(model_output["scores"].shape)
# Get masks
masks = model_output["masks"]
# Filter output based on scores
# Filter the output based on the confidence ZERO_SHOT_THRESHOLD
scores_mask = model_output["scores"] > ZERO_SHOT_THRESHOLD
# Scale and stack the predicted segmentation masks
pred_masks = F.interpolate(model_output["masks"][scores_mask], size=image.shape[-2:])
pred_masks = torch.concat(
    [
        Mask(torch.where(mask >= ZERO_SHOT_THRESHOLD, 1, 0), dtype=torch.bool)
        for mask in pred_masks
    ]
)
# Plot
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(image[0, 0, :, :], cmap="Oranges")
# axes[1].imshow(pred_masks[0], cmap="Oranges")
# plt.plot()
# plt.close()

# %% [markdown]
# ## Lightining Module Instantiation and Forward Test

# %%
# Instantiate Lightning module
lit_model = LitMaskRCNN(num_classes=2, batch_size=project_settings["batch_size"])
print(lit_model)

# %%
# Zero-shot forward pass (train)
lit_model.train()
image, labels = dataset[0]
image = image.unsqueeze(0)
labels = [labels]
output = lit_model(image, labels)
print(output)

# %%
# Zero-shot forward pass (test)
lit_model.eval()
image, labels = dataset[0]
image = image.unsqueeze(0)
labels = [labels]
output = lit_model(image)
print(output)


# %% [markdown]
# # Training (with Lightning and MLFlow)

# %%
# Instantiate data loaders
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=project_settings["batch_size"],
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=project_settings["batch_size"],
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
)

# %%
# iterate on dataloaders (test)
for batch in train_dataloader:
    x, y = batch
    print(x.shape, len(y))
    break

for batch in val_dataloader:
    x, y = batch
    print(x.shape, len(y))
    break

# %%
# Instantiate Lightning module
lit_model = LitMaskRCNN(
    num_classes=2,
    batch_size=project_settings["batch_size"],
    early_stopping_patience=4,
    lr_scheduler_patience=2,
    learning_rate=1e-5,
)
# Metrics are logged with MLFlow logger
mlflow_logger = pl.loggers.MLFlowLogger(
    experiment_name="MaskRCNN-IMLOGS",
    tracking_uri="http://localhost:5000",
    log_model=True,
)
# Instantiate Lightning trainer (with callbacks and logger)
trainer = pl.Trainer(
    logger=mlflow_logger,
    max_epochs=100,
    log_every_n_steps=min(50, len(train_dataloader)),
)

# %%
# Train model
trainer.fit(
    lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
)

# %%
