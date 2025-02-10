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
    package_path = os.getenv("PACKAGEPATH", "")
    print(f"PACKAGEPATH: {package_path}")
    sys.path.append(package_path)

# %%
import copy

import lightning.pytorch as pl
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as T
from torchvision.tv_tensors import Mask
import tqdm

from imagelog_ai.features.methodologies.dinov2.datasets.dataset import DINODataset
from imagelog_ai.features.methodologies.dinov2.modules.lit_module import LitDINOv2
from imagelog_ai.features.methodologies.dinov2.neural_networks.dinov2 import (
    Dinov2ForSemanticSegmentation,
)
from imagelog_ai.utils.io_functions import json_load

# %%
# Load environment variables
dotenv.load_dotenv(".env", override=True)

# %%
torch.set_float32_matmul_precision("high")
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


ID2LABEL = {0: "background", 1: "foreground"}


# %% [markdown]
# # Data Loading and Preprocessing

# %% [markdown]
# ## DINO Dataset


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
    # Convert from RGB to grayscale
    grayscale_transform = T.Grayscale(num_output_channels=3)
    x = grayscale_transform(x)
    # Invert intensities
    x = x.max() - x
    # Convert to float
    x = x.float()
    # Scale to [0, 1]
    x = x / x.max()
    # Resize to (448, 448)
    resize_transform = T.Resize((448, 448))
    x = resize_transform(x)
    return x


def target_transform_func(x):
    """
    Applies transformations to the target data.

    Args:
        x (dict): A dictionary containing the target data.

    Returns:
        dict: The transformed target data.
    """
    # Convert bboxes to float
    x["boxes"] = x["boxes"].float()
    # Binarize labels
    x["labels"] = torch.where(x["labels"] > 0, 1, 0)
    # Convert list of masks to a single 2D mask tensor
    gt_mask = torch.zeros_like(x["masks"][0]).long()
    for mask, label in zip(x["masks"], x["labels"]):
        gt_mask = torch.where(mask > 0, label, gt_mask)
    gt_mask = gt_mask.unsqueeze(0)
    # Resize gt_mask to (448, 448)
    resize_transform = T.Resize((448, 448))
    gt_mask = resize_transform(gt_mask)
    x["gt_mask"] = gt_mask
    return x


project_name = "DINOv2"
project_settings = json_load(f"experiment_configs/{project_name}.json")

# test dataset
dataset = DINODataset(
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
train_dataset.dataframe_dino = dataset.dataframe_dino.copy(deep=True)
train_dataset.dataframe_dino = train_dataset.dataframe_dino.iloc[
    idxs_train
].reset_index(drop=True)
val_dataset = copy.deepcopy(dataset)
val_dataset.dataframe_dino = dataset.dataframe_dino.copy(deep=True)
val_dataset.dataframe_dino = val_dataset.dataframe_dino.iloc[idxs_test].reset_index(
    drop=True
)

# %%
# Iterate on all dataset samples and check if the labels are valid
for idx in tqdm.trange(len(dataset)):
    image, labels, _, _ = dataset[idx]
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
# # Load pre-trained model
# model = Dinov2ForSemanticSegmentation.from_pretrained(
#     "facebook/dinov2-base", id2label=ID2LABEL, num_labels=len(ID2LABEL)
# )
# print(model)

# # %%
# # Zero-shot forward pass (test)
# model.eval()
# image, labels, _, _ = dataset[0]
# image = image.unsqueeze(0)
# output = model(image, labels=labels)

# # %%
# # Plot output
# ZERO_SHOT_THRESHOLD = 0.5
# upsampled_logits = torch.nn.functional.interpolate(
#     output.logits, size=image.size()[-2:], mode="bilinear", align_corners=False
# )
# print("upsampled_logits.shape:", upsampled_logits.shape)
# predicted_map = upsampled_logits.argmax(dim=1)
# print("predicted_map.shape:", predicted_map.shape)
# # Plot
# # import matplotlib.pyplot as plt

# # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# # axes[0].imshow(image[0, 0, :, :], cmap="Oranges")
# # axes[1].imshow(predicted_map[0], cmap="Oranges")
# # plt.plot()
# # # plt.close()

# # %%
# # Release resources from model
# del model
# torch.cuda.empty_cache()

# %% [markdown]
# ## Lightning Module Instantiation and Forward Test

# # %%
# # Instantiate Lightning module
# lit_model = LitDINOv2(
#     "facebook/dinov2-base",
#     id2label=ID2LABEL,
#     num_labels=len(ID2LABEL),
#     batch_size=project_settings["batch_size"],
#     early_stopping_patience=16,
#     lr_scheduler_patience=8,
#     learning_rate=1e-4,
#     weight_decay=1e-4,
# )
# print(lit_model)

# # %%
# # Zero-shot forward pass (train)
# lit_model.train()
# image, labels, _, _ = dataset[0]
# image = image.unsqueeze(0)
# labels = [labels]
# output = lit_model(image, labels)
# print(output)

# # %%
# # Zero-shot forward pass (test)
# lit_model.eval()
# image, labels, _, _ = dataset[0]
# image = image.unsqueeze(0)
# labels = [labels]
# output = lit_model(image)
# print(output)

# # %%
# # Release resources from model
# del lit_model
# torch.cuda.empty_cache()

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

trainer = pl.Trainer(
    logger=[
        pl.loggers.CSVLogger("logs", name="dinov2-finetuning"),
        pl.loggers.MLFlowLogger(
            experiment_name="dinov2-finetuning", tracking_uri="http://localhost:5000"
        ),
    ],
    max_epochs=100,
)

# %%
# Instantiate Lightning module
checkpoint_path = (
    "./12/99c3679a44a948e2acbd2a8ef736087c/checkpoints/epoch=28-step=377.ckpt"
)
# try:
#     lit_model = LitDINOv2.load_from_checkpoint(
#         checkpoint_path,
#         pretrained_model_name_or_path="facebook/dinov2-base",
#         id2label=ID2LABEL,
#         num_labels=len(ID2LABEL),
#         batch_size=project_settings["batch_size"],
#         early_stopping_patience=16,
#         lr_scheduler_patience=8,
#         learning_rate=1e-4,
#         weight_decay=1e-4,
#     )
# except Exception as e:
#     print(e)
lit_model = LitDINOv2(
    "facebook/dinov2-base",
    id2label=ID2LABEL,
    num_labels=len(ID2LABEL),
    batch_size=project_settings["batch_size"],
    early_stopping_patience=16,
    lr_scheduler_patience=8,
    learning_rate=1e-4,
    weight_decay=1e-4,
)
# Metrics are logged with MLFlow logger
mlflow_logger = pl.loggers.MLFlowLogger(
    experiment_name="DINOv2-IMLOGS",
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
# metrics = trainer.validate(lit_model, val_dataloader)
# print(f"Validation metrics: {metrics}")

# %%
# Free memory cache
torch.cuda.empty_cache()
# Train model
trainer.fit(
    lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
)

# %%
