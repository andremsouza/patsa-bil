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

from imagelog_ai.features.methodologies.detr.datasets.dataset import DETRDataset
from imagelog_ai.features.methodologies.detr.modules.lit_module import LitDETR
from imagelog_ai.features.methodologies.detr.neural_networks.detr import build
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


# %% [markdown]
# # Data Loading and Preprocessing

# %% [markdown]
# ## DETR Dataset


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
    return x


project_name = "DETR"
project_settings = json_load(f"experiment_configs/{project_name}.json")

# test dataset
dataset = DETRDataset(
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
train_dataset.dataframe_detr = dataset.dataframe_detr.copy(deep=True)
train_dataset.dataframe_detr = train_dataset.dataframe_detr.iloc[
    idxs_train
].reset_index(drop=True)
val_dataset = copy.deepcopy(dataset)
val_dataset.dataframe_detr = dataset.dataframe_detr.copy(deep=True)
val_dataset.dataframe_detr = val_dataset.dataframe_detr.iloc[idxs_test].reset_index(
    drop=True
)

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
# model = maskrcnn_resnet50_fpn_v2(weights=DETR_ResNet50_FPN_V2_Weights.DEFAULT)
# Main
# args: An argument parser object containing the following attributes:
#     - dataset_file (str): The name of the dataset file.
#     - num_classes (int, optional): The number of classes. If not provided,
#         it will be inferred based on the dataset.
#     - device (str): The device to run the model on (e.g., 'cuda' or 'cpu').
#     - num_queries (int): The number of queries.
#     - aux_loss (bool): Whether to use auxiliary loss.
#     - masks (bool): Whether to include mask head for segmentation.
#     - frozen_weights (str, optional): Path to frozen weights.
#     - bbox_loss_coef (float): Coefficient for bounding box loss.
#     - giou_loss_coef (float): Coefficient for GIoU loss.
#     - mask_loss_coef (float): Coefficient for mask loss.
#     - dice_loss_coef (float): Coefficient for dice loss.
#     - dec_layers (int): Number of decoder layers.
#     - eos_coef (float): End-of-sequence coefficient.
# Backbone
# args: An object containing the following attributes:
#     - lr_backbone (float): Learning rate for the backbone.
#         If greater than 0, the backbone will be trained.
#     - masks (bool): If True, return intermediate layers.
#     - backbone (str): The type of backbone to use.
#     - dilation (bool): If True, apply dilation in the backbone.
# Transformer
# args: An object containing the following attributes:
#     hidden_dim (int): The dimension of the model.
#     dropout (float): The dropout rate.
#     nheads (int): The number of attention heads.
#     dim_feedforward (int): The dimension of the feedforward network.
#     enc_layers (int): The number of encoder layers.
#     dec_layers (int): The number of decoder layers.
#     pre_norm (bool): Whether to apply normalization before the attention and feedforward
#         layers.
# Matcher
# args: An object containing the following attributes:
#     - set_cost_class (float): The cost associated with class prediction.
#     - set_cost_bbox (float): The cost associated with bounding box prediction.
#     - set_cost_giou (float): The cost associated with
#         Generalized Intersection over Union (GIoU) prediction.
# Position Encoding
# args (Namespace): A namespace object containing the following attributes:
#     - hidden_dim (int): The dimension of the hidden layer.
#     - position_embedding (str): The type of position embedding to use.
#         It can be "v2" or "sine" for sine-based position embedding,
#         or "v3" or "learned" for learned position embedding.
args = project_settings["args"].copy()
args["batch_size"] = project_settings["batch_size"]
args["device"] = "cuda" if torch.cuda.is_available() else "cpu"
# Convert args to object with attributes
args = type("Args", (object,), args)()

# %%
model, criterion, postprocessors = build(args)
print(model)

# %%
# Zero-shot forward pass (test)
model.eval()
image, labels = dataset[0]
image = image.unsqueeze(0)
output = model(image)

# %%
# Plot output
ZERO_SHOT_THRESHOLD = 0.1
model_output = output
print(model_output.keys())
print(model_output["pred_boxes"].shape)
# print(model_output["aux_outputs"])
print(model_output["pred_masks"].shape)
# If masks shape are different than image shape, interpolate
if model_output["pred_masks"].shape[-2:] != image.shape[-2:]:
    model_output["pred_masks"] = F.interpolate(
        model_output["pred_masks"], size=image.shape[-2:]
    )
# if number of mask dimensions is less than 5, add channels dimension
if model_output["pred_masks"].ndim < 5:
    model_output["pred_masks"] = model_output["pred_masks"][:, :, None, :, :]
print(model_output["pred_masks"].shape)
print(model_output["pred_logits"].shape)
# Get masks
masks = model_output["pred_masks"]
# Filter output based on scores
# Filter the output based on the confidence ZERO_SHOT_THRESHOLD
scores_mask = model_output["pred_logits"][..., 0] > ZERO_SHOT_THRESHOLD
# Scale and stack the predicted segmentation masks
pred_masks = F.interpolate(
    model_output["pred_masks"][scores_mask], size=image.shape[-2:]
)
pred_masks = torch.concat(
    [
        Mask(torch.where(mask >= ZERO_SHOT_THRESHOLD, 1, 0), dtype=torch.bool)
        for mask in pred_masks
    ]
)
# Plot
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))
# axes[0].imshow(image[0, 0, :, :], cmap="Oranges")
# axes[1].imshow(pred_masks[0], cmap="Oranges")
# plt.plot()
# plt.close()

# %%
# Release resources from model
del model
del criterion
del postprocessors
torch.cuda.empty_cache()

# %% [markdown]
# ## Lightning Module Instantiation and Forward Test

# %%
# Instantiate Lightning module
lit_model = LitDETR(
    **project_settings["args"],
    batch_size=project_settings["batch_size"],
)
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

# %%
# Release resources from model
del lit_model
torch.cuda.empty_cache()

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
        pl.loggers.CSVLogger("logs", name="detr-finetuning"),
        pl.loggers.MLFlowLogger(
            experiment_name="detr-finetuning", tracking_uri="http://localhost:5000"
        ),
    ],
    max_epochs=100,
)

# %%
# Instantiate Lightning module
checkpoint_path = (
    "11/cff42f8e2e8c464bbafc9474a46765e2/checkpoints/epoch=33-step=442.ckpt"
)
try:
    lit_model = LitDETR.load_from_checkpoint(checkpoint_path)
except Exception as e:
    print(e)
    lit_model = LitDETR(
        **project_settings["args"],
        batch_size=project_settings["batch_size"],
        early_stopping_patience=16,
        lr_scheduler_patience=8,
        learning_rate=1e-4,
        weight_decay=1e-4,
    )
    # Load pretrained checkpoint
    if project_settings["checkpoint_path"]:
        if project_settings["checkpoint_path"].startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(
                project_settings["checkpoint_path"], map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(
                project_settings["checkpoint_path"], map_location="cpu"
            )
        lit_model.model.load_state_dict(checkpoint["model"], strict=False)
# Metrics are logged with MLFlow logger
mlflow_logger = pl.loggers.MLFlowLogger(
    experiment_name="DETR-IMLOGS",
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
metrics = trainer.validate(lit_model, val_dataloader)
print(f"Validation metrics: {metrics}")

# %%
# Free memory cache
torch.cuda.empty_cache()
# Train model
trainer.fit(
    lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
)

# %%
