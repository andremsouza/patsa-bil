"""This module contains the code for the SLIC+DBSCAN segmentation pipeline."""

# %% [markdown]
# ## Constants and Imports

# %%
# Change directory to the root of the project
if __name__ == "__main__":
    import os
    import sys

    import dotenv

    os.chdir(os.getcwd().split("notebooks")[0])
    print(f"cwd: {os.getcwd()}")
    dotenv.load_dotenv()
    sys.path.append(os.getenv("PACKAGEPATH", "/home/user/patsa-bil"))

# %%
import logging
import os
from urllib.parse import quote_plus

import cv2
import dotenv
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import radiomics
import seaborn as sns
from skimage.color import label2rgb
from skimage.segmentation import mark_boundaries
from sklearn.metrics import silhouette_score
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision.transforms.v2 as T

from imagelog_ai.features.methodologies.sam.datasets.dataset import SamDataset
from imagelog_ai.features.methodologies.sam.utils.iou import calc_iou
from imagelog_ai.features.methodologies.slicdbscan.models.slicdbscan import (
    SLICDBSCANSegmentation,
)
from imagelog_ai.utils.io_functions import json_load

# %%
# Set logging to file
os.makedirs("./notebooks/logs", exist_ok=True)
log_file = "./notebooks/logs/radiomics_slicdbscan_5r_map.log"
handler = logging.FileHandler(filename=log_file, mode="w")
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
radiomics.logger.addHandler(handler)
# Set logging verbosity
logging.getLogger("radiomics.glcm").setLevel(logging.ERROR)

# %%
# Load environment variables
dotenv.load_dotenv("docker/.env", override=True)
# Get PostgreSQL environment variables
POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "")
POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "")
POSTGRES_DB: str = os.getenv("POSTGRES_DB", "")
POSTGRES_USER_OPTUNA: str = os.getenv("POSTGRES_USER_OPTUNA", "")
POSTGRES_PASSWORD_OPTUNA: str = os.getenv("POSTGRES_PASSWORD_OPTUNA", "")

POSTGRES_URL = (
    f"postgresql+psycopg2://{POSTGRES_USER_OPTUNA}:{quote_plus(POSTGRES_PASSWORD_OPTUNA)}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)
# Print URL (without password)
print(POSTGRES_URL.replace(f":{POSTGRES_PASSWORD_OPTUNA}@", ":***@"))

# Create Optuna storage
storage = optuna.storages.RDBStorage(url=POSTGRES_URL)

# %%
# try:
#     while True:
#         optuna.delete_study(study_name="SLICDBSCANSegmentation_5r_map", storage=storage)
#         print("Deleted study")
# except KeyError as exc:
#     print(exc)

# %%
np.random.seed(0)
torch.manual_seed(0)

OUTPUT_DIR: str = "test_figures"
OVERRIDE: bool = False

pil_transform = T.ToPILImage()
tensor_transform = T.ToImage()


def tensor_to_numpy_transform(x: torch.Tensor) -> np.ndarray:
    """Converts a PyTorch tensor to a NumPy array."""
    return x.numpy().transpose((1, 2, 0))


# %% [markdown]
# ## Data Loading and Preprocessing

# %% [markdown]
# ### Dataset


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


project_name = "SLICDBSCAN"
project_settings = json_load(f"experiment_configs/{project_name}.json")
# test dataset
# Using SamDataset from SAM experiments
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
# Filter dataset rows with "component" == 0
dataset.dataframe = dataset.dataframe[dataset.dataframe["component"] != 0].reset_index(
    drop=True
)
print(dataset.dataframe)

# %%
i = 0
image, labels = dataset[i]
masks = labels["masks"]
image_row = dataset.dataframe.iloc[i]
print(image_row)
print(image.shape, labels)
np_image = tensor_to_numpy_transform(image)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(np_image[:, :, 0], cmap="Oranges")
axes[1].imshow(masks[0], cmap="Oranges")

# %%
dataset_images = dataset.dataframe["image_file"].unique()
image_path = dataset_images[0]
print(image_path)
# Get all indexes from the dataset with the same image_path
indexes = dataset.dataframe[dataset.dataframe["image_file"] == image_path].index
print(indexes)
# Load all indexes and get masks
masks = []
for index in indexes:
    image, labels = dataset[index]
    masks.append(labels["masks"])
# Union masks
gt_mask = torch.zeros_like(masks[0])
for mask in masks:
    gt_mask = torch.logical_or(gt_mask, mask)
# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 10))
axes[0].imshow(image[0, :, :], cmap="Oranges")
axes[1].imshow(gt_mask[0], cmap="Oranges")
plt.show()
# Close figure
plt.close(fig)

# %% [markdown]
# ## SLICDBSCANSegmentation Optimization

# %% [markdown]
# ### Radiomics

# %% [markdown]
# #### Optuna (obj function)

# %%
# Create objective function


def slicdbscan_radiomics_objective(trial: optuna.Trial) -> float:
    """Objective function for the SLICDBSCANSegmentation optimization."""
    # Define the hyperparameters
    # SLIC
    slic_algorithm = trial.suggest_categorical(
        "slic_algorithm", [cv2.ximgproc.SLIC, cv2.ximgproc.SLICO, cv2.ximgproc.MSLIC]
    )
    region_size = trial.suggest_int("region_size", 5, 12)
    ruler = trial.suggest_float("ruler", 8.0, 12.0)
    slic_iterations = trial.suggest_categorical("slic_iterations", [100])
    gaussian_blur = trial.suggest_categorical("gaussian_blur", [None])
    # gaussian_blur = trial.suggest_int("gaussian_blur", -1, 9, step=2)
    sigma = trial.suggest_categorical("sigma", [0.0])
    # sigma = trial.suggest_float("sigma", 0.0, 1.0)
    enforce_connectivity = trial.suggest_categorical(
        "enforce_connectivity", [False, True]
    )
    min_element_size = trial.suggest_int("min_element_size", 0, 25)
    # DBSCAN
    min_label_size = trial.suggest_int("min_label_size", 5, 5)
    use_radiomics = trial.suggest_categorical("use_radiomics", [True])
    dbscan_features = trial.suggest_categorical("dbscan_features", [None])
    use_pca = trial.suggest_categorical("use_pca", [True])
    eps = trial.suggest_float("eps", 1e-6, 1e2, log=True)
    min_samples = trial.suggest_int("min_samples", 1, 10)
    metric = trial.suggest_categorical("metric", ["euclidean"])
    metric_params = trial.suggest_categorical("metric_params", [None])
    algorithm = trial.suggest_categorical("algorithm", ["auto"])
    leaf_size = trial.suggest_int("leaf_size", 10, 100)
    p = trial.suggest_float("p", 1.0, 2.0)
    # Train and evaluate
    superpixel_nums: list[int] = []
    map_scores: list[float] = []
    map_metric = MeanAveragePrecision(
        iou_type="segm",
        iou_thresholds=np.arange(start=0.05, stop=1.00, step=0.05).tolist(),
        extended_summary=True,
    )
    # Get unique image files
    dataset_images = dataset.dataframe["image_file"].unique()
    # Iterate through dataset
    for idx, image_path in enumerate(dataset_images):
        indexes = dataset.dataframe[dataset.dataframe["image_file"] == image_path].index
        # Get filepath to image
        img_file = dataset.list_images_paths[idx]
        # Load all indexes and get masks
        masks = []
        for index in indexes:
            image, labels = dataset[index]
            masks.append(labels["masks"])
        # Union masks
        gt_mask = torch.zeros_like(masks[0])
        for mask in masks:
            gt_mask = torch.logical_or(gt_mask, mask)
        # Apply the segmentation
        slicdbscan = SLICDBSCANSegmentation(
            slic_algorithm=slic_algorithm,
            region_size=region_size,
            ruler=ruler,
            slic_iterations=slic_iterations,
            gaussian_blur=gaussian_blur,
            sigma=sigma,
            enforce_connectivity=enforce_connectivity,
            min_element_size=min_element_size,
            min_label_size=min_label_size,
            use_radiomics=use_radiomics,
            dbscan_features=dbscan_features,
            use_pca=use_pca,
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            metric_params=metric_params,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            n_jobs=-1,
        )
        try:
            np_segmentation = slicdbscan(image[0].unsqueeze(0))
            bin_mask = np_segmentation > 0
            preds = [
                {
                    "masks": bin_mask,
                    "scores": torch.tensor([1.0]),
                    "labels": torch.tensor([1]),
                }
            ]
            target = [
                {
                    "masks": gt_mask,
                    "labels": torch.tensor([1]),
                }
            ]
            map_score = map_metric(preds=preds, target=target)["map"].float()
        except Exception as exc:
            raise optuna.TrialPruned(
                f"Trial pruned at iteration {idx} ({img_file}). Reason: {exc}"
            )
        if np_segmentation.max() <= 4 or np_segmentation.max() >= 50:
            raise optuna.TrialPruned(
                f"Trial pruned at iteration {idx} ({img_file})."
                f"Reason: Superpixels == {np_segmentation.max()} <= 4"
            )
        superpixel_nums.append(np_segmentation.max())
        map_scores.append(map_score)
        trial.report(float(np.mean(map_scores)), idx)
        if trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial pruned at iteration {idx} ({img_file})."
                f"Reason: Mean superpixels == {np.mean(superpixel_nums)}"
            )
    # Return evaluation metric to optimize
    return float(np.mean(map_scores))


# %% [markdown]
# #### Optuna (study)

# %%
# Create study for SLICDBSCANSegmentation
study_name = "SLICDBSCANSegmentation_5r_map"
study = optuna.create_study(
    storage=storage,
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=5, interval_steps=1
    ),
    study_name=study_name,
    direction="maximize",
    load_if_exists=True,
)
try:
    print(study.best_params)
    print(study.best_trials)
    print(study.best_value)
except Exception as exc:
    print(exc)


# %% [markdown]
# #### Optuna (optimize)

# %%
# Resume unfinished trials
for trial in study.trials:
    if trial.state in [
        optuna.trial.TrialState.RUNNING,
        optuna.trial.TrialState.WAITING,
    ]:
        study.enqueue_trial(trial.params)

# %%
# Optimize study
study.optimize(slicdbscan_radiomics_objective, n_trials=1000, timeout=None, n_jobs=20)

# %% [markdown]
# #### Plot 2

# %%
# Set colors
colors = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "gray",
    "black",
    "white",
    "cyan",
    "magenta",
    "teal",
    "navy",
    "olive",
    "maroon",
    "lime",
    "aqua",
    "silver",
]
# Test the best parameters
slicdbscan_best_params = study.best_params
print(slicdbscan_best_params)
# Iterate through dataset
subfolder: str = "slicdbscan_5r_map_pca"
metrics: dict = {}
metrics_dfs = {}
datasets: dict[str, SamDataset] = {}
datasets["annotated"] = dataset
for dset_key, dset in datasets.items():
    # Collect metrics
    metrics[dset_key] = []
    # If override is false and metrics file already exists, skip
    if not OVERRIDE and os.path.exists(
        os.path.join(OUTPUT_DIR, subfolder, f"{dset_key}_metrics.csv")
    ):
        print(f"Skipping {dset_key} (metrics already exist)")
        # load metrics
        metrics_df = pd.read_csv(
            os.path.join(OUTPUT_DIR, subfolder, f"{dset_key}_metrics.csv")
        )
        metrics_dfs[dset_key] = metrics_df
        continue
    for idx, (image, target) in enumerate(dset):
        # Get original filename
        file_idx = dset.list_images_paths[idx].split("/")[-1].split(".")[0]
        # If override is false and file already exists, skip
        if not OVERRIDE and os.path.exists(
            os.path.join(OUTPUT_DIR, subfolder, dset_key, f"{file_idx}.png")
        ):
            print(f"Skipping {dset_key}:{file_idx}")
            continue
        np_image = tensor_to_numpy_transform(image)[:, :, :3]
        # Apply SLIC segmentation

        try:
            slicdbscan = SLICDBSCANSegmentation(
                slic_algorithm=slicdbscan_best_params["slic_algorithm"],
                region_size=slicdbscan_best_params["region_size"],
                ruler=slicdbscan_best_params["ruler"],
                slic_iterations=slicdbscan_best_params["slic_iterations"],
                gaussian_blur=slicdbscan_best_params["gaussian_blur"],
                sigma=slicdbscan_best_params["sigma"],
                enforce_connectivity=slicdbscan_best_params["enforce_connectivity"],
                min_element_size=slicdbscan_best_params["min_element_size"],
                min_label_size=slicdbscan_best_params["min_label_size"],
                use_radiomics=slicdbscan_best_params["use_radiomics"],
                dbscan_features=slicdbscan_best_params["dbscan_features"],
                use_pca=slicdbscan_best_params["use_pca"],
                eps=slicdbscan_best_params["eps"],
                min_samples=slicdbscan_best_params["min_samples"],
                n_jobs=-1,
                return_mask=True,
            )
            slic_superpixels = slicdbscan(image)
        except Exception as exc:
            print(f"Exception: {exc}")
            continue
        num_superpixels = slic_superpixels.max() + 1
        np_superpixels = np.array(slic_superpixels)
        boundaries = mark_boundaries(np_image[:, :, 2], np_superpixels[2])
        print(f"Number of superpixels: {num_superpixels}")
        # Plot image per channel on first row and superpixel contour mask on second row
        # with bounding boxes on both rows
        fig, axes = plt.subplots(1, 3, figsize=(15, 10))
        # Image
        axes[0].imshow(image[0], cmap="Oranges")
        axes[0].axis("off")
        axes[1].imshow(image[2], cmap="Oranges")
        axes[1].axis("off")
        # Image with label2rgb
        axes[2].imshow(image[2], cmap="Oranges")
        # Use labels2rgb to color the superpixels
        axes[2].imshow(
            label2rgb(
                label=np.array(slic_superpixels[2]),
                image=np_image[:, :, 2],
                colors=colors,
                bg_label=0,
            ),
            alpha=1.0,
        )
        axes[2].contour(slic_superpixels[2], colors="gray", linewidths=0.5)
        axes[2].axis("off")
        # Save to .png
        # Create dirs if not exist
        os.makedirs(os.path.join(OUTPUT_DIR, subfolder, dset_key), exist_ok=True)
        # save figure
        # If Override is false, do not save if file already exists
        if (
            not os.path.exists(
                os.path.join(OUTPUT_DIR, subfolder, dset_key, f"{file_idx}.png")
            )
            or OVERRIDE
        ):
            fig.savefig(
                os.path.join(OUTPUT_DIR, subfolder, dset_key, f"{file_idx}.png"),
                bbox_inches="tight",
                dpi=300,
            )
        # Save feature set as .CSV
        feature_set = slicdbscan.dbscan.last_feature_set
        feature_set.to_csv(
            os.path.join(OUTPUT_DIR, subfolder, dset_key, f"{file_idx}.csv"),
            index=True,
        )
        # Collect metrics
        metrics_set = slicdbscan.dbscan.last_metrics
        metrics_set["num_superpixels"] = num_superpixels
        metrics_set["image_shape"] = image.shape
        metrics_set["image_path"] = dset.list_images_paths[idx]
        metrics[dset_key].append(metrics_set)
        # Close figure
        plt.close(fig)
    # Create DataFrame with metrics
    metrics_df = pd.DataFrame(metrics[dset_key])
    metrics_dfs[dset_key] = metrics_df
    # Save metrics to .CSV
    if (
        not os.path.exists(
            os.path.join(OUTPUT_DIR, subfolder, f"{dset_key}_metrics.csv")
        )
        or OVERRIDE
    ):
        print(f"Saving {dset_key} metrics")
        metrics_df.to_csv(
            os.path.join(OUTPUT_DIR, subfolder, f"{dset_key}_metrics.csv"), index=False
        )

# %%
for key, df in metrics_dfs.items():
    df["dataset"] = key

# %%
# Concatenate all dfs
metrics_df = pd.concat(list(metrics_dfs.values()), ignore_index=True)
print(metrics_df)

# %%
# Melting the DataFrame
metrics_melted = metrics_df.melt(
    id_vars=["dataset", "num_superpixels", "image_shape", "image_path"],
    var_name="feature",
    value_name="value",
)
print(metrics_melted)

# %% [markdown]
# ### Silhouette Score

# %%
# Describe overall silhouette_score
metrics_df["silhouette_score"].describe()

# %%
metrics_df["silhouette_score"].describe()

# %%
# Describe overall silhouette_score_pca
metrics_df["silhouette_score_pca"].describe()

# %%
# Describe for each dataset
metrics_melted[metrics_melted["feature"] == "silhouette_score"].groupby("dataset")[
    "value"
].describe()

# %%
# Describe for each dataset
metrics_melted[metrics_melted["feature"] == "silhouette_score_pca"].groupby("dataset")[
    "value"
].describe()

# %%
# Plot silhouette_score_pca for each dataset
plt.figure(figsize=(15, 10))
sns.boxplot(
    x="dataset",
    y="value",
    data=metrics_melted[metrics_melted["feature"] == "silhouette_score_pca"],
)

# %% [markdown]
# ### mAP Score

# %%
slicdbscan_best_params = study.best_params
slicdbscan = SLICDBSCANSegmentation(
    slic_algorithm=slicdbscan_best_params["slic_algorithm"],
    region_size=slicdbscan_best_params["region_size"],
    ruler=slicdbscan_best_params["ruler"],
    slic_iterations=slicdbscan_best_params["slic_iterations"],
    gaussian_blur=slicdbscan_best_params["gaussian_blur"],
    sigma=slicdbscan_best_params["sigma"],
    enforce_connectivity=slicdbscan_best_params["enforce_connectivity"],
    min_element_size=slicdbscan_best_params["min_element_size"],
    min_label_size=slicdbscan_best_params["min_label_size"],
    use_radiomics=slicdbscan_best_params["use_radiomics"],
    dbscan_features=slicdbscan_best_params["dbscan_features"],
    use_pca=slicdbscan_best_params["use_pca"],
    eps=slicdbscan_best_params["eps"],
    min_samples=slicdbscan_best_params["min_samples"],
    n_jobs=-1,
    return_mask=True,
)
dataset_images = dataset.dataframe["image_file"].unique()
image_path = dataset_images[0]
print(image_path)
# Get all indexes from the dataset with the same image_path
indexes = dataset.dataframe[dataset.dataframe["image_file"] == image_path].index
print(indexes)
# Load all indexes and get masks
masks = []
for index in indexes:
    image, labels = dataset[index]
    masks.append(labels["masks"])
# Union masks
gt_mask = torch.zeros_like(masks[0])
for mask in masks:
    gt_mask = torch.logical_or(gt_mask, mask)
# Get SLIC+DBSCAN mask
image, labels = dataset[indexes[0]]
slic_superpixels = slicdbscan(image[0].unsqueeze(0))
bin_mask = slic_superpixels > 0
# Get dbscan features
features = slicdbscan.dbscan.last_feature_set
# Drop all columns starting with "diagnostics_"
features = features.loc[:, ~features.columns.str.startswith("diagnostics_")]
cluster_features = features.groupby("dbscan_label").mean()
# Remove border cluster (cluster with lowest energy)
border_cluster = cluster_features["original_firstorder_Energy"].idxmin()
# Set border cluster to 0
slic_superpixels[slic_superpixels == border_cluster] = 0
bin_mask_noborder = slic_superpixels > 0
# Plot
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0][0].imshow(image[0, :, :], cmap="Oranges")
axes[0][1].imshow(gt_mask[0], cmap="Oranges")
axes[1][0].imshow(bin_mask[0], cmap="Oranges")
axes[1][1].imshow(bin_mask_noborder[0], cmap="Oranges")
plt.show()
# Close figure
plt.close(fig)

# %%
print(slicdbscan.dbscan.last_metrics)

# %% [markdown]
# #### Calculate mAP on all clusters

# %%
# Set colors
colors = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "gray",
    "black",
    "white",
    "cyan",
    "magenta",
    "teal",
    "navy",
    "olive",
    "maroon",
    "lime",
    "aqua",
    "silver",
]
# Test the best parameters
slicdbscan_best_params = study.best_params
slicdbscan = SLICDBSCANSegmentation(
    slic_algorithm=slicdbscan_best_params["slic_algorithm"],
    region_size=slicdbscan_best_params["region_size"],
    ruler=slicdbscan_best_params["ruler"],
    slic_iterations=slicdbscan_best_params["slic_iterations"],
    gaussian_blur=slicdbscan_best_params["gaussian_blur"],
    sigma=slicdbscan_best_params["sigma"],
    enforce_connectivity=slicdbscan_best_params["enforce_connectivity"],
    min_element_size=slicdbscan_best_params["min_element_size"],
    min_label_size=slicdbscan_best_params["min_label_size"],
    use_radiomics=slicdbscan_best_params["use_radiomics"],
    dbscan_features=slicdbscan_best_params["dbscan_features"],
    use_pca=slicdbscan_best_params["use_pca"],
    eps=slicdbscan_best_params["eps"],
    min_samples=slicdbscan_best_params["min_samples"],
    n_jobs=-1,
    return_mask=True,
)
print(slicdbscan_best_params)
# Iterate through dataset
subfolder = "slicdbscan_map"
metrics = []
metrics_df = None
dataset_images = dataset.dataframe["image_file"].unique()
ious = []
maps = []
silhouette_scores: list = []
silhouette_scores_pca: list = []
ns_clusters = []
map_metric = MeanAveragePrecision(
    iou_type="segm",
    iou_thresholds=np.arange(start=0.05, stop=1.00, step=0.05).tolist(),
    extended_summary=True,
)
for idx, image_path in enumerate(dataset_images):
    print(f"Processing image {idx + 1}/{len(dataset_images)}")
    print(f"Image path: {image_path}")
    indexes = dataset.dataframe[dataset.dataframe["image_file"] == image_path].index
    # Load all indexes and get masks
    masks = []
    for index in indexes:
        image, labels = dataset[index]
        masks.append(labels["masks"])
    # Union masks
    gt_mask = torch.zeros_like(masks[0])
    for mask in masks:
        gt_mask = torch.logical_or(gt_mask, mask)
    # Convert to torch uint8
    gt_mask = gt_mask.byte()
    print(f"GT Mask shape: {gt_mask.shape}")
    # Apply SLIC segmentation
    try:
        slic_superpixels = slicdbscan(image[0].unsqueeze(0))
    except Exception as exc:
        print(f"Exception: {exc}")
        continue
    # Calculate silhouette score
    feature_array_pca = slicdbscan.dbscan.last_feature_set_pca.drop(
        columns=["dbscan_label"]
    ).to_numpy()
    labels_pca = slicdbscan.dbscan.last_feature_set_pca["dbscan_label"].to_numpy()
    silhouette_pca = silhouette_score(feature_array_pca, labels_pca, metric="euclidean")
    # Binarize mask
    bin_mask = slic_superpixels > 0
    print(f"Binary Mask shape: {bin_mask.shape}")
    # Plot gt_mask and bin_mask
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(gt_mask[0], cmap="Oranges")
    axes[1].imshow(bin_mask[0], cmap="Oranges")
    # Print shapes
    # Calculate iou
    iou = calc_iou(bin_mask, gt_mask)
    preds = [
        {
            "masks": bin_mask,
            "scores": torch.tensor([1.0]),
            "labels": torch.tensor([1]),
        }
    ]
    target = [
        {
            "masks": gt_mask,
            "labels": torch.tensor([1]),
        }
    ]
    maps.append(map_metric(preds=preds, target=target))
    ious.append(iou)
    silhouette_scores_pca.append(silhouette_pca)
    ns_clusters.append(slic_superpixels.max())
    mean_iou = np.mean(ious)
    # Collect dbscan metrics
    feature_set = slicdbscan.dbscan.last_feature_set
    metrics_set = slicdbscan.dbscan.last_metrics
    metrics_set["num_superpixels"] = slic_superpixels.max()
    metrics_set["image_shape"] = image.shape
    metrics_set["image_path"] = image_path
    metrics_set["iou"] = iou
    metrics_set["silhouette_pca"] = silhouette_pca

    metrics.append(metrics_set)
    print(
        f"Silhouette Score (PCA): {silhouette_pca} (Mean: {np.mean(silhouette_scores_pca)})"
    )
    print(f"IoU: {iou} (Mean: {mean_iou})")
    print(f"# of clusters: {slic_superpixels.max()} (Mean: {np.mean(ns_clusters)})")
    plt.close(fig)
metrics_df = pd.DataFrame(metrics)

# %%
print(metrics_df)

# %%
print(f"Mean IoU: {np.mean(ious)}")
print(map_metric.compute())
print(f"Silhouette Score: {np.mean(silhouette_scores_pca)}")

# %%
print({k: float(i) for k, i in map_metric.compute()["ious"].items()})

# %% [markdown]
# #### Calculate mAP without border clusters

# %%
# Set colors
colors = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "gray",
    "black",
    "white",
    "cyan",
    "magenta",
    "teal",
    "navy",
    "olive",
    "maroon",
    "lime",
    "aqua",
    "silver",
]
# Test the best parameters
slicdbscan_best_params = study.best_params
slicdbscan = SLICDBSCANSegmentation(
    slic_algorithm=slicdbscan_best_params["slic_algorithm"],
    region_size=slicdbscan_best_params["region_size"],
    ruler=slicdbscan_best_params["ruler"],
    slic_iterations=slicdbscan_best_params["slic_iterations"],
    gaussian_blur=slicdbscan_best_params["gaussian_blur"],
    sigma=slicdbscan_best_params["sigma"],
    enforce_connectivity=slicdbscan_best_params["enforce_connectivity"],
    min_element_size=slicdbscan_best_params["min_element_size"],
    min_label_size=slicdbscan_best_params["min_label_size"],
    use_radiomics=slicdbscan_best_params["use_radiomics"],
    dbscan_features=slicdbscan_best_params["dbscan_features"],
    use_pca=slicdbscan_best_params["use_pca"],
    eps=slicdbscan_best_params["eps"],
    min_samples=slicdbscan_best_params["min_samples"],
    n_jobs=-1,
    return_mask=True,
)
print(slicdbscan_best_params)
# Iterate through dataset
subfolder = "slicdbscan_5r_map_noborder"
metrics_noborder = []
metrics_df_noborder = None
dataset_images = dataset.dataframe["image_file"].unique()
ious = []
maps = []
silhouette_scores = []
silhouette_scores_pca = []
ns_clusters = []
map_metric = MeanAveragePrecision(
    iou_type="segm",
    iou_thresholds=np.arange(start=0.05, stop=1.00, step=0.05).tolist(),
    extended_summary=True,
)
for idx, image_path in enumerate(dataset_images):
    print(f"Processing image {idx + 1}/{len(dataset_images)}")
    print(f"Image path: {image_path}")
    indexes = dataset.dataframe[dataset.dataframe["image_file"] == image_path].index
    # Load all indexes and get masks
    masks = []
    for index in indexes:
        image, labels = dataset[index]
        masks.append(labels["masks"])
    # Union masks
    gt_mask = torch.zeros_like(masks[0])
    for mask in masks:
        gt_mask = torch.logical_or(gt_mask, mask)
    # Convert to torch uint8
    gt_mask = gt_mask.byte()
    print(f"GT Mask shape: {gt_mask.shape}")
    # Apply SLIC segmentation
    try:
        slic_superpixels = slicdbscan(image[0].unsqueeze(0))
    except Exception as exc:
        print(f"Exception: {exc}")
        continue
    # Get dbscan features
    features = slicdbscan.dbscan.last_feature_set
    # Drop all columns starting with "diagnostics_"
    features = features.loc[:, ~features.columns.str.startswith("diagnostics_")]
    cluster_features = features.groupby("dbscan_label").mean()
    # Remove border cluster (cluster with lowest energy)
    border_cluster = cluster_features["original_firstorder_Energy"].idxmin()
    # Set border cluster to 0
    slic_superpixels[slic_superpixels == border_cluster] = 0
    # Calculate silhouette score
    # Drop rows with with dbscan_label == border cluster
    feature_array_pca = (
        slicdbscan.dbscan.last_feature_set_pca.drop(
            index=slicdbscan.dbscan.last_feature_set_pca[
                slicdbscan.dbscan.last_feature_set_pca["dbscan_label"] == border_cluster
            ].index
        )
        .drop(columns=["dbscan_label"])
        .to_numpy()
    )

    labels_pca = slicdbscan.dbscan.last_feature_set_pca.drop(
        index=slicdbscan.dbscan.last_feature_set_pca[
            slicdbscan.dbscan.last_feature_set_pca["dbscan_label"] == border_cluster
        ].index
    )["dbscan_label"].to_numpy()
    silhouette_pca = silhouette_score(feature_array_pca, labels_pca, metric="euclidean")
    # Binarize mask
    bin_mask = slic_superpixels > 0
    print(f"Binary Mask shape: {bin_mask.shape}")
    # Plot gt_mask and bin_mask
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(gt_mask[0], cmap="Oranges")
    axes[1].imshow(bin_mask[0], cmap="Oranges")
    # Print shapes
    # Calculate iou
    iou = calc_iou(bin_mask, gt_mask)
    preds = [
        {
            "masks": bin_mask,
            "scores": torch.tensor([1.0]),
            "labels": torch.tensor([1]),
        }
    ]
    target = [
        {
            "masks": gt_mask,
            "labels": torch.tensor([1]),
        }
    ]
    maps.append(map_metric(preds=preds, target=target))
    ious.append(iou)
    silhouette_scores_pca.append(silhouette_pca)
    ns_clusters.append(slic_superpixels.max())
    mean_iou = np.mean(ious)
    metrics_set = slicdbscan.dbscan.last_metrics
    metrics_set["num_superpixels"] = slic_superpixels.max()
    metrics_set["image_shape"] = image.shape
    metrics_set["image_path"] = image_path
    metrics_set["iou"] = iou
    metrics_set["silhouette_pca"] = silhouette_pca

    metrics_noborder.append(metrics_set)
    print(
        f"Silhouette Score: {silhouette_pca} (Mean: {np.mean(silhouette_scores_pca)})"
    )
    print(f"IoU: {iou} (Mean: {mean_iou})")
    print(f"# of clusters: {slic_superpixels.max()} (Mean: {np.mean(ns_clusters)})")
    plt.close(fig)
metrics_df_noborder = pd.DataFrame(metrics_noborder)

# %%
metrics_df_noborder

# %%
print(f"Mean IoU: {np.mean(ious)}")
print(map_metric.compute())
print(f"Silhouette Score: {np.mean(silhouette_scores_pca)}")

# %%
print({k: float(i) for k, i in map_metric.compute()["ious"].items()})

# %% [markdown]
# #### Calculate mAP without border clusters and noise cluster

# %%
# Set colors
colors = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "gray",
    "black",
    "white",
    "cyan",
    "magenta",
    "teal",
    "navy",
    "olive",
    "maroon",
    "lime",
    "aqua",
    "silver",
]
# Test the best parameters
slicdbscan_best_params = study.best_params
slicdbscan = SLICDBSCANSegmentation(
    slic_algorithm=slicdbscan_best_params["slic_algorithm"],
    region_size=slicdbscan_best_params["region_size"],
    ruler=slicdbscan_best_params["ruler"],
    slic_iterations=slicdbscan_best_params["slic_iterations"],
    gaussian_blur=slicdbscan_best_params["gaussian_blur"],
    sigma=slicdbscan_best_params["sigma"],
    enforce_connectivity=slicdbscan_best_params["enforce_connectivity"],
    min_element_size=slicdbscan_best_params["min_element_size"],
    min_label_size=slicdbscan_best_params["min_label_size"],
    use_radiomics=slicdbscan_best_params["use_radiomics"],
    dbscan_features=slicdbscan_best_params["dbscan_features"],
    use_pca=slicdbscan_best_params["use_pca"],
    eps=slicdbscan_best_params["eps"],
    min_samples=slicdbscan_best_params["min_samples"],
    n_jobs=-1,
    return_mask=True,
)
print(slicdbscan_best_params)
# Iterate through dataset
subfolder = "slicdbscan_5r_map_noborder"
metrics_nobordernoise = []
metrics_df_nobordernoise = None
dataset_images = dataset.dataframe["image_file"].unique()
ious = []
maps = []
silhouette_scores = []
silhouette_scores_pca = []
ns_clusters = []
map_metric = MeanAveragePrecision(
    iou_type="segm",
    iou_thresholds=np.arange(start=0.05, stop=1.00, step=0.05).tolist(),
    extended_summary=True,
)
for idx, image_path in enumerate(dataset_images):
    print(f"Processing image {idx + 1}/{len(dataset_images)}")
    print(f"Image path: {image_path}")
    indexes = dataset.dataframe[dataset.dataframe["image_file"] == image_path].index
    # Load all indexes and get masks
    masks = []
    for index in indexes:
        image, labels = dataset[index]
        masks.append(labels["masks"])
    # Union masks
    gt_mask = torch.zeros_like(masks[0])
    for mask in masks:
        gt_mask = torch.logical_or(gt_mask, mask)
    # Convert to torch uint8
    gt_mask = gt_mask.byte()
    print(f"GT Mask shape: {gt_mask.shape}")
    # Apply SLIC segmentation
    try:
        slic_superpixels = slicdbscan(image[0].unsqueeze(0))
    except Exception as exc:
        print(f"Exception: {exc}")
        continue
    # Get dbscan features
    features = slicdbscan.dbscan.last_feature_set
    # Drop all columns starting with "diagnostics_"
    features = features.loc[:, ~features.columns.str.startswith("diagnostics_")]
    cluster_features = features.groupby("dbscan_label").mean()
    # Remove border cluster (cluster with lowest energy)
    border_cluster = cluster_features["original_firstorder_Energy"].idxmin()
    # Set noise cluster (1) to 0
    slic_superpixels[slic_superpixels == 1] = 0
    # Set border cluster to 0
    slic_superpixels[slic_superpixels == border_cluster] = 0
    # Calculate silhouette score
    # Drop rows with with dbscan_label == noise cluster
    # Drop rows with with dbscan_label == border cluster
    feature_array_pca = (
        slicdbscan.dbscan.last_feature_set_pca.drop(
            index=slicdbscan.dbscan.last_feature_set_pca[
                slicdbscan.dbscan.last_feature_set_pca["dbscan_label"] == 1
            ].index
        )
        .drop(
            index=slicdbscan.dbscan.last_feature_set_pca[
                slicdbscan.dbscan.last_feature_set_pca["dbscan_label"] == border_cluster
            ].index
        )
        .drop(columns=["dbscan_label"])
        .to_numpy()
    )

    labels_pca = (
        slicdbscan.dbscan.last_feature_set_pca.drop(
            index=slicdbscan.dbscan.last_feature_set_pca[
                slicdbscan.dbscan.last_feature_set_pca["dbscan_label"] == 1
            ].index
        )
        .drop(
            index=slicdbscan.dbscan.last_feature_set_pca[
                slicdbscan.dbscan.last_feature_set_pca["dbscan_label"] == border_cluster
            ].index
        )["dbscan_label"]
        .to_numpy()
    )
    silhouette_pca = silhouette_score(feature_array_pca, labels_pca, metric="euclidean")
    bin_mask = slic_superpixels > 0
    print(f"Binary Mask shape: {bin_mask.shape}")
    # Plot gt_mask and bin_mask
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(gt_mask[0], cmap="Oranges")
    axes[1].imshow(bin_mask[0], cmap="Oranges")
    # Print shapes
    # Calculate iou
    iou = calc_iou(bin_mask, gt_mask)
    preds = [
        {
            "masks": bin_mask,
            "scores": torch.tensor([1.0]),
            "labels": torch.tensor([1]),
        }
    ]
    target = [
        {
            "masks": gt_mask,
            "labels": torch.tensor([1]),
        }
    ]
    maps.append(map_metric(preds=preds, target=target))
    ious.append(iou)
    silhouette_scores_pca.append(silhouette_pca)
    ns_clusters.append(slic_superpixels.max())
    mean_iou = np.mean(ious)
    metrics_set = slicdbscan.dbscan.last_metrics
    metrics_set["num_superpixels"] = slic_superpixels.max()
    metrics_set["image_shape"] = image.shape
    metrics_set["image_path"] = image_path
    metrics_set["iou"] = iou
    metrics_set["silhouette_pca"] = silhouette_pca

    metrics_nobordernoise.append(metrics_set)
    print(
        f"Silhouette Score: {silhouette_pca} (Mean: {np.mean(silhouette_scores_pca)})"
    )
    print(f"IoU: {iou} (Mean: {mean_iou})")
    plt.close(fig)
metrics_df_nobordernoise = pd.DataFrame(metrics_nobordernoise)

# %%
print(metrics_df_nobordernoise)

# %%
print(f"Mean IoU: {np.mean(ious)}")
print(map_metric.compute())
print(f"Silhouette Score: {np.mean(silhouette_scores_pca)}")

# %%
print({k: float(i) for k, i in map_metric.compute()["ious"].items()})
