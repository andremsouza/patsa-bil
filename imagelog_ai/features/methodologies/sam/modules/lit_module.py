"""Implementation of the Lightning Module for SAM fine-tuning."""

# %% [markdown]
# ## Imports

# %%
# Change directory to the root of the project
if __name__ == "__main__":
    import os
    import sys

    import dotenv
    import matplotlib.pyplot as plt
    from skimage.color import label2rgb
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader

    os.chdir(os.getcwd().split("imagelog_ai")[0])
    print(f"cwd: {os.getcwd()}")
    dotenv.load_dotenv()
    sys.path.append(os.getenv("PACKAGEPATH", ""))

    from imagelog_ai.features.methodologies.sam.datasets.dataset import SamDataset

# %%
import os
from typing import Any, Final, Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanSquaredError
from torchmetrics.detection import MeanAveragePrecision

from torchmetrics import MetricCollection

from imagelog_ai.base.base_lit_module import BaseLitModule
from imagelog_ai.features.methodologies.sam.neural_networks.sam import (
    Sam,
    DiceLoss,
    FocalLoss,
)
from imagelog_ai.features.methodologies.sam.utils.iou import calc_iou

# %% [markdown]
# ## Constants

# %%
torch.set_float32_matmul_precision("high")

# %% [markdown]
# ## Classes


class SamModule(BaseLitModule):
    """A PyTorch Lightning module for the Sam model.

    Parameters
    ----------
    early_stopping_patience : int, optional
        The number of epochs with no improvement after which training will be stopped, by default None
    lr_scheduler_patience : int, optional
        The number of epochs with no improvement after which the learning rate scheduler will reduce the learning rate, by default None
    learning_rate : float, optional
        The learning rate for the optimizer, by default 0.00001
    weight_decay : float, optional
        The weight decay (L2 penalty) for the optimizer, by default 0
    checkpoint_fname : str, optional
        The filename of the checkpoint for the Sam model, by default "sam_vit_l_0b3195.pth"
    sam_type : str, optional
        The type of Sam model to use, by default "vit_l"
    freeze_image_encoder : bool, optional
        Whether to freeze the image encoder in the Sam model, by default True
    freeze_prompt_encoder : bool, optional
        Whether to freeze the prompt encoder in the Sam model, by default True
    freeze_mask_decoder : bool, optional
        Whether to freeze the mask decoder in the Sam model, by default False
    """

    def __init__(
        self,
        early_stopping_patience: int = None,
        lr_scheduler_patience: int = None,
        learning_rate: float = 0.00001,
        weight_decay: float = 0,
        checkpoint_fname: str = "sam_vit_l_0b3195.pth",
        sam_type: str = "vit_l",
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        freeze_mask_decoder: bool = False,
    ) -> None:
        """
        Initializes the SamModule.

        Parameters
        ----------
        early_stopping_patience : int, optional
            The number of epochs with no improvement after which training will be stopped, by default None
        lr_scheduler_patience : int, optional
            The number of epochs with no improvement after which the learning rate scheduler will reduce the learning rate, by default None
        learning_rate : float, optional
            The learning rate for the optimizer, by default 0.00001
        weight_decay : float, optional
            The weight decay (L2 penalty) for the optimizer, by default 0
        checkpoint_fname : str, optional
            The filename of the checkpoint for the Sam model, by default "sam_vit_l_0b3195.pth"
        sam_type : str, optional
            The type of Sam model to use, by default "vit_l"
        freeze_image_encoder : bool, optional
            Whether to freeze the image encoder in the Sam model, by default True
        freeze_prompt_encoder : bool, optional
            Whether to freeze the prompt encoder in the Sam model, by default True
        freeze_mask_decoder : bool, optional
            Whether to freeze the mask decoder in the Sam model, by default False
        """
        super().__init__(
            early_stopping_patience,
            lr_scheduler_patience,
            learning_rate,
            weight_decay,
        )
        torch.hub.set_dir(os.path.join(os.getcwd(), "data/PretrainedModels/Sam"))
        self.activation_func: Final = torch.nn.Identity()
        # self.loss_func: Final = torch.nn.MSELoss()
        self.loss_dice: Final = DiceLoss()
        self.loss_iou: Final = torch.nn.MSELoss(reduction="sum")
        self.loss_focal: Final = FocalLoss()
        self.checkpoint_fname: str = checkpoint_fname
        self.sam_type: str = sam_type
        self.freeze_image_encoder: bool = freeze_image_encoder
        self.freeze_prompt_encoder: bool = freeze_prompt_encoder
        self.freeze_mask_decoder: bool = freeze_mask_decoder
        self.train_metrics, self.val_metrics, self.test_metrics = self._build_metrics()
        self.map_metric = {
            "Train": MeanAveragePrecision(
                iou_type="segm",
                iou_thresholds=np.arange(start=0.05, stop=1.00, step=0.05).tolist(),
            ),
            "Val": MeanAveragePrecision(
                iou_type="segm",
                iou_thresholds=np.arange(start=0.05, stop=1.00, step=0.05).tolist(),
            ),
            "Test": MeanAveragePrecision(
                iou_type="segm",
                iou_thresholds=np.arange(start=0.05, stop=1.00, step=0.05).tolist(),
            ),
        }

        self.network = self._build_network()

    def _build_network(self) -> Any:
        """Method to build the module's network.

        Returns
        -------
        network: Any
            Returns the network of the module.
        """
        return Sam(
            checkpoint_fname=self.checkpoint_fname,
            sam_type=self.sam_type,
            freeze_image_encoder=self.freeze_image_encoder,
            freeze_prompt_encoder=self.freeze_prompt_encoder,
            freeze_mask_decoder=self.freeze_mask_decoder,
        )

    def _build_metrics(self) -> tuple:
        """Method to build the module's metrics.

        Returns
        -------
        tuple
            Returns a tuple containing the training, validation, and testing metrics.
        """
        metrics_dict = {
            "MSE": MeanSquaredError,
            "RMSE": MeanSquaredError,
        }
        metric_collection = {}
        for metric_name, metric in metrics_dict.items():
            metrics_kwargs = {}
            if metric_name == "RMSE":
                metrics_kwargs["squared"] = False
            if metric_name == "MSE":
                metrics_kwargs["squared"] = True
            if metric_name == "mAP":
                metrics_kwargs["iou_type"] = "segm"
            metric_collection[metric_name] = metric(**metrics_kwargs)
        metric_collection = MetricCollection(metric_collection)

        train_metrics = metric_collection.clone(postfix="_Train")
        val_metrics = metric_collection.clone(postfix="_Val")
        test_metrics = metric_collection.clone(postfix="_Test")

        return train_metrics, val_metrics, test_metrics

    def forward(
        self, images: Tensor, bboxes: Optional[Tensor], masks: Optional[Tensor]
    ) -> Any:
        """Method to forward pass the input tensor through the network.

        Parameters
        ----------
        images: Tensor
            The input tensor to be passed through the network.
        bboxes: Optional[Tensor]
            The bounding boxes of the input tensor.
        masks: Optional[Tensor]
            The masks of the input tensor.

        Returns
        -------
        Any
            Returns the output of the network.
        """
        return self.network(images, bboxes, masks)

    def shared_step(self, batch: tuple[Tensor], prefix: str) -> Optional[Tensor]:
        """Method to perform a shared step for the module.

        Parameters
        ----------
        batch: tuple[Tensor]
            Batch of data to be used in the shared step.
        prefix: str
            Prefix to be used in the shared step.

        Returns
        -------
        Optional[Tensor]
            Returns the loss of the shared step.
        """
        x, y = batch
        # y is a list of dictionaries with keys "boxes" and "masks"
        boxes = y["boxes"]
        masks = y["masks"]
        assert boxes.shape[0] == masks.shape[0] == x.shape[0], (
            f"Batch size mismatch. x: {x.shape[0]}"
            f", boxes: {boxes.shape[0]}"
            f", masks: {masks.shape[0]}"
        )

        pred_masks, pred_ious = self(images=x, bboxes=boxes, masks=None)
        num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
        loss_focal = torch.tensor(0.0, device=self.device)
        loss_dice = torch.tensor(0.0, device=self.device)
        loss_iou = torch.tensor(0.0, device=self.device)
        for pred_mask, gt_mask, pred_iou in zip(pred_masks, masks, pred_ious):
            batch_iou = calc_iou(pred_mask, gt_mask)
            loss_focal += self.loss_focal(pred_mask, gt_mask)
            loss_dice += self.loss_dice(pred_mask, gt_mask)
            loss_iou += self.loss_iou(pred_iou, batch_iou) / num_masks

        loss = 20.0 * loss_focal + loss_dice + loss_iou

        pred_masks = torch.stack(pred_masks)
        binary_masks = F.normalize(F.threshold(pred_masks, 0.0, 0))

        # Log loss
        self.log_dict({f"FocalLoss_{prefix}": loss_focal.detach()}, prog_bar=True)
        self.log_dict({f"DiceLoss_{prefix}": loss_dice.detach()}, prog_bar=True)
        self.log_dict({f"IoULoss_{prefix}": loss_iou.detach()}, prog_bar=True)
        self.log_dict({f"Loss_{prefix}": loss.detach()}, prog_bar=True)
        # Log map
        map_preds = [
            {
                "masks": mask.detach().to(torch.uint8),
                "scores": torch.tensor([1.0]),
                "labels": torch.tensor([0]),
            }
            for mask in binary_masks
        ]
        map_target = [
            {"masks": mask.detach().to(torch.uint8), "labels": torch.tensor([0])}
            for mask in masks
        ]
        map_metric = self.map_metric[prefix](map_preds, map_target)
        for map_key, map_value in map_metric.items():
            self.log_dict({f"mAP_{map_key}_{prefix}": map_value}, prog_bar=True)
        # Log remaining metrics
        self._log_metrics(binary_masks, masks, prefix)

        return loss

    def predict_step(self, batch: tuple[Tensor], batch_idx: int) -> list:
        raise NotImplementedError

    def configure_callbacks(self) -> list[Callback]:
        """Function that is run by lightning when using the module.
                        Configures the callbacks used by the Model.

        Returns
        -------
        List[Callback]
                List of Callbacks
        """
        callback_list = []
        if self.hparams.early_stopping_patience is not None:
            callback_list.append(
                EarlyStopping(
                    patience=self.hparams.early_stopping_patience,
                    stopping_threshold=1e-04,
                    monitor="Loss_Val",
                    mode="min",
                )
            )
        # Save top 1 model
        callback_list.append(
            ModelCheckpoint(
                monitor="Loss_Val",
                mode="min",
                save_top_k=1,
                save_last=True,
                dirpath="models",
                filename="sam-{epoch:02d}-{Loss_Val:.2f}",
            )
        )

        return callback_list

    def configure_optimizers(self):
        """Function that is run by lightning when using the module.
                        Configures the optimizer used by the Model.

        Returns
        -------
        dict[str, Any]
                Dictionary of optimizer configuration
        """
        optimizer = torch.optim.AdamW(
            self.network.model.mask_decoder.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.lr_scheduler_patience is not None:
            scheduler = ReduceLROnPlateau(
                patience=self.hparams.lr_scheduler_patience,
                cooldown=0,
                optimizer=optimizer,
                factor=0.5,
                mode="min",
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "Loss_Val",
                },
            }
        else:
            return {"optimizer": optimizer}


# %% [markdown]
# ## Main (Testing)

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
    project_name: str = "SAMFineTuning"
    preprocess_name: str = "lstudio"
    list_datasource_names: list[str] = [
        "WellD",
    ]
    train_list_datasource_names: list[str] = [
        "WellD",
    ]
    val_list_datasource_names: list[str] = [
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
    # Get dataset.dataframe to separate train and val
    dataframe = dataset.dataframe
    train_dataframe, val_dataframe = train_test_split(
        dataframe, test_size=0.2, random_state=42
    )
    train_dataset = SamDataset(
        project_name=project_name,
        preprocess_name=preprocess_name,
        list_datasource_names=train_list_datasource_names,
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
    train_dataset.dataframe = train_dataframe
    val_dataset = SamDataset(
        project_name=project_name,
        preprocess_name=preprocess_name,
        list_datasource_names=val_list_datasource_names,
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
    val_dataset.dataframe = val_dataframe

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=os.cpu_count()
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=os.cpu_count()
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count()
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
    # mlflow.pytorch.autolog()
    # # Train model
    # with mlflow.start_run():
    #     trainer.fit(model, train_dataloader, val_dataloader)
    #     # Save model to MLFlow
    #     mlflow.pytorch.log_model(model, "models")
    # Load from checkpoint
    # ! Comment these lines to train the model
    model = SamModule.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
        checkpoint_path="models/sam-epoch=07-Loss_Val=0.43.ckpt",
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
