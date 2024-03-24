# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.import_utils import check_package_minimum_version, check_requirements

logger = logging.getLogger(__name__)


class DINODetectionModel(DetectionModel):
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_config: Optional[str] = None,
        model: Optional[Any] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        category_mapping: Optional[Dict] = None,
        category_remapping: Optional[Dict] = None,
        load_at_init: bool = True,
        image_size: int = None,
    ):
        """
        Init object detection/instance segmentation model.
        Args:
            model_path: str
                Path for the instance segmentation model weight
            config_path: str
                Path for the mmdetection instance segmentation model config file
            device: str
                Torch device, "cpu" or "cuda"
            mask_threshold: float
                Value to threshold mask pixels, should be between 0 and 1
            confidence_threshold: float
                All predictions with score < confidence_threshold will be discarded
            category_mapping: dict: str to str
                Mapping from category id (str) to category name (str) e.g. {"1": "pedestrian"}
            category_remapping: dict: str to int
                Remap category ids based on category names, after performing inference e.g. {"car": 3}
            load_at_init: bool
                If True, automatically loads the model at initalization
            image_size: int
                Inference input size.
        """
        super().__init__(
            model_path=model_path,
            model=model,
            config_path=config_path,
            device=device,
            mask_threshold=mask_threshold,
            confidence_threshold=confidence_threshold,
            category_mapping=category_mapping,
            category_remapping=category_remapping,
            load_at_init=False,
            image_size=image_size,
        )
        self.model_path = model_path
        self.model_config = model_config
        self.config_path = config_path
        self.model = None
        self.device = device
        self.mask_threshold = mask_threshold
        self.confidence_threshold = confidence_threshold
        self.category_mapping = category_mapping
        self.category_remapping = category_remapping
        self.image_size = image_size
        self._original_predictions = None
        self._object_prediction_list_per_image = None
        self._model_postprocessors = None

        self.set_device()

        # automatically load model if load_at_init is True
        if load_at_init:
            if model:
                self.set_model(model)
            else:
                self.load_model(self.model_config, self.model_path)

    def check_dependencies(self) -> None:
        check_requirements(["torch", "yolov5"])

    def load_model(self, config, resume_path):
        """
        Detection model is initialized and set to self.model.
        """

        import torch, json
        import numpy as np

        from dino.main import build_model_main
        from dino.util.slconfig import SLConfig
        from dino.datasets import build_dataset
        from dino.util.visualizer import COCOVisualizer
        from dino.util import box_ops

        try:
            args = SLConfig.fromfile(config)
            args.device = 'cuda'
            model, criterion, postprocessors = build_model_main(args)
            checkpoint = torch.load(resume_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            _ = model.eval()

            self.set_model(model)
            self._model_postprocessors = postprocessors
        except Exception as e:
            raise TypeError("model_path is not a valid dino model path: ", e)

    def set_model(self, model: Any):
        """
        Sets the underlying RT_DETR model.
        Args:
            model: Any
                A RT_DETR model
        """

        self.model = model

        # set category_mapping
        if not self.category_mapping:
            # category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
            # self.category_mapping = category_mapping
            raise ValueError("category_mapping is required for RTDETRDetectionModel")

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """
        
        from PIL import Image
        import torch
        import torchvision.transforms.v2 as T

        imgCpy = image.copy()

        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")
        
        image_height, image_width, _ = imgCpy.shape
        
        transform = T.Compose([
            T.ToPILImage(),  # Convert ndarray to PIL Image
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image, _ = transform(image, None)
        output = self.model.cuda()(image[None].cuda())
        output = self._model_postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
        boxes = output['boxes'].cpu().tolist()

        new_boxes = []
        for out in boxes:
            new_boxes.append([out[0]*image_width, out[1]*image_height, out[2]*image_width, out[3]*image_height])
        
        scores = output['scores'].cpu().tolist()
        labels = output['labels'].cpu().tolist()

        preds_xyxy = []
        for box, score, label in zip(new_boxes, scores, labels):
            if score >= self.confidence_threshold:
                preds_xyxy.append([box[0], box[1], box[2], box[3], float(score), int(label)])

        self._original_predictions = [preds_xyxy]

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        if self.category_remapping:
            return len([category_name for category_name in self.category_remapping.keys()])

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        # import yolov5
        # from packaging import version

        # if version.parse(yolov5.__version__) < version.parse("6.2.0"):
        #     return False
        # else:
        #     return False  # fix when yolov5 supports segmentation models

        raise NotImplementedError("has_mask method is not implemented for RTDETRDetectionModel")

    @property
    def category_names(self):
        # convert self.category_remapping dict to list
        if self.category_remapping:
            return [category_name for category_name in self.category_remapping.keys()]

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # handle all predictions
        object_prediction_list_per_image = []
        for image_ind, image_predictions_in_xyxy_format in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []

            # process predictions
            for prediction in image_predictions_in_xyxy_format:
                x1 = prediction[0]
                y1 = prediction[1]
                x2 = prediction[2]
                y2 = prediction[3]
                bbox = [x1, y1, x2, y2]
                score = prediction[4]
                category_id = int(prediction[5])
                category_name = self.category_mapping[str(category_id)]

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])

                # fix out of image box coords
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                # ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    bool_mask=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image
