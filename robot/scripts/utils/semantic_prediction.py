# The following code is largely borrowed from
# https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py and
# https://github.com/facebookresearch/detectron2/blob/master/demo/predictor.py

import argparse
import time
import os
n_dir = os.path.dirname(__file__)

import torch
import numpy as np

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data.catalog import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import ColorMode, Visualizer
import detectron2.data.transforms as T

from constants import coco_categories_mapping

from constants import coco_categories_mapping

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cv2
from detectron2.data import MetadataCatalog
import clip
from PIL import Image
class SemanticPredMaskRCNN():

    def __init__(self, args):
        self.segmentation_model = ImageSegmentation(args)
        self.args = args

    def get_prediction(self, img):
        args = self.args
        image_list = []
        img = img[:, :, ::-1]
        image_list.append(img)
        seg_predictions, vis_output = self.segmentation_model.get_predictions(
            image_list, visualize=args.visualize == 2)
        # print("seg_predictions=",seg_predictions)
        # print("vis_output=",vis_output)
        if args.visualize == 2:
            img = vis_output.get_image()

        semantic_input = np.zeros((img.shape[0], img.shape[1], 15 + 1))

        for j, class_idx in enumerate(
                seg_predictions[0]['instances'].pred_classes.cpu().numpy()):
            if class_idx in list(coco_categories_mapping.keys()):
                idx = coco_categories_mapping[class_idx]
                obj_mask = seg_predictions[0]['instances'].pred_masks[j] * 1.
                semantic_input[:, :, idx] += obj_mask.cpu().numpy()

        # 定义的颜色列表
        bgr_colors_list = [
        [0.12156862745098039, 0.47058823529411764, 0.7058823529411765],
        [0.9400000000000001, 0.7818, 0.66],
        [0.9400000000000001, 0.8868, 0.66],
        [0.8882000000000001, 0.9400000000000001, 0.66],
        [0.7832000000000001, 0.9400000000000001, 0.66],
        [0.6782000000000001, 0.9400000000000001, 0.66],
        [0.66, 0.9400000000000001, 0.7468000000000001],
        [0.66, 0.9400000000000001, 0.8518000000000001],
        [0.66, 0.9232, 0.9400000000000001],
        [0.66, 0.8182, 0.9400000000000001],
        [0.66, 0.7132, 0.9400000000000001],
        [0.7117999999999999, 0.66, 0.9400000000000001],
        [0.8168, 0.66, 0.9400000000000001],
        [0.9218, 0.66, 0.9400000000000001],
        [0.9400000000000001, 0.66, 0.8531999999999998]]

        colors_list = [[b, g, r] for [r, g, b] in bgr_colors_list]

        # 确保颜色的长度和coco_categories_mapping一致
        assert len(colors_list) == len(coco_categories_mapping)

        # 获取COCO数据集的元数据，包括类别名称
        coco_metadata = MetadataCatalog.get("coco_2017_train")
        class_names = coco_metadata.thing_classes

        # 为每个类别分配颜色
        class_colors = {mapped_id: np.array(color) * 255 for mapped_id, color in zip(coco_categories_mapping.values(), colors_list)}

        # 假设 img 是BGR格式的原始图像，seg_predictions 是模型的输出
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output_img = img_rgb.copy()
        # img_rgb_clip = img_rgb.copy()

        # 获取seg_predictions中的第一个元素的 'instances'
        instances = seg_predictions[0]["instances"]

        if instances.has("pred_masks") and instances.has("pred_classes") and instances.has("pred_boxes"):
            masks = instances.pred_masks
            classes = instances.pred_classes
            boxes = instances.pred_boxes.tensor.cpu().numpy()

            for j in range(len(masks)):
                class_id = classes[j].item()
                if class_id in coco_categories_mapping:
                    mapped_class_id = coco_categories_mapping[class_id]
                    mask = masks[j].cpu().numpy()
                    bbox = boxes[j]

                    # 绘制边界框
                    x1, y1, x2, y2 = bbox.astype(int)
                    cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # 获取映射类别的颜色
                    color = class_colors[mapped_class_id]

                    # 创建彩色mask图像
                    mask_colored = np.zeros_like(output_img)
                    mask_colored[mask] = color

                    # 将彩色mask叠加到output_img上
                    output_img = cv2.addWeighted(output_img, 1, mask_colored, 0.5, 0)

                    # 获取类别名称
                    class_name = class_names[class_id]

                    # 在边界框左上角放置类别名称文本
                    cv2.putText(output_img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        
        output_img=cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)


        return semantic_input, output_img


def compress_sem_map(sem_map):
    c_map = np.zeros((sem_map.shape[1], sem_map.shape[2]))
    for i in range(sem_map.shape[0]):
        c_map[sem_map[i] > 0.] = i + 1
    return c_map


class ImageSegmentation():
    def __init__(self, args):
        string_args = """
            --config-file /home/sly/objnav_ws_6/src/FSE-ROS-main/fsp_node/scripts/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
            --input input1.jpeg
            --confidence-threshold {}
            --opts MODEL.WEIGHTS
            detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
            """.format(args.sem_pred_prob_thr)

        if args.sem_gpu_id == -2:
            string_args += """ MODEL.DEVICE cpu"""
        else:
            string_args += """ MODEL.DEVICE cuda:{}""".format(args.sem_gpu_id)

        string_args = string_args.split()

        args = get_seg_parser().parse_args(string_args)
        logger = setup_logger()
        logger.info("Arguments: " + str(args))

        cfg = setup_cfg(args)
        self.demo = VisualizationDemo(cfg)

    def get_predictions(self, img, visualize=0):
        return self.demo.run_on_image(img, visualize=visualize)


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = \
        args.confidence_threshold
    cfg.freeze()
    return cfg


def get_seg_parser():
    parser = argparse.ArgumentParser(
        description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.predictor = BatchPredictor(cfg)

    def run_on_image(self, image_list, visualize=0):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        all_predictions = self.predictor(image_list)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.

        if visualize:
            predictions = all_predictions[0]
            image = image_list[0]
            visualizer = Visualizer(
                image, self.metadata, instance_mode=self.instance_mode)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_output = visualizer.draw_panoptic_seg_predictions(
                    panoptic_seg.to(self.cpu_device), segments_info
                )
            else:
                if "sem_seg" in predictions:
                    vis_output = visualizer.draw_sem_seg(
                        predictions["sem_seg"].argmax(
                            dim=0).to(self.cpu_device)
                    )
                if "instances" in predictions:
                    instances = predictions["instances"].to(self.cpu_device)
                    vis_output = visualizer.draw_instance_predictions(
                        predictions=instances)

        return all_predictions, vis_output


class BatchPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a list of input images.

    Compared to using the model directly, this class does the following
    additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by
         `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take a list of input images

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained
            from cfg.DATASETS.TEST.

    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, image_list):
        """
        Args:
            image_list (list of np.ndarray): a list of images of
                                             shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for all images.
                See :doc:`/tutorials/models` for details about the format.
        """
        inputs = []
        for original_image in image_list:
            # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = original_image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            instance = {"image": image, "height": height, "width": width}

            inputs.append(instance)

        with torch.no_grad():
            predictions = self.model(inputs)
            return predictions
