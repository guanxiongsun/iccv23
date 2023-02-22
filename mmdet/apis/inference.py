# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torchvision
from torch import nn
import cv2
from pathlib import Path
import os
import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector, build_model
from typing import Dict, Iterable, Callable
from tools.visualisation import utils as vis_utils


import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image



def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    config.model.pretrained = None
    config.model.train_cfg = None
    cfg = config
    # for video models
    if 'detector' in cfg.model.keys():
        # multi-frame video model
        if cfg.model.get('type', False) in ("SELSA", "MAMBA", "RDN",
                                            "FCOSAtt", "YOLOAtt", "CenterNetAtt",
                                            "VideoPrompt", 'DeepVideoPrompt',
                                            'SELSAVideoPrompt'):
            model = build_model(cfg.model)

        # single-frame video base model
        else:
            cfg.model = cfg.model.detector
            model = build_detector(
                cfg.model,
                test_cfg=cfg.get('test_cfg'))

    # for single-frame models
    else:
        model = build_detector(
            cfg.model,
            test_cfg=cfg.get('test_cfg'))

    # model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage:
    """Deprecated.

    A simple pipeline to load image.
    """

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        warnings.simplefilter('once')
        warnings.warn('`LoadImage` is deprecated and will be removed in '
                      'future releases. You may use `LoadImageFromWebcam` '
                      'from `mmdet.datasets.pipelines.` instead.')
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, data):
        # _ = self.model(x)
        _ = self.model(return_loss=False, rescale=True, **data)
        return self._features


class FasterRCNNBoxScoreTarget:
    def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()

        if len(model_outputs["boxes"]) == 0:
            return output

        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            if torch.cuda.is_available():
                box = box.cuda()

            ious = torchvision.ops.box_iou(box, model_outputs["boxes"])
            index = ious.argmax()
            if ious[0, index] > self.iou_threshold and model_outputs["labels"][index] == label:
                score = ious[0, index] + model_outputs["scores"][index]
                output = output + score
        return output


def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    # cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    data['img'][0] = data['img'][0].unsqueeze(0)
    # forward the model
    with torch.no_grad():
        feature_hook = FeatureExtractor(model, layers=["backbone"])
        results = feature_hook(data)

        # save backbone features visualization
        all_feats = results['backbone']
        for i in range(len(all_feats)):
            feat = all_feats[i][0]
            feat_img = vis_utils.feature2im(feat)
            # vis_utils.plt_show(feat_img)

            # save img to file
            p = Path(img)
            img_foler = os.path.join(p.parent, 'backbone_feat')
            if not os.path.exists(img_foler):
                os.makedirs(img_foler)

            new_filepath = os.path.join(img_foler, p.stem + f'_backbone_feat_{i}.jpg')
            pth_filepath = os.path.join(img_foler, p.stem + f'_backbone_feat_{i}.pth')
            # mmcv.imwrite(feat_img, new_filepath)
            vis_utils.plt_save(feat_img, new_filepath)
            torch.save(feat, pth_filepath)

        # results = model(return_loss=False, rescale=True, **data)
    return results
    # if not is_batch:
    #     return results[0]
    # else:
    #     return results




def inference_detector_gradcam(model, imgs):
    """Inference image(s) with the detector.

        Args:
            model (nn.Module): The loaded detector.
            imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
               Either image files or loaded images.

        Returns:
            If imgs is a list or tuple, the same length list type results
            will be returned, otherwise return the detection results directly.
        """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    # cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    data['img'][0] = data['img'][0].unsqueeze(0)
    # forward the model
    with torch.no_grad():
        # results = model(data['img'][0])
        def fasterrcnn_reshape_transform(x):
            target_size = x[-1].size()[-2:]
            activations = []
            for _x in x:
                activations.append(torch.nn.functional.interpolate(torch.abs(_x), target_size, mode='bilinear'))
            activations = torch.cat(activations, axis=1)
            return activations




        target_layers = [model.backbone]

        cam = EigenCAM(model, target_layers, use_cuda=True,
                       reshape_transform=fasterrcnn_reshape_transform)
        tensor = data['img'][0]
        grayscale_cam = cam(tensor, targets=[torch.Tensor([26]).cuda()])[0, :, :]
        cv_img = cv2.imread(img)
        cv_img = cv2.resize(cv_img, (1008, 576))
        cv_img = np.asarray(cv_img, dtype=np.float64)
        cv_img /= 255

        cam_image = show_cam_on_image(cv_img, grayscale_cam, use_rgb=True)
        Image.fromarray(cam_image)
        #
        # feature_hook = FeatureExtractor(model, layers=["backbone"])
        # results = feature_hook(data)
        #
        # # save backbone features visualization
        # all_feats = results['backbone']
        # for i in range(len(all_feats)):
        #     feat = all_feats[i][0]
        #     feat_img = vis_utils.feature2im(feat)
        #     # vis_utils.plt_show(feat_img)
        #
        #     # save img to file
        #     p = Path(img)
        #     img_foler = os.path.join(p.parent, 'backbone_feat')
        #     if not os.path.exists(img_foler):
        #         os.makedirs(img_foler)
        #
        #     new_filepath = os.path.join(img_foler, p.stem + f'_backbone_feat_{i}.jpg')
        #     pth_filepath = os.path.join(img_foler, p.stem + f'_backbone_feat_{i}.pth')
        #     # mmcv.imwrite(feat_img, new_filepath)
        #     vis_utils.plt_save(feat_img, new_filepath)
        #     torch.save(feat, pth_filepath)

        # results = model(return_loss=False, rescale=True, **data)
    return
    # if not is_batch:
    #     return results[0]
    # else:
    #     return results


async def async_inference_detector(model, imgs):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        img (str | ndarray): Either image files or loaded images.

    Returns:
        Awaitable detection results.
    """
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    results = await model.aforward_test(rescale=True, **data)
    return results


def show_result_pyplot(model,
                       img,
                       result,
                       score_thr=0.3,
                       title='result',
                       wait_time=0):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        title (str): Title of the pyplot figure.
        wait_time (float): Value of waitKey param.
                Default: 0.
    """
    if hasattr(model, 'module'):
        model = model.module
    model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=True,
        wait_time=wait_time,
        win_name=title,
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241))
