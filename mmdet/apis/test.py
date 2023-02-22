# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
from torch import nn
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results
from pathlib import Path
import os
from typing import Dict, Iterable, Callable
from tools.visualisation import utils as vis_utils
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import cv2
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image
import torchvision
import numpy as np


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



def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        if 'ILSVRC2015_val_00000001' not in data['img_metas'][0].data[0][0]['filename']:
            prog_bar.update()
            continue

        if 0 < data['img_metas'][0].data[0][0]['frame_id'] < 339:
            prog_bar.update()
            continue

        if data['img_metas'][0].data[0][0]['frame_id'] > 345:
            prog_bar.update()
            continue

        img_metas = data['img_metas'][0].data[0]
        out_file = osp.join(out_dir, img_metas[0]['ori_filename'])

        with torch.no_grad():
            def fasterrcnn_reshape_transform(x):
                target_size = x[-1].size()[-2:]
                activations = []
                for _x in x:
                    activations.append(torch.nn.functional.interpolate(torch.abs(_x), target_size, mode='bilinear'))
                activations = torch.cat(activations, axis=1)
                return activations

            try:
                target_layers = [model.module.backbone]
            except:
                target_layers = [model.module.detector.backbone]

            cam = EigenCAM(model, target_layers, use_cuda=True,
                           reshape_transform=fasterrcnn_reshape_transform)
            tensor = data['img'][0]
            torch.save(data, 'data.pth')
            labels = data['gt_labels'][0]
            boxes = data['gt_bboxes'][0]
            targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]

            grayscale_cam = cam(tensor, targets=targets)[0, :, :]
            cv_img = cv2.imread(data['img_metas'][0].data[0][0]['filename'])
            cv_img = cv2.resize(cv_img, (1008, 576))
            cv_img = np.asarray(cv_img, dtype=np.float64)
            cv_img /= 255

            cam_image = show_cam_on_image(cv_img, grayscale_cam, use_rgb=True)
            # save img to file
            p = Path(out_file)
            img_foler = os.path.join(p.parent, 'grad_cam')
            if not os.path.exists(img_foler):
                os.makedirs(img_foler)

            new_filepath = os.path.join(img_foler, p.stem + f'_grad_cam.jpg')
            # mmcv.imwrite(feat_img, new_filepath)
            vis_utils.plt_save(cam_image, new_filepath)

            #
            #
            # feature_hook = FeatureExtractor(model, layers=["module.detector.backbone"])
            # results = feature_hook(data)
            #
            # # save backbone features visualization
            # all_feats = results['module.detector.backbone']
            # for i in range(len(all_feats)):
            #     feat = all_feats[i][0]
            #     feat_img = vis_utils.feature2im(feat)
            #     # vis_utils.plt_show(feat_img)
            #
            #     # save img to file
            #     p = Path(out_file)
            #     img_foler = os.path.join(p.parent, 'backbone_feat')
            #     if not os.path.exists(img_foler):
            #         os.makedirs(img_foler)
            #
            #     new_filepath = os.path.join(img_foler, p.stem + f'_backbone_feat_{i}.jpg')
            #     pth_filepath = os.path.join(img_foler, p.stem + f'_backbone_feat_{i}.pth')
            #     # mmcv.imwrite(feat_img, new_filepath)
            #     vis_utils.plt_save(feat_img, new_filepath)
            #     torch.save(feat, pth_filepath)
            #
            # result = model(return_loss=False, rescale=True, **data)

        continue

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        # results = collect_results_gpu(results, len(dataset))
        raise NotImplementedError
    else:
        # results = collect_results_cpu(results, len(dataset), tmpdir)
        results = collect_results_cpu(results, tmpdir)
    return results


def collect_results_cpu(result_part, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            # part_list.append(mmcv.load(part_file))
            part_list.extend(mmcv.load(part_file))
        # # sort the results
        # ordered_results = []
        # for res in zip(*part_list):
        #     ordered_results.extend(list(res))
        # # the dataloader may pad some samples
        # ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        # return ordered_results
        return part_list


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
