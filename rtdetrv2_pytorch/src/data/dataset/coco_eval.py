"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib

# MiXaiLL76 replacing pycocotools with faster-coco-eval for better performance and support.
"""

import os
import contextlib
import copy
import numpy as np
import torch

from faster_coco_eval import COCO, COCOeval_faster
import faster_coco_eval.core.mask as mask_util
from ...core import register
from ...misc import dist_utils
__all__ = ['CocoEvaluator',]


@register()
class CocoEvaluator(object):
    def __init__(self, coco_gt, iou_types, include_iou_10=True):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt : COCO = coco_gt
        self.iou_types = iou_types
        self.include_iou_10 = include_iou_10

        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(coco_gt, iouType=iou_type, print_function=print, separate_eval=True)

        # Create separate evaluator for IoU@0.1 if requested
        if self.include_iou_10:
            self.coco_eval_iou10 = {}
            for iou_type in iou_types:
                eval_instance = COCOeval_faster(coco_gt, iouType=iou_type, print_function=print, separate_eval=True)
                # Set IoU threshold to 0.1
                eval_instance.params.iouThrs = np.array([0.1])
                self.coco_eval_iou10[iou_type] = eval_instance

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        if self.include_iou_10:
            self.eval_imgs_iou10 = {k: [] for k in iou_types}

    def cleanup(self):
        self.coco_eval = {}
        for iou_type in self.iou_types:
            self.coco_eval[iou_type] = COCOeval_faster(self.coco_gt, iouType=iou_type, print_function=print, separate_eval=True)
        
        if self.include_iou_10:
            self.coco_eval_iou10 = {}
            for iou_type in self.iou_types:
                eval_instance = COCOeval_faster(self.coco_gt, iouType=iou_type, print_function=print, separate_eval=True)
                eval_instance.params.iouThrs = np.array([0.1])
                self.coco_eval_iou10[iou_type] = eval_instance
                
        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}
        if self.include_iou_10:
            self.eval_imgs_iou10 = {k: [] for k in self.iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            
            # Standard COCO evaluation
            coco_eval = self.coco_eval[iou_type]
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = self.coco_gt.loadRes(results) if results else COCO()
                    coco_eval.cocoDt = coco_dt
                    coco_eval.params.imgIds = list(img_ids)
                    coco_eval.evaluate()

            self.eval_imgs[iou_type].append(np.array(coco_eval._evalImgs_cpp).reshape(len(coco_eval.params.catIds), len(coco_eval.params.areaRng), len(coco_eval.params.imgIds)))

            # IoU@0.1 evaluation
            if self.include_iou_10:
                coco_eval_iou10 = self.coco_eval_iou10[iou_type]
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        coco_dt_iou10 = self.coco_gt.loadRes(results) if results else COCO()
                        coco_eval_iou10.cocoDt = coco_dt_iou10
                        coco_eval_iou10.params.imgIds = list(img_ids)
                        coco_eval_iou10.evaluate()

                self.eval_imgs_iou10[iou_type].append(np.array(coco_eval_iou10._evalImgs_cpp).reshape(len(coco_eval_iou10.params.catIds), len(coco_eval_iou10.params.areaRng), len(coco_eval_iou10.params.imgIds)))

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            # Standard evaluation
            img_ids, eval_imgs = merge(self.img_ids, self.eval_imgs[iou_type])
            coco_eval = self.coco_eval[iou_type]
            coco_eval.params.imgIds = img_ids
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
            coco_eval._evalImgs_cpp = eval_imgs

            # IoU@0.1 evaluation
            if self.include_iou_10:
                _, eval_imgs_iou10 = merge(self.img_ids, self.eval_imgs_iou10[iou_type])
                coco_eval_iou10 = self.coco_eval_iou10[iou_type]
                coco_eval_iou10.params.imgIds = img_ids
                coco_eval_iou10._paramsEval = copy.deepcopy(coco_eval_iou10.params)
                coco_eval_iou10._evalImgs_cpp = eval_imgs_iou10

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()
        
        if self.include_iou_10:
            for coco_eval_iou10 in self.coco_eval_iou10.values():
                coco_eval_iou10.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            
            # Standard COCO summarize
            coco_eval.summarize()
            
            # Add custom AP@0.1 evaluation if requested
            if self.include_iou_10:
                self._summarize_ap_at_iou_10(iou_type)

    def _summarize_ap_at_iou_10(self, iou_type):
        """
        Add AP@0.1 and AR@0.1 evaluation using the dedicated IoU@0.1 evaluator
        """
        coco_eval_iou10 = self.coco_eval_iou10[iou_type]
        
        def _summarize_single_iou(ap=1, areaRng='all', maxDets=100):
            p = coco_eval_iou10.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '0.10'

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            
            if ap == 1:
                # dimension of precision: [TxRxKxAxM] - for IoU@0.1, T=1
                s = coco_eval_iou10.eval['precision']
                s = s[0, :, :, aind, mind]  # Take the first (and only) IoU threshold
            else:
                # dimension of recall: [TxKxAxM] - for IoU@0.1, T=1
                s = coco_eval_iou10.eval['recall']
                s = s[0, :, aind, mind]  # Take the first (and only) IoU threshold
            
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        # Compute and store AP@0.1 and AR@0.1 statistics
        print(" Custom IoU@0.1 evaluation:")
        
        # Initialize stats list to store the computed metrics
        stats = []
        
        # AP metrics (similar to standard COCO: AP, AP50, AP75, APs, APm, APl)
        # For IoU@0.1: AP@0.1, APs@0.1, APm@0.1, APl@0.1
        stats.append(_summarize_single_iou(1, areaRng='all', maxDets=coco_eval_iou10.params.maxDets[2]))    # AP@0.1
        stats.append(_summarize_single_iou(1, areaRng='small', maxDets=coco_eval_iou10.params.maxDets[2]))  # APs@0.1
        stats.append(_summarize_single_iou(1, areaRng='medium', maxDets=coco_eval_iou10.params.maxDets[2])) # APm@0.1
        stats.append(_summarize_single_iou(1, areaRng='large', maxDets=coco_eval_iou10.params.maxDets[2]))  # APl@0.1
        
        # AR metrics (similar to standard COCO: AR1, AR10, AR100, ARs, ARm, ARl)
        # For IoU@0.1: AR1@0.1, AR10@0.1, AR100@0.1
        stats.append(_summarize_single_iou(0, areaRng='all', maxDets=coco_eval_iou10.params.maxDets[0]))    # AR1@0.1
        stats.append(_summarize_single_iou(0, areaRng='all', maxDets=coco_eval_iou10.params.maxDets[1]))    # AR10@0.1
        stats.append(_summarize_single_iou(0, areaRng='all', maxDets=coco_eval_iou10.params.maxDets[2]))    # AR100@0.1
        
        # Store the computed stats in the evaluator for logging
        coco_eval_iou10.stats = stats

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        'keypoints': keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def merge(img_ids, eval_imgs):
    all_img_ids = dist_utils.all_gather(img_ids)
    all_eval_imgs = dist_utils.all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.extend(p)


    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, axis=2).ravel()
    # merged_eval_imgs = np.array(merged_eval_imgs).T.ravel()

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)

    return merged_img_ids.tolist(), merged_eval_imgs.tolist()