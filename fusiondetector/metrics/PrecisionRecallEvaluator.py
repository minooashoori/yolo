import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from BoundingBox import *
from BoundingBoxes import *
from utils import *
from utils import BBFormat, BBType
from Evaluator import Evaluator




class PrecisionRecallEvaluator(Evaluator):
    
    
    
    def _prepareData(self, boundingboxes):
        groundtruths = []
        detections = []
        classes = []
        # Loop through all bounding boxes and separate them into gts and detections
        for bb in boundingboxes.getBoundingBoxes():
            # [imageName, class, confidence, (bb coordinates XYX2Y2)]
            if bb.getBBType() == BBType.GroundTruth:
                groundtruths.append([bb.getImageName(), bb.getClassId(), 1, bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)])
            else:
                detections.append([bb.getImageName(), bb.getClassId(), bb.getConfidence(), bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)])
            # get class
            if bb.getClassId() not in classes:
                classes.append(bb.getClassId())
        classes = sorted(classes)
        return groundtruths, detections, classes

    def getPrecisionRecall(self,
                     conf,
                     boundingboxes,
                     IOUThreshold=0.5,
    ):

        """
        Returns the precision of the bounding boxes for a given confidence and IOU threshold.

        args:
            conf: float
                confidence threshold
            boundingboxes: BoundingBoxes
                bounding boxes to evaluate, of the class BoundingBoxes representing the ground truth and detections
            IOUThreshold: float
                IOU threshold
        return:
            float
                precision
        """
        ret = []
        groundTruths = []
        detections = []
        classes = []
        # Loop through all bounding boxes and separate them into GTs and detections
        for bb in boundingboxes.getBoundingBoxes():
            # [imageName, class, confidence, (bb coordinates XYX2Y2)]
            if bb.getBBType() == BBType.GroundTruth:
                groundTruths.append([
                    bb.getImageName(),
                    bb.getClassId(), 1,
                    bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
                ])
            else:
                detections.append([
                    bb.getImageName(),
                    bb.getClassId(),
                    bb.getConfidence(),
                    bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
                ])
            # get class
            if bb.getClassId() not in classes:
                classes.append(bb.getClassId())
        classes = sorted(classes)
        
        # we will now compute the metrics for each class
        for c in classes:
            dects = []
            [dects.append(d) for d in detections if d[1] == c]
            gts = {}
            npos = 0
            for g in groundTruths:
                if g[1] == c:
                    npos += 1
                    # g[0] is the image name, if not found set it to an empty list and append g, where g is the ground truth
                    gts[g[0]] = gts.get(g[0], []) + [g]
            # sort detections by decreasing confidence
            dects = sorted(dects, key=lambda conf: conf[2], reverse=True)
            TP = np.zeros(len(dects))
            FP = np.zeros(len(dects))
            det = {key: np.zeros(len(gts[key])) for key in gts} # create a dictionary with the image name as key and an array of zeros of length len(gts[key]) as value
            # compute precision/recall
            for d in range(len(dects)):
                
                # if the confidence is lower than the threshold, then we assume there are no more detections
                if dects[d][2] < conf:
                    break
                
                # get the corresponding image ground truth
                gt  = gts.get(dects[d][0], [])
                # if there are no ground truths, then this is a false positive
                if len(gt) == 0:
                    FP[d] = 1
                    continue
                
                # find which ground truth bb matches this detection the best
                iouMax = sys.float_info.min
                for j in range(len(gt)):
                    det_box = dects[d][3]
                    gt_box = gt[j][3]
                    iou = self.iou(det_box, gt_box)
                    if iou > iouMax:
                        iouMax = iou
                        jmax = j
                
                # assign detection as true positive/don't care/false positive
                if iouMax >= IOUThreshold:
                    image_name = dects[d][0]
                    if det[image_name][jmax] == 0: # if the detection is not already assigned to a ground truth
                        det[image_name][jmax] = 1 # assign it to the ground truth
                        TP[d] = 1 # this is a true positive
                    else:
                        FP[d] = 1
                else: # iouMax < IOUThreshold then it is a false positive
                    FP[d] = 1
            # compute precision/recall
            eps = sys.float_info.epsilon # smallest representable number
            FPs = np.sum(FP)
            TPs = np.sum(TP)
            # print(TPs, FPs, npos)
            recall = TPs / (npos + eps)
            precision = TPs / (TPs + FPs + eps)
            resp = {'class': c, 'precision': round(precision, 2), 'recall': round(recall,2)}
            ret.append(resp)
        return ret


# # test out the class
# if __name__ == '__main__':
