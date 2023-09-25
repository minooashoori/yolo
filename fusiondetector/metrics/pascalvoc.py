
import argparse
import glob
import os
import shutil
import sys

from fusiondetector.metrics.BoundingBox import BoundingBox
from fusiondetector.metrics.BoundingBoxes import BoundingBoxes
from fusiondetector.metrics.Evaluator import *
from fusiondetector.metrics.utils import BBFormat
from fusiondetector.metrics.PrecisionRecallEvaluator import PrecisionRecallEvaluator
import yaml

# Get current path to set default folders
currentPath = os.path.dirname(os.path.abspath(__file__))

VERSION = '0.2 (beta)'

parser = argparse.ArgumentParser(
    prog='Object Detection Metrics - Pascal VOC',
    description=
    f'This project applies the most popular metrics used to evaluate object detection '
    'further reference: please check:\nhttps://github.com/rafaelpadilla/Object-Detection-Metrics')
parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + VERSION)
# Positional arguments
# Mandatory
parser.add_argument('-gt',
                    '--gtfolder',
                    dest='gtFolder',
                    default=os.path.join(currentPath, 'groundtruths'),
                    metavar='',
                    help='folder containing your ground truth bounding boxes')
parser.add_argument('-det',
                    '--detfolder',
                    dest='detFolder',
                    default=os.path.join(currentPath, 'detections'),
                    metavar='',
                    help='folder containing your detected bounding boxes')
# Optional
parser.add_argument('-t',
                    '--threshold',
                    dest='iouThreshold',
                    type=float,
                    default=0.45,
                    metavar='',
                    help='IOU threshold. Default 0.45')
parser.add_argument('-gtformat',
                    dest='gtFormat',
                    metavar='',
                    default='xywh',
                    help='format of the coordinates of the ground truth bounding boxes: '
                    '(\'xywh\': <left> <top> <width> <height>)'
                    ' or (\'xyrb\': <left> <top> <right> <bottom>)')
parser.add_argument('-detformat',
                    dest='detFormat',
                    metavar='',
                    default='xywh',
                    help='format of the coordinates of the detected bounding boxes '
                    '(\'xywh\': <left> <top> <width> <height>) '
                    'or (\'xyrb\': <left> <top> <right> <bottom>)')
parser.add_argument('-gtcoords',
                    dest='gtCoordinates',
                    default='abs',
                    metavar='',
                    help='reference of the ground truth bounding box coordinates: absolute '
                    'values (\'abs\') or relative to its image size (\'rel\')')
parser.add_argument('-detcoords',
                    default='abs',
                    dest='detCoordinates',
                    metavar='',
                    help='reference of the ground truth bounding box coordinates: '
                    'absolute values (\'abs\') or relative to its image size (\'rel\')')
parser.add_argument('-imgsize',
                    dest='imgSize',
                    metavar='',
                    help='image size. Required if -gtcoords or -detcoords are \'rel\'')
parser.add_argument('-sp',
                    '--savepath',
                    dest='savePath',
                    metavar='',
                    help='folder where the plots are saved')
parser.add_argument('-np',
                    '--noplot',
                    dest='showPlot',
                    action='store_false',
                    help='no plot is shown during execution')
parser.add_argument('-conf',
                    '--confidence',
                    dest='confidence',
                    type=float,
                    default=0.5)

args = parser.parse_args()




# Validate formats
def ValidateFormats(argFormat, argName, errors):
    if argFormat == 'xywh':
        return BBFormat.XYWH
    elif argFormat == 'xyrb':
        return BBFormat.XYX2Y2
    elif argFormat is None:
        return BBFormat.XYWH  # default when nothing is passed
    else:
        errors.append('argument %s: invalid value. It must be either \'xywh\' or \'xyrb\'' %
                    argName)


# Validate mandatory args
def ValidateMandatoryArgs(arg, argName, errors):
    if arg is None:
        errors.append('argument %s: required argument' % argName)
    else:
        return True


def ValidateImageSize(arg, argName, argInformed, errors):
    errorMsg = 'argument %s: required argument if %s is relative' % (argName, argInformed)
    ret = None
    if arg is None:
        errors.append(errorMsg)
    else:
        arg = arg.replace('(', '').replace(')', '')
        args = arg.split(',')
        if len(args) != 2:
            errors.append('%s. It must be in the format \'width,height\' (e.g. \'600,400\')' %
                        errorMsg)
        else:
            if not args[0].isdigit() or not args[1].isdigit():
                errors.append(
                    '%s. It must be in INdiaTEGER the format \'width,height\' (e.g. \'600,400\')' %
                    errorMsg)
            else:
                ret = (int(args[0]), int(args[1]))
    return ret


# Validate coordinate types
def ValidateCoordinatesTypes(arg, argName, errors):
    if arg == 'abs':
        return CoordinatesType.Absolute
    elif arg == 'rel':
        return CoordinatesType.Relative
    elif arg is None:
        return CoordinatesType.Absolute  # default when nothing is passed
    errors.append('argument %s: invalid value. It must be either \'rel\' or \'abs\'' % argName)


def ValidatePaths(arg, nameArg, errors):
    if arg is None:
        errors.append('argument %s: invalid directory' % nameArg)
    elif os.path.isdir(arg) is False and os.path.isdir(os.path.join(currentPath, arg)) is False:
        errors.append('argument %s: directory does not exist \'%s\'' % (nameArg, arg))
    # elif os.path.isdir(os.path.join(currentPath, arg)) is True:
    #     arg = os.path.join(currentPath, arg)
    else:
        arg = os.path.join(currentPath, arg)
    return arg


def getBoundingBoxes(directory,
                    isGT,
                    bbFormat,
                    coordType,
                    allBoundingBoxes=None,
                    allClasses=None,
                    imgSize=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
    # Read ground truths
    os.chdir(directory)
    files = glob.glob("*.txt")
    files.sort()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            if isGT:
                # idClass = int(splitLine[0]) #class
                idClass = (splitLine[0])  # class
                x = float(splitLine[1])
                y = float(splitLine[2])
                w = float(splitLine[3])
                h = float(splitLine[4])
                bb = BoundingBox(nameOfImage,
                                idClass,
                                x,
                                y,
                                w,
                                h,
                                coordType,
                                imgSize,
                                BBType.GroundTruth,
                                format=bbFormat)
            else:
                # idClass = int(splitLine[0]) #class
                idClass = (splitLine[0])  # class
                confidence = float(splitLine[1])
                x = float(splitLine[2])
                y = float(splitLine[3])
                w = float(splitLine[4])
                h = float(splitLine[5])
                bb = BoundingBox(nameOfImage,
                                idClass,
                                x,
                                y,
                                w,
                                h,
                                coordType,
                                imgSize,
                                BBType.Detected,
                                confidence,
                                format=bbFormat)
            allBoundingBoxes.addBoundingBox(bb)
            if idClass not in allClasses:
                allClasses.append(idClass)
        fh1.close()
    return allBoundingBoxes, allClasses


def process_args(args):
    iouThreshold = args.iouThreshold
    conf = args.confidence

    # Arguments validation
    errors = []
    # Validate formats
    gtFormat = ValidateFormats(args.gtFormat, '-gtformat', errors)
    detFormat = ValidateFormats(args.detFormat, '-detformat', errors)
    # Groundtruth folder
    if ValidateMandatoryArgs(args.gtFolder, '-gt/--gtfolder', errors):
        gtFolder = ValidatePaths(args.gtFolder, '-gt/--gtfolder', errors)
    else:
        # errors.pop()
        gtFolder = os.path.join(currentPath, 'groundtruths')
        if os.path.isdir(gtFolder) is False:
            errors.append('folder %s not found' % gtFolder)
    # Coordinates types
    gtCoordType = ValidateCoordinatesTypes(args.gtCoordinates, '-gtCoordinates', errors)
    detCoordType = ValidateCoordinatesTypes(args.detCoordinates, '-detCoordinates', errors)
    imgSize = (0, 0)
    if gtCoordType == CoordinatesType.Relative:  # Image size is required
        imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-gtCoordinates', errors)
    if detCoordType == CoordinatesType.Relative:  # Image size is required
        imgSize = ValidateImageSize(args.imgSize, '-imgsize', '-detCoordinates', errors)
    # Detection folder
    if ValidateMandatoryArgs(args.detFolder, '-det/--detfolder', errors):
        detFolder = ValidatePaths(args.detFolder, '-det/--detfolder', errors)
    else:
        # errors.pop()
        detFolder = os.path.join(currentPath, 'detections')
        if os.path.isdir(detFolder) is False:
            errors.append('folder %s not found' % detFolder)
    if args.savePath is not None:
        savePath = ValidatePaths(args.savePath, '-sp/--savepath', errors)
    else:
        savePath = os.path.join(currentPath, 'results')
    # Validate savePath
    # If error, show error messages
    if len(errors) != 0:
        print("""usage: Object Detection Metrics [-h] [-v] [-gt] [-det] [-t] [-gtformat]
                                    [-detformat] [-save]""")
        print('Object Detection Metrics: error(s): ')
        [print(e) for e in errors]
        sys.exit()

    # Check if path to save results already exists and is not empty
    if os.path.isdir(savePath) and os.listdir(savePath):
        print(f'Folder {savePath} already exists and may contain important results.\n')

    # Clear folder and save results
    shutil.rmtree(savePath, ignore_errors=True)
    os.makedirs(savePath)
    # Show plot during execution
    showPlot = args.showPlot

    return iouThreshold, conf, gtFormat, detFormat, gtFolder, gtCoordType, detCoordType, imgSize, detFolder, savePath, showPlot





def get_bounding_boxes(gtFolder, detFolder, gtFormat, detFormat, gtCoordType, detCoordType, imgSize):
    allBoundingBoxes, allClasses = getBoundingBoxes(gtFolder, True, gtFormat, gtCoordType, imgSize=imgSize)
    allBoundingBoxes, allClasses = getBoundingBoxes(detFolder, False, detFormat, detCoordType, allBoundingBoxes, allClasses, imgSize=imgSize)
    # allClasses.sort()
    return allBoundingBoxes, allClasses


def process_precision_recall_curves(evaluator, allBoundingBoxes, iouThreshold, savePath, showPlot):
    acc_AP, validClasses, results = 0, 0, {}

    for metricsPerClass in evaluator.PlotPrecisionRecallCurve(
        allBoundingBoxes,
        IOUThreshold=iouThreshold,
        method=MethodAveragePrecision.EveryPointInterpolation,
        showAP=True,
        showInterpolatedPrecision=False,
        savePath=savePath,
        showGraphic=showPlot
    ):
        cl = metricsPerClass['class']
        ap = metricsPerClass['AP']
        precision = metricsPerClass['precision']
        recall = metricsPerClass['recall']
        totalPositives = metricsPerClass['total positives']
        total_TP = metricsPerClass['total TP']
        total_FP = metricsPerClass['total FP']
        confs = metricsPerClass['confs']

        results_per_class = {
            "precision": [float(p) for p in precision],
            "recall": [float(r) for r in recall],
            "total positives": float(totalPositives),
            "total TP": float(total_TP),
            "total FP": float(total_FP),
            "confs": [float(c) for c in confs],
        }

        if totalPositives > 0:
            validClasses += 1
            acc_AP += ap
            ap_str = "{0:.2f}%".format(ap * 100)
            print('AP: %s (%s)' % (ap_str, cl))
            results_per_class["ap"] = float(ap)
            results[cl] = results_per_class

    return acc_AP, validClasses, results


def process_prec_recall_at_threshold(prec_recall_evaluator, conf, allBoundingBoxes, iouThreshold, results):

    prec_rec = prec_recall_evaluator.getPrecisionRecall(
        conf,
        allBoundingBoxes,
        iouThreshold,
    )
    print(prec_rec)
    for res in prec_rec:
        class_key = res["class"]
        results[class_key][f"prec@conf{conf}@iou{iouThreshold}"] = float(res["precision"])
        results[class_key][f"recall@conf{conf}@iou{iouThreshold}"] = float(res["recall"])
        results[class_key]["conf"] = conf
        results[class_key]["iou"] = iouThreshold
    return results



def main(args):

    iouThreshold, conf, gtFormat, detFormat, gtFolder, gtCoordType, detCoordType, imgSize, detFolder, savePath, showPlot = process_args(args)

    allBoundingBoxes, _ = get_bounding_boxes(gtFolder, detFolder, gtFormat, detFormat, gtCoordType, detCoordType, imgSize)

    evaluator = Evaluator()
    prec_recall_evaluator = PrecisionRecallEvaluator()

    acc_AP, validClasses, results = process_precision_recall_curves(evaluator, allBoundingBoxes, iouThreshold, savePath, showPlot)

    mAP = acc_AP / validClasses
    results["mAP"] = mAP
    results = process_prec_recall_at_threshold(prec_recall_evaluator, conf, allBoundingBoxes, iouThreshold, results)

    with open(os.path.join(savePath, 'results.yaml'), 'w') as f:
        yaml.dump(results, f)

if __name__ == "__main__":
    main(args)