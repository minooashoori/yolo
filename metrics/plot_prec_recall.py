import argparse
import os
import sys
import glob
import matplotlib.pyplot as plt

args = argparse.ArgumentParser()
args.add_argument("--prod", type=str, default="/home/ec2-user/dev/ctx-logoface-detector/metrics/saved_results/results_logos_prod.txt")
args.add_argument("--dev", type=str, default="/home/ec2-user/dev/ctx-logoface-detector/metrics/saved_results/results_logos_yolom.txt")

args = args.parse_args()



def get_metric(metricstr, name="Precision: "):
    metricstr = metricstr.replace(name, "")
    metricstr = metricstr.replace("[", "")
    metricstr = metricstr.replace("]", "")
    metricstr = metricstr.replace("'", "")
    metricstr = metricstr.split(", ")
    metric = [float(x) for x in metricstr]
    return metric

def get_data(filename):
    # data looks like this:
    # Class: 1
    # AP: 63.94%
    # Precision: ['1.00', '1.00', '1.00']
    with open(filename, "r") as f:
        data = f.readlines()
    # split by \n
    data = [x.strip() for x in data]
    print(data[6:8])
    # get the ap
    ap_str = data[7]
    # remove the AP: and the % sign
    ap = float(ap_str.replace("AP: ", "").replace("%", ""))

    precisions = []
    # get precisions
    prec_str = data[8] # this is a string Precision: ['1.00', '1.00', '1.00'] I need only the numbers in  a list
    prec = get_metric(prec_str)
    recall_str = data[9]
    recall = get_metric(recall_str, name="Recall: ")

    return ap, prec, recall

# get data
prod_ap, prod_prec, prod_recall = get_data(args.prod)
dev_ap, dev_prec, dev_recall = get_data(args.dev)

# plot
plt.plot(prod_recall, prod_prec, label=f"prod - yolov5 - ap: {round(prod_ap, 2)}")
plt.plot(dev_recall, dev_prec, label=f"dev - yolov8 - ap: {round(dev_ap, 2)}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.savefig("precision_recall.png")
plt.show()

# new plot
plt.clf()
# plot precision
plt.plot(prod_prec, label=f"prod - yolov5 - ap: {round(prod_ap, 2)}")
plt.plot(dev_prec, label=f"dev - yolov8 - ap: {round(dev_ap, 2)}")
plt.xlabel("detections")
plt.ylabel("Precision")
plt.legend()
plt.savefig("precision.png")
plt.show()

# new plot
plt.clf()
# plot recall
plt.plot(prod_recall, label=f"prod - yolov5 - ap: {round(prod_ap, 2)}")
plt.plot(dev_recall, label=f"dev - yolov8 - ap: {round(dev_ap, 2)}")
plt.xlabel("detections")
plt.ylabel("Recall")
plt.legend()
plt.savefig("recall.png")
plt.show()



