import argparse
import os
import yaml
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def find_confidence(acc_metric, confs, m_value):
    inter_f =  interp1d(acc_metric, confs, kind='linear')
    conf_ = inter_f(m_value)

    return conf_

def get_data(filename):
    with open(filename, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def get_prec_recall_ap(data):
    metrics = {}
    for cl, metric in data.items():
        if cl != 'mAP': # we have a class so we will need to get the precision and recall
            precision  = metric["precision"]
            recall = metric["recall"]
            conf  = metric["confs"]
            ap = metric["ap"]
            metrics[cl] = {"precision": precision, "recall": recall, "confs": conf, "ap": ap}

    return metrics

def plot_prec_recall(metrics, type_: str):
    # loop over the classes
    for cl, metric in metrics.items():
        precision = metric["precision"]
        recall = metric["recall"]
        ap = metric["ap"]
        plt.plot(recall, precision, label=f"{type_} - class: {cl} - ap: {round(ap, 2)}")


def plot_prec_or_recall_conf(metrics, m: str, type_: str, proposed_confidence_th=None):
    for cl, metric in metrics.items():
        m_ = metric[m]
        ap = metric["ap"]
        conf = metric["confs"]
        # plt.plot(conf, m_, label=f" {type_} - class: {cl} - ap: {round(ap, 2)}")
        plt.step(conf, m_, where='mid', label=f" {type_} - class: {cl} - ap: {round(ap, 2)}")
        if proposed_confidence_th:
            # get the proposed confidence
            proposed_conf = proposed_confidence_th[cl][m]["conf"]
            # get the target value
            target_value = proposed_confidence_th[cl][m]["target"]
            # plot the proposed confidence
            plt.axvline(x=proposed_conf, color="lightgray", linestyle="--")
            # plot the target value
            plt.axhline(y=target_value, color="lightgray", linestyle="--", label=f"conf: {proposed_conf} - target {m}: {target_value}")


def plot_prec_or_recall_conf_all(prod_data, dev_data, m: str, proposed_confidence_th, save_dir, model, c):
    # start with precision
    prod_metrics = get_prec_recall_ap(prod_data)
    dev_metrics = get_prec_recall_ap(dev_data)
    # close all plots
    plt.close('all')
    plot_prec_or_recall_conf(prod_metrics, m ,"prod")
    plot_prec_or_recall_conf(dev_metrics, m , model, proposed_confidence_th)

    plt.xlabel("confidence")
    #invert x axis if we are plotting precision
    if m == "precision":
        plt.gca().invert_xaxis()
    plt.ylabel(m)
    # add mAP for prod and dev on the top of the plot
    plt.title(f"mAP: prod: {round(prod_data['mAP'], 2)} - {model}: {round(dev_data['mAP'], 2)}")
    plt.legend()
    plt.grid()
    # add saved dir
    plt.savefig(os.path.join(save_dir, f"{m}_conf_{c}.png"))
    plt.show()
    plt.clf()


def plot_prec_recall_all(prod_data, dev_data, save_dir, model:str, c):
    prod_metrics = get_prec_recall_ap(prod_data)
    dev_metrics = get_prec_recall_ap(dev_data)
    # close all plots
    plt.close('all')
    plot_prec_recall(prod_metrics, "prod")
    plot_prec_recall(dev_metrics, model)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"mAP: prod: {round(prod_data['mAP'], 2)} - {model}: {round(dev_data['mAP'], 2)}")
    plt.legend()
    plt.grid()
    #add saved dir
    plt.savefig(os.path.join(save_dir, f"prec_recall_{c}.png"))
    plt.show()
    plt.clf()


def get_confidence_threshold(prod_data, dev_data):

    proposed_confs = {}
    # get the mininum precision of prod
    prod_metrics = get_prec_recall_ap(prod_data)
    dev_metrics = get_prec_recall_ap(dev_data)
    # for each class - get the conf for the min precision of prod and the conf for the max recall of prod
    for cl, metric in prod_metrics.items():
        precision_prod = metric["precision"]
        recall_prod = metric["recall"]

        # get the targets from prod
        prec_target_value = min(precision_prod)
        recall_target_value = max(recall_prod)

        # get the precision for the target precision
        precision_dev = dev_metrics[cl]["precision"]
        recall_dev = dev_metrics[cl]["recall"]
        confs_dev = dev_metrics[cl]["confs"]

        # get the confidence for the target precision
        conf_prec_target = find_confidence(precision_dev, confs_dev, prec_target_value)
        # same for recall
        conf_recall_target = find_confidence(recall_dev, confs_dev, recall_target_value)
        # round evrything to 2 decimals
        prec_target_value = round(prec_target_value, 2)
        recall_target_value = round(recall_target_value, 2)
        conf_prec_target = round(float(conf_prec_target), 2)
        conf_recall_target = round(float(conf_recall_target), 2)

        proposed_confs[cl] = {"precision": {"target": prec_target_value, "conf": conf_prec_target}, "recall": {"target": recall_target_value, "conf": conf_recall_target}}
    return proposed_confs

def main(args):
    data_prod = get_data(args.prod)
    data_dev = get_data(args.dev)
    proposed_confidence_th = get_confidence_threshold(data_prod, data_dev)
    plot_prec_or_recall_conf_all(data_prod, data_dev, "precision", proposed_confidence_th, args.save_dir, args.model, args.c)
    plot_prec_or_recall_conf_all(data_prod, data_dev, "recall", proposed_confidence_th, args.save_dir, args.model, args.c)
    plot_prec_recall_all(data_prod, data_dev, args.save_dir, args.model, args.c)
    print(proposed_confidence_th)
    # save the proposed confidences
    with open(os.path.join(args.save_dir, f"proposed_confs_{args.c}.yaml"), "w") as f:
        yaml.dump(proposed_confidence_th, f)


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--prod", type=str, default="/home/ec2-user/dev/ctx-logoface-detector/metrics/save/prod_face.yaml")
    args.add_argument("--dev", type=str, default="/home/ec2-user/dev/ctx-logoface-detector/metrics/save/yolov8s_face_2.yaml")
    args.add_argument("--save_dir", type=str, default="/home/ec2-user/dev/ctx-logoface-detector/metrics/plots")
    args.add_argument("--c",  type=str, default="face")
    args.add_argument("--model",  type=str, default="yolov8s")

    args = args.parse_args()


    main(args)

