import matplotlib.pyplot as plt
from pylab import genfromtxt
import sys
import argparse
import numpy as np

"""
Read file from command line argument.
Expected format: tab separated columns
x   f(x)    g(x)    h(x)
"""

COLORS = ["b", "r", "y", "k"]

def plot():
    parser = argparse.ArgumentParser(description="Make Nice Plots")

    parser.add_argument ("--mode", default = None, required = True,
                         help = "Which plots iou, loss, custom")
    parser.add_argument ("--title", default = None,
                         help = "title")
    parser.add_argument ("--file", default = None, required = True,
                         help = "Input plot")
    parser.add_argument ("--show", action="store_true",
                         help = "Shows plot instead of saving it")
    parser.add_argument ("--cols", default = None, nargs="*")

    args = parser.parse_args ()
    file_name = args.file
    cols = []
    y_label = None
    if args.mode == "iou":
        cols = ["Epochs", "Train iou", "Val iou"]
        y_label = "IoU"
    elif args.mode == "loss":
        cols = ["Epochs", "Train Loss", "Val Loss"]
        y_label = "Loss"
    elif args.mode == "custom":
        cols = args.cols

    labels = genfromtxt(file_name, delimiter="\t", max_rows = 1, dtype=str)
    data = genfromtxt(file_name, delimiter="\t", skip_header = 1)
    cols = np.array (cols)

    #clean labels and columns
    
    for names in [cols, labels]:
        np.place (names, names=="Val iou", "Val IoU")
        np.place (names, names=="Train iou", "Train IoU")
        
    
    for i in range (1, len (cols)):
        y_idx = np.argwhere (labels == cols [i])[0][0]
        x_idx = np.argwhere (labels == cols [0])[0][0]
        plt.plot(data[:, x_idx], data[:, y_idx], label=labels[y_idx], linewidth=4.0)
    plt.legend (loc=0)
    plt.xlabel(labels[x_idx],fontsize=16)
    plt.ylabel(y_label,  fontsize=16)
    if args.title:
        plt.suptitle (args.title,fontsize=20)
    if args.show:
        plt.show()

    save_name = file_name.split("/")[-1].split(".")[0] +"_" + args.mode + ".png"
    plt.savefig(save_name)

if __name__ == "__main__":
    plot ()
