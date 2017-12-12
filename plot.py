import matplotlib.pyplot as plt
from pylab import genfromtxt
import sys

"""
Read file from command line argument.
Expected format: tab separated columns
x   f(x)    g(x)    h(x)
"""
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Expect file name and one label per column in file')
        print("Example usage: python plot.py data.txt 'Epochs' 'Train Loss' 'Val Loss'")
    else:
        file_name = sys.argv[1]

        mat0 = genfromtxt(file_name, delimiter="\t")
        for i in range(3, len(sys.argv)):
            plt.plot(mat0[:, 0], mat0[:, i - 2], label=sys.argv[i])
        plt.legend()
        plt.xlabel(sys.argv[2])
        plt.show()
