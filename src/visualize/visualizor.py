import matplotlib.pyplot as plt
import matplotlib
from visdom import Visdom
import numpy as np
matplotlib.use('agg')

class visualizor:
    def __init__(self, port = None):
        if port is not None:
            self.viz = Visdom(port=port)
        else:
            self.viz = Visdom()
    
    def showKeypoint(self, img, xCoords, yCoords, visibility=None, saveFile = None):
        c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
        plt.imshow(img)
        plt.plot(xCoords[visibility > 0], yCoords[visibility > 0], 'o', markersize=8,
                markerfacecolor=c, markeredgecolor='k', markeredgewidth=2,)
        plt.plot(xCoords[visibility > 1], yCoords[visibility > 1], 'o', markersize=8,
                markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)
        self.viz.matplot(plt)
    
    def showDirectionalKeypoint(self, img, xJoint, yJoint, xDir, yDir):
        c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
        plt.imshow(img)
        plt.plot(xJoint, yJoint, 'o', markersize=5,
                 markerfacecolor=c, markeredgecolor=c, markeredgewidth=2,)
        plt.plot(xDir, yDir, 'o', markersize=5,
                 markerfacecolor='r', markeredgecolor='r', markeredgewidth=2,)
        self.viz.matplot(plt)

    def showHeatmap():
        pass

