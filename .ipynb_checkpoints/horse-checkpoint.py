# scale an image of a game to see how easily one could just use color thresholds to mark walls.\
#verdict: it's not so easy to tell. Let's write the problem in a spreadsheet.

import networkx as nx
import pulp as pl
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from matplotlib.pyplot import imshow, subplots, show
import matplotlib.pyplot as plt
from os import path, listdir, getcwd, chdir
from matplotlib.animation import FuncAnimation

pix_folder = './pix/'
pixnames = listdir(pix_folder)

pix = [imread(path.join(pix_folder, pic)) for pic in pixnames]

expic = pix[1]

frames = [
    resize(expic, (expic.shape[0]//divider, expic.shape[1]//divider))
    for divider in range(15,17*2+1)
]

fig, ax = subplots()
im = ax.imshow(frames[0], vmin=0, vmax=1.)

def update(i):
    ax.set_xlabel(f"frame {i}\ndiv by {i+15}")
    im.set_data(frames[i])
    return (im, ax)

ani = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=500,
    blit = False
)

plt.show()