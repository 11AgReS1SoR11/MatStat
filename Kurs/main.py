#https://matplotlib.org/tutorials/introductory/images.html#sphx-glr-tutorials-introductory-images-py
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pylab
from matplotlib import mlab, transforms
import math
from matplotlib.patches import Ellipse
fileName = 'KLM 5 reflection D=1.binary.bgData'

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def Sum(matrix):
    sum_matrix = 0
    for row in matrix:
        for element in row:
            sum_matrix += element
    return sum_matrix

def fooo(data):
    X, Y = 0, 0
    for i, row in enumerate(data):
        for j, elem in enumerate(row):
            X += elem * j
    X = X / Sum(data)
    for i, row in enumerate(data):
        for j, elem in enumerate(row):
            Y += elem * i
    Y = Y / Sum(data)
    Sx, Sy = 0, 0
    for i, row in enumerate(data):
        for j, elem in enumerate(row):
            Sx += (elem * j - X) ** 2
    Sx = Sx / Sum(data)
    for i, row in enumerate(data):
        for j, elem in enumerate(row):
            Sy += (elem * i - Y) ** 2
    Sy = Sy / Sum(data)

    return X, Y, Sx**0.5, Sy**0.5

def plot_points_with_ellipses(data, ax):

    Xmean, Ymean, Sx, Sy = fooo(data)

    cov = 0
    for i, row in enumerate(data):
        for j, elem in enumerate(row):
            cov += (elem * i - Ymean) * (elem * j - Xmean)    
    cov = cov / Sum(data)

    r = cov / (Sx * Sy)
    ell_radius_x = (Sx / Sy * 150)
    ell_radius_y = 150 #* 2

    theta = np.degrees(math.atan(2 * r * Sx * Sy / (Sx ** 2 - Sy ** 2)) / 2)

    print("Sx", Sx)
    print("Sy ", Sy)
    print("cov ", cov)
    print("Xmean ", Xmean)
    print("Ymean ", Ymean)
    print("Theta ", theta)

    ellipse = Ellipse(
        (Xmean * 1.2, Ymean * 1.2),
        width=ell_radius_x,
        height=ell_radius_y,
        angle=-theta,
        color='black',
        alpha=0.5
    )

    ax.add_patch(ellipse)
    return ax

def traverse_datasets(hdf_file):
    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset): #for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): #for group (go down)
                yield from h5py_dataset_iterator(item, path)

    with h5py.File(hdf_file, 'r') as f:
        for path, _ in h5py_dataset_iterator(f):
            yield path

#сначала узнаю имена подгрупп, в которых лежат данные
with h5py.File(fileName, 'r') as f:
    for dset in traverse_datasets(fileName):
        print('Path:', dset)
        print('Shape:', f[dset])
        print('Data type:', f[dset].dtype)
    dataset = f["/BG_DATA/1/DATA"]
    data = np.reshape(dataset, (1920, 1000), 'F')
    fig, ax = plt.subplots()
    pylab.plot(data)
    imgplot = ax.imshow(data[200:500, :], origin='lower')
    imgplot.set_cmap('nipy_spectral')
    colorbar = plt.colorbar(imgplot, orientation='horizontal')
    plot_points_with_ellipses(data[200:500, :], ax)
    plt.savefig("aaa")
    plt.show()
    plt.close()
