#https://matplotlib.org/tutorials/introductory/images.html#sphx-glr-tutorials-introductory-images-py
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pylab
from matplotlib.patches import Ellipse
fileName = 'KLM 5 reflection D=1.binary.bgData'


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


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]


def plot_points_with_ellipses(xdata, ydata, file_name, text=None):
    fig, ax = plt.subplots()

    cov = np.cov(xdata, ydata)
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 1.5 * 2 * np.sqrt(vals)

    ell = Ellipse(xy=(np.mean(xdata), np.mean(ydata)),
            width=w, height=h,
            angle=theta, color='black', alpha=0.2)

    ax.add_artist(ell)
    ax.scatter(xdata, ydata, c='blue', lw = 0, alpha=0.7, s=95)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Plot of Generated Points with Ellipses of Dispersion')
    ax.grid(True)

    if text is not None:
        plt.text(0.5, -0.1, text, ha='center', fontsize=12)  # Добавляем текст ниже графика

    plt.savefig(file_name)  # Сохраняем график в файл
    plt.close()  # Закрываем текущий график, чтобы он не отображался в блокноте


def find_slice_indices(data, left, right):
    x_indices = []
    y_indices = []

    for i, row in enumerate(data):
        for j, value in enumerate(row):
            if left <= value <= right:
                x_indices.append(j)
                y_indices.append(i)

    return x_indices, y_indices


def getMax(data):
    m = -1
    for i in data:
        m = max(m, max(i))
    return m


#сначала узнаю имена подгрупп, в которых лежат данные
with h5py.File(fileName, 'r') as f:
    for dset in traverse_datasets(fileName):
        print('Path:', dset)
        print('Shape:', f[dset])
        print('Data type:', f[dset].dtype)
    dataset = f["/BG_DATA/1/DATA"]
    data = np.reshape(dataset, (1920, 1000), 'F')

    left_bound = float(input("Введите левую границу: "))
    right_bound = float(input("Введите правую границу: "))
    left_bound *= getMax(data)
    right_bound *= getMax(data)
    x_indices, y_indices = find_slice_indices(data[200:500, 400:900], left_bound, right_bound)
    plot_points_with_ellipses(x_indices, y_indices, "picture")

