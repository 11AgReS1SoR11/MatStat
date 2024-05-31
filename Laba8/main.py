import matplotlib.pyplot as plt
import numpy as np
import math

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]


def Sum(matrix, low, upp):
    sum_matrix = 0
    for row in matrix:
        for element in row:
            if (low <= element <= upp):
                sum_matrix += element
    return sum_matrix


def calculate_variance(data):
    n = len(data)
    if n < 2:
        raise ValueError("Для вычисления дисперсии требуется как минимум два значения.")

    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    return variance


def fooo(data, low, upp):
    X, Y = 0, 0
    for i, row in enumerate(data):
        for j, elem in enumerate(row):
            if (low <= elem <= upp):
                X += elem * j
    X = X / Sum(data, low, upp)
    for i, row in enumerate(data):
        for j, elem in enumerate(row):
            if (low <= elem <= upp):
                Y += elem * i
    Y = Y / Sum(data, low, upp)
    Sx, Sy = 0, 0
    for i, row in enumerate(data):
        for j, elem in enumerate(row):
            if (low <= elem <= upp):
                Sx += elem * (j - X) ** 2
    Sx = Sx / Sum(data, low, upp)
    for i, row in enumerate(data):
        for j, elem in enumerate(row):
            if (low <= elem <= upp):
                Sy += elem * (i - Y) ** 2
    Sy = Sy / Sum(data, low, upp)

    return X, Y, Sx**0.5, Sy**0.5
    

def draw_sep(x, y, ellipse):
    fig, ax = plt.subplots()  # Создаем объекты Figure и Axes
    ax.scatter(x, y, color='blue', label='Points')  # Рисуем точки
    ax.add_patch(ellipse)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    plt.savefig("ccc")
    plt.close()


def plot_points_with_ellipses_Maxim(data, low, upp, ax, x, y):

    Xmean, Ymean, Sx, Sy = fooo(data, low, upp)

    cov = 0
    for i, row in enumerate(data):
        for j, elem in enumerate(row):
            if (low <= elem <= upp):
                cov += elem * (i - Ymean) * (j - Xmean)
    cov = cov / Sum(data, low, upp)

    r = cov / (Sx * Sy)
    ell_radius_x = (-4*(1 - r*r)*math.log((upp + low) / getMax(data) / 2))*Sx #(Sx / Sy * 250)
    ell_radius_y = (-4*(1 - r*r)*math.log((upp + low) / getMax(data) / 2))*Sy #250
    print(f"x = {ell_radius_x}, y = {ell_radius_y}, r = {r}, log = {math.log((upp + low) / getMax(data) / 2)}")

    theta = np.degrees(math.atan(2 * r * Sx * Sy / (Sx ** 2 - Sy ** 2)) / 2)

    print("Sx", Sx)
    print("Sy ", Sy)
    print("cov ", cov)
    print("Xmean ", Xmean)
    print("Ymean ", Ymean)
    print("Theta ", theta)

    ellipse = Ellipse(
        (Xmean, Ymean),
        width=ell_radius_x,
        height=ell_radius_y,
        angle=theta,
        color='black',
        alpha=0.5
    )
    draw_sep(x, y, ellipse)
    ax.add_patch(ellipse)
    return ax

def covariance(x, y):
    if len(x) != len(y):
        raise ValueError("Arrays x and y must have the same length")
    
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / (n - 1)
    
    return cov


def plot_points_with_ellipses(x, y, ax, low = None, upp = None):

    Xmean = np.mean(x)
    Ymean = np.mean(y)
    Sx = calculate_variance(x)
    Sy = calculate_variance(y)
    cov = covariance(x, y)

    r = cov / (Sx * Sy)
    ell_radius_x = (-4*(1 - r*r)*math.log((upp + low) / 2))*Sx#(Sx / Sy * 300)
    ell_radius_y = (-4*(1 - r*r)*math.log((upp + low) / 2))*Sy#300 #* 2
    print(f"radius: x = {ell_radius_x}, y = {ell_radius_y}")


    theta = np.degrees(math.atan(2 * r * Sx * Sy / (Sx ** 2 - Sy ** 2)) / 2)

    print("Sx", Sx)
    print("Sy ", Sy)
    print("cov ", cov)
    print("Xmean ", Xmean)
    print("Ymean ", Ymean)
    print("Theta ", theta)

    ellipse = Ellipse(
        (Xmean, Ymean),
        width=ell_radius_x *2,
        height=ell_radius_y *2,
        angle=theta,
        color='black',
        alpha=0.5
    )
    draw_sep(x, y, ellipse)
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


def getMax(data):
    m = -1
    for i in data:
        m = max(m, max(i))
    return m


def find_slice_indices(data, left, right):
    x_indices = []
    y_indices = []

    for i, row in enumerate(data):
        for j, value in enumerate(row):
            if left <= value <= right:
                x_indices.append(j)
                y_indices.append(i)

    return x_indices, y_indices


with h5py.File(fileName, 'r') as f:
    for dset in traverse_datasets(fileName):
        print('Path:', dset)
        print('Shape:', f[dset])
        print('Data type:', f[dset].dtype)
    dataset = f["/BG_DATA/1/DATA"]
    data = np.reshape(dataset, (1920, 1000), 'F')

    print(0.5*getMax(data))
    left_bound = 0.58*getMax(data)
    right_bound = 0.60*getMax(data)
    x_indices, y_indices = find_slice_indices(data[:,:], left_bound, right_bound)

    fig, ax = plt.subplots()
    pylab.plot(data)
    imgplot = ax.imshow(data[:, :], origin='lower')
    imgplot.set_cmap('nipy_spectral')
    colorbar = plt.colorbar(imgplot, orientation='horizontal')
    plt.savefig("aaa")
    plt.show()
    plt.close()
