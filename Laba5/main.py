import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import spearmanr

def generate_normal_samples(size, rho):
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    samples = np.random.multivariate_normal(mean, cov, size)
    return samples[:, 0], samples[:, 1]


def generate_mixture_samples(size, rho):
    samples1 = np.random.multivariate_normal([0, 0], [[1, 0.9], [0.9, 1]], size)
    x1, y1 = samples1[:, 0], samples1[:, 1]
    samples2 = np.random.multivariate_normal([0, 0], [[10, -0.9*100], [-0.9*100, 10]], size)
    x2, y2 = samples2[:, 0], samples2[:, 1]
    X = [0.9*x1[i] + 0.1*x2[i] for i in range(len(x1))]
    Y = [0.9*y1[i] + 0.1*y2[i] for i in range(len(y1))]
    return X, Y


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]


def plot_points_with_ellipses(xdata, ydata, rho, file_name, text=None):
    fig, ax = plt.subplots()

    cov = np.cov(xdata, ydata)
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * 2 * np.sqrt(vals)

    ell = Ellipse(xy=(np.mean(xdata), np.mean(ydata)),
            width=w, height=h,
            angle=theta, color='black', alpha=0.2)

    ax.add_artist(ell)
    ax.scatter(xdata, ydata, c='red', lw = 0, alpha=0.7, s=95)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Plot of Generated Points with Ellipses of Dispersion')
    ax.grid(True)

    if text is not None:
        plt.text(0.5, -0.1, text, ha='center', fontsize=12)  # Добавляем текст ниже графика

    plt.savefig(file_name)  # Сохраняем график в файл
    plt.close()  # Закрываем текущий график, чтобы он не отображался в блокноте


def pearson_correlation(x, y):
    n = len(x)
    if n != len(y):
        raise ValueError("Размеры выборок не совпадают")

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator = (sum((xi - mean_x) ** 2 for xi in x) * sum((yi - mean_y) ** 2 for yi in y)) ** 0.5

    if denominator == 0:
        return 0  # Если одна из выборок состоит из одного и того же значения

    return numerator / denominator


def quadrant_correlation(x, y):
    n = len(x)
    if n != len(y):
        raise ValueError("Размеры выборок не совпадают")

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    xy = zip(x, y)
    plus_plus = sum(1 for xi, yi in xy if xi > x_mean and yi > y_mean)
    minus_plus = sum(1 for xi, yi in xy if xi < x_mean and yi > y_mean)
    plus_minus = sum(1 for xi, yi in xy if xi > x_mean and yi < y_mean)
    minus_minus = sum(1 for xi, yi in xy if xi < x_mean and yi < y_mean)

    return (plus_plus + minus_minus - minus_plus - plus_minus) / n


def spearman_rank_correlation(x, y):
    n = len(x)
    if n != len(y):
        raise ValueError("Размеры выборок не совпадают")

    rho, p = spearmanr(x, y)
    return rho


def variance(data):
    n = len(data)
    if n < 2:
        raise ValueError("Дисперсия не может быть вычислена для выборки размером меньше 2")

    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    return variance


def process_correlation_statistics(size, rho, generate_samples_function, name):
    pearson = []
    quadrant = []
    spearman = []

    for cnt in range(1000):
        x, y = generate_samples_function(size, rho)

        pearson.append(pearson_correlation(x, y))
        quadrant.append(quadrant_correlation(x, y))
        spearman.append(spearman_rank_correlation(x, y))

    text = f"""
    name = {name}, size = {size}, rho = {rho}
    mean:
        pearson: {round(np.mean(pearson), 6)}
        spearman: {round(np.mean(spearman), 6)}
        quadrant: {round(np.mean(quadrant), 6)}
    mean_square:
        pearson: {round(np.mean(np.square(pearson)), 6)}
        spearman: {round(np.mean(np.square(spearman)), 6)}
        quadrant: {round(np.mean(np.square(quadrant)), 6)}
    variance:
        pearson: {round(variance(pearson), 6)}
        spearman: {round(variance(spearman), 6)}
        quadrant: {round(variance(quadrant), 6)}
    """
    print(text)
    plot_points_with_ellipses(x, y, rho, f"{name}_{size}_{rho}.png")


def main():
    sample_sizes = [20, 60, 100]     # Размеры выборок
    rhos = [0, 0.5, 0.9] # Значения коэффициента корреляции

    for size in sample_sizes:
        for rho in rhos:
            process_correlation_statistics(size, rho, generate_normal_samples, "normal")

    for size in sample_sizes:
            process_correlation_statistics(size, rho, generate_mixture_samples, "mixture")

if __name__ == "__main__":
    main()
