import matplotlib.pyplot as plt
import numpy as np
import math
np.random.seed(61)
    

def Boxplot(sample1, sample2, title):
    # Создаем boxplot для первого набора данных
    plt.boxplot(sample1, vert = False, positions=[1], widths=0.6)

    # Создаем boxplot для второго набора данных
    plt.boxplot(sample2, vert = False, positions=[2], widths=0.6)

    plt.yticks([1, 2], ['n = 20', 'n = 100'])

    # Добавляем название графика и метки осей
    plt.title(title)
    plt.xlabel('x')

    plt.savefig(title)  # Сохраняем график в файл
    plt.close()  # Закрываем текущий график, чтобы он не отображался в блокноте


def main():
    normal1 = np.random.normal(0, 1, 20)
    cauchy1 = np.random.standard_cauchy(20)
    student1 = np.random.standard_t(df=3, size=20)
    puasson1 = np.random.poisson(10, size=20)
    uniform1 = np.random.uniform(-(math.sqrt(3)), math.sqrt(3), 20)

    normal2 = np.random.normal(0, 1, 100)
    cauchy2 = np.random.standard_cauchy(100)
    student2 = np.random.standard_t(df=3, size=100)
    puasson2 = np.random.poisson(10, size=100)
    uniform2 = np.random.uniform(-(math.sqrt(3)), math.sqrt(3), 100)

    Boxplot(normal1, normal2, "normal 3-4")
    Boxplot(cauchy1, cauchy2, "cauchy 3-4")
    Boxplot(student1, student2, "student 3-4")
    Boxplot(puasson1, puasson2, "puasson 3-4")
    Boxplot(uniform1, uniform2, "uniform 3-4")


if __name__ == "__main__":
    main()
