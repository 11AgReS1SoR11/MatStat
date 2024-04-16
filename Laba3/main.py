import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
np.random.seed(42)

def quartile_1_4(sample):
    n = len(sample)
    sorted_sample = sorted(sample)
    index = n // 4
    if n % 4 == 0:
        return sorted_sample[index - 1]
    else:
        return sorted_sample[index]

def quartile_3_4(sample):
    n = len(sample)
    sorted_sample = sorted(sample)
    index = 3 * n // 4
    if n % 4 == 0:
        return sorted_sample[index - 1]
    else:
        return sorted_sample[index]
    
def Boxplot(sample1, sample2, title):
    # Q1 = quartile_1_4(sample)
    # Q3 = quartile_3_4(sample)
    # X1 = Q1 - 3/2*(Q3 - Q1)
    # X3 = Q3 + 3/2*(Q3 - Q1)
    # outliers = [x for x in sample if x > X3 or x < X1]

    # Создаем boxplot для первого набора данных
    plt.boxplot(sample1, vert = False, positions=[1], widths=0.6)

    # Создаем boxplot для второго набора данных
    plt.boxplot(sample2, vert = False, positions=[2], widths=0.6)

    plt.yticks([1, 2], ['n = 20', 'n = 100'])

    # Добавляем название графика и метки осей
    plt.title(title)
    plt.xlabel('x')

    # Отображаем график
    plt.savefig(f'{title}')



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

    Boxplot(normal1, normal2, "normal")
    Boxplot(cauchy1, cauchy2, "cauchy")
    Boxplot(student1, student2, "student")
    Boxplot(puasson1, puasson2, "puasson")
    Boxplot(uniform1, uniform2, "uniform")


if __name__ == "__main__":
    main()