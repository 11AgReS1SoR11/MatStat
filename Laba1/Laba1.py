import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
np.random.seed(30)

normal = np.random.standard_normal(900)
cauchy = np.random.standard_cauchy(201)
student = np.random.standard_t(df=3, size=1001)
poisson = np.random.poisson(10, size=1001)
uniform = np.random.uniform(-(math.sqrt(3)), math.sqrt(3), 1001)

array_of_powers = [10, 50, 1000]

array_of_intervals_normal = [7, 20, 30]
array_of_intervals_cauchy = [5, 12, 15]
array_of_intervals_student = [7, 20, 30]
array_of_intervals_poisson = [3, 10, 15]
array_of_intervals_uniform = [3, 10, 15]

arr_intervals = [array_of_intervals_normal, array_of_intervals_cauchy, array_of_intervals_student, array_of_intervals_poisson, array_of_intervals_uniform]
array_of_distributions = [normal, cauchy, student, poisson, uniform]
names = ["normal", "cauchy", "student", "puasson", "uniform"]

for distribution, name, intervals in zip(array_of_distributions, names, arr_intervals):
    for power, interval in zip(array_of_powers, intervals):
        plt.figure(figsize=(8, 6))
        sns.histplot(distribution[:power], bins=interval, kde=True, stat="density")
        plt.title(f'{name}, size {power}', fontsize=18)
        plt.xlabel('Values', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.savefig(f'{name}, size {power}')