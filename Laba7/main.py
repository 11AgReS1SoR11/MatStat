import numpy as np
from scipy.stats import chi2
import scipy.stats as stats

def norm(n):
    return np.random.normal(loc=0, scale=1, size=n)

def student(n):
    return np.random.standard_t(df=3, size=n)

def uniform(n):
    return np.random.uniform(-5, 5, n)

def F_norm(x, mean=0, std_dev=1):
    return stats.norm.cdf(x, loc=mean, scale=std_dev)

def F_student(x, df=3):
    return stats.t.cdf(x, df)

def F_uniform(x, low=-5, high=5):
    return stats.uniform.cdf(x, low, high - low)

inf = 100000000

def CHI(n, sampl, F):

    k = int(1.72*n**(1/3))
    data = sampl(n)  # Generate sample data
    start = -3
    end = 3
    intervals = [-inf]
    for a in np.linspace(start, end, k - 2):
        intervals.append(a)
    intervals.append(inf)

    # Step 1
    alpha = 0.05

    # Step 2:
    kvantil = chi2.ppf(1 - alpha, k - 1)

    # Step 3:
    p = [F(intervals[i+1]) - F(intervals[i]) for i in range(k - 1)]

    # Step 4:
    hist, bins = np.histogram(data, bins=intervals)

    # Step 5:
    chi2_B = 0
    for i in range(k - 1):
        chi2_B += (hist[i] - n*p[i])**2 / (n*p[i])

    # Step 6:
    if chi2_B < kvantil:
        return True  # Null hypothesis is accepted
    else:
        return False  # Null hypothesis is rejected

def main():
    n = 20
    print("n = 20, real = norm")
    print(f"norm = {CHI(n, norm, F_norm)}, student = {CHI(n, norm, F_student)}, uniform = {CHI(n, norm, F_uniform)}")
    print("n = 20, real = student")
    print(f"norm = {CHI(n, student, F_norm)}, student = {CHI(n, student, F_student)}, uniform = {CHI(n, student, F_uniform)}")
    print("n = 20, real = uniform")
    print(f"norm = {CHI(n, uniform, F_norm)}, student = {CHI(n, uniform, F_student)}, uniform = {CHI(n, uniform, F_uniform)}")

    n = 100
    print("n = 100, real = norm")
    print(f"norm = {CHI(n, norm, F_norm)}, student = {CHI(n, norm, F_student)}, uniform = {CHI(n, norm, F_uniform)}")
    print("n = 100, real = student")
    print(f"norm = {CHI(n, student, F_norm)}, student = {CHI(n, student, F_student)}, uniform = {CHI(n, student, F_uniform)}")
    print("n = 100, real = uniform")
    print(f"norm = {CHI(n, uniform, F_norm)}, student = {CHI(n, uniform, F_student)}, uniform = {CHI(n, uniform, F_uniform)}")

if __name__ == "__main__":
    main()