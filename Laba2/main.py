import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import statistics
np.random.seed(50)


def trimmed_mean(sample, trim_fraction):
    sorted_sample = sorted(sample)
    trim_size = int(len(sorted_sample) * trim_fraction)
    trimmed_sample = sorted_sample[trim_size:-trim_size]
    return sum(trimmed_sample) / len(trimmed_sample)


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


def calculation(Distribution, *args, **kwargs):
    M = [{'M': 0, 'med x': 0, 'Zr': 0, 'Zq': 0, 'Ztr': 0},
         {'M': 0, 'med x': 0, 'Zr': 0, 'Zq': 0, 'Ztr': 0},
         {'M': 0, 'med x': 0, 'Zr': 0, 'Zq': 0, 'Ztr': 0}]

    D = [{'M': 0, 'med x': 0, 'Zr': 0, 'Zq': 0, 'Ztr': 0},
         {'M': 0, 'med x': 0, 'Zr': 0, 'Zq': 0, 'Ztr': 0},
         {'M': 0, 'med x': 0, 'Zr': 0, 'Zq': 0, 'Ztr': 0}]

    array_of_powers = [10, 50, 1000]

    for i in range(1000):
        distrib = Distribution(*args, **kwargs, size=1000)

        for power, k in zip(array_of_powers, range(3)):
            q1 = quartile_1_4(distrib[:power])
            q3 = quartile_3_4(distrib[:power])

            M[k]['M'] += np.mean(distrib[:power]) / 1000
            M[k]['med x'] += statistics.median(distrib[:power]) / 1000
            M[k]['Zr'] += (np.max(distrib[:power]) + np.min(distrib[:power])) / 2 / 1000
            M[k]['Zq'] += (q1 + q3) / 2 / 1000
            M[k]['Ztr'] += trimmed_mean(distrib[:power], 0.25) / 1000

            D[k]['M'] += np.mean(distrib[:power]) ** 2 / 1000
            D[k]['med x'] += statistics.median(distrib[:power]) ** 2 / 1000
            D[k]['Zr'] += (np.max(distrib[:power]) + np.min(distrib[:power])) ** 2 / 2 / 1000
            D[k]['Zq'] += (q1 + q3) ** 2 / 2 / 1000
            D[k]['Ztr'] += trimmed_mean(distrib[:power], 0.25) ** 2 / 1000

    return M, D


def fill_table(table, M, arr_idx):
    for row_offset, values in zip(arr_idx, M):
        for i, val in enumerate(values.values(), start=1):
            table._cells[row_offset, i].get_text().set_text(round(val, 4))


def create_table(name, M, D):
    data = {
        f'{name} n = 10': ['$E$($z$)', '$D$($z$)', f'{name} n = 50', '$E$($z$)', '$D$($z$)', f'{name} n = 1000',
                          '$E$($z$)', '$D$($z$)'],
        r'$MX$': ['', '', r'$MX$', '', '', r'$MX$', '', ''],
        'med x': ['', '', 'med x', '', '', 'med x', '', ''],
        r'$z_R$': ['', '', r'$z_R$', '', '', r'$z_R$', '', ''],
        r'$z_Q$': ['', '', r'$z_Q$', '', '', r'$z_Q$', '', ''],
        r'$z_{tr}$': ['', '', r'$z_{tr}$', '', '', r'$z_{tr}$', '', '']
    }

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 4))
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', fontsize=10)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    fill_table(table, M, [1, 4, 7])
    fill_table(table, D, [2, 5, 8])

    plt.savefig(f'{name}.png', bbox_inches='tight', pad_inches=0.05, transparent=True)
    plt.show()



normal = np.random.normal
M_normal, D_normal = calculation(normal, 0, 1)
create_table("normal", M_normal, D_normal)

cauchy = np.random.standard_cauchy
M_cauchy, D_cauchy = calculation(cauchy)
create_table("cauchy", M_cauchy, D_cauchy)

student = np.random.standard_t
M_student, D_student = calculation(student, 3)
create_table("student", M_student, D_student)

poisson = np.random.poisson
M_poisson, D_poisson = calculation(poisson, 10)
create_table("poisson", M_poisson, D_poisson)

uniform = np.random.uniform
M_uniform, D_uniform = calculation(uniform, -(math.sqrt(3)), math.sqrt(3))
create_table("uniform", M_uniform, D_uniform)
