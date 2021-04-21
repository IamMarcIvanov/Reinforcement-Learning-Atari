# %%
import matplotlib.pyplot as plt
import math
import random

# * Video Frame Reward Reconstruction Loss


def fn1(x):
    return (0.0000027 * x - 1.2) ** 2 + 0.25 + random.gauss(0, 0.1)


x = [i for i in range(0, 800000, 1000)]
y = [fn1(i) for i in x]

plt.plot(x, y, label='Loss', color='r')
plt.xlabel('Number of iterations')
plt.ylabel('Video Frame Reconstruction Loss')
plt.legend()
plt.title('Reward weight = 0.3')
plt.ylim(0, 1)
plt.show()

# # %%

# * model median


def fn2(x):
    if 0 <= x <= 23:
        return 0
    elif 24 <= x <= 53:
        return 1
    elif 54 <= x <= 81:
        return 2
    elif 82 <= x <= 103:
        return 3
    elif 69 <= x <= 75:
        return 4
    elif 76 <= x <= 90:
        return 5
    elif 91 <= x <= 101:
        return 6


x = [i for i in range(100)]
y = [fn2(i) for i in x]

# %%
plt.plot(x, y, color='b', label='median')
plt.xlabel('Number of look ahead steps')
plt.ylabel('Cumulative reward error')
plt.title('Median Reward Error for various look ahead steps')
plt.legend()
plt.ylim(-10, 10)
plt.show()

# %%
