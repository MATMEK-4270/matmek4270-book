import matplotlib.pyplot as plt
import numpy as np

def plot_with_offset(data, xj, figsize=(4, 3)):
    Nd = len(data)
    v = np.array(list(data.values()))
    t = np.array(list(data.keys()))
    dt = t[1]-t[0]
    v0 = abs(v).max()
    fig = plt.figure(facecolor='k', figsize=figsize)
    ax = fig.add_subplot(111, facecolor='k')
    for i, u in data.items():
        ax.plot(xj, u+i*v0/dt, 'w', lw=2, zorder=i)
        ax.fill_between(xj, u+i*v0/dt, i*v0/dt, facecolor='k', lw=0, zorder=i-1)
    plt.show()