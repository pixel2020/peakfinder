import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import argrelextrema
import heapq

# Generate a noisy AR(1) sample
np.random.seed(0)
rs = np.random.randn(800)
xs = [0]
for r in rs:
    xs.append(xs[-1] * 0.9 + r)


n = int(len(xs)/8)  # number of points to be checked before and after


maxd = np.maximum.accumulate(xs) - xs
#plt.plot(maxd)
df = pd.DataFrame(maxd, columns=['data'])

df['max'] = df.iloc[argrelextrema(df.data.values, np.greater_equal,
                     order=n)[0]]['data']

peak_max = list(df['max'].fillna(0))
high_peaks = heapq.nlargest(2, list(range(len(peak_max))), key=peak_max.__getitem__)
print(high_peaks)

# Plot results
i1 = high_peaks[0]
j1 = high_peaks[1]

plt.scatter(i1, xs[i1], c='r')
plt.scatter(j1, xs[j1], c='r')
                                


# Find local peaks
mind = np.minimum.accumulate(xs) - xs

#plt.plot(mind)
df = pd.DataFrame(mind, columns=['data'])

df['min'] = df.iloc[argrelextrema(df.data.values, np.less_equal,
                     order=n)[0]]['data']

peak_max = list(df['min'].fillna(0))
low_peaks = heapq.nsmallest(2, list(range(len(peak_max))), key=peak_max.__getitem__)
print(low_peaks)

# Plot results
i1 = low_peaks[0]
j1 = low_peaks[1]
#plt.scatter(df.index, df['min'], c='r')
plt.scatter(i1, xs[i1], c='g')
plt.scatter(j1, xs[j1], c='g')
plt.grid()
plt.plot(xs)
