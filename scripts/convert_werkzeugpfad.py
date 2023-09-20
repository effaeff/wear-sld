import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('./WZ1_V9.csv', delimiter=',', skiprows=12, header=None, names=["time", "spsp", "OffsetX", "OffsetY", "OffsetZ", "X", "Y", "Z", "B", "C"])

data[["OffsetX", "OffsetY", "OffsetZ"]] -= data[["OffsetX", "OffsetY", "OffsetZ"]].iloc[-1,:]
data["X"] += data["OffsetX"]
data["Y"] += data["OffsetY"]
data["Z"] += data["OffsetZ"]


data["X"] += 48 # aestart
data["X"] -= 6 # tool radius

data = data[["time", "spsp", "X", "Y", "Z"]].to_numpy(np.float32)

# spsp = np.max(data[:,1])
# data = data[data[:,1] > 0.9 * spsp, :]

dp = data[1:,2:] - data[:-1,2:]
dp = np.sqrt(np.sum(dp**2, axis=1))
dt = data[1:,0] - data[:-1,0]

vf = dp / (dt / 60.0)

# vf = np.row_stack((vf[0], vf))

data = np.column_stack((data[1:,2:], vf))

# with open("./ncpath_schruppen.txt", "w") as file:
#     for row in data:
#         file.write(f'L {row[0]} {row[1]} {row[2]} 0 0 1 {row[3]}\n')

# print(data)

plt.plot(data[:,0], data[:,1])
plt.grid()
plt.axhline(0, 0, 1, c='k', lw=3)
plt.axhline(35, 0, 1, c='k', lw=3)
plt.show()

plt.plot(data[:,1])
plt.plot(vf)
plt.show()
