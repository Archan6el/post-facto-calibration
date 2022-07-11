import matplotlib.pyplot as plt
import random

#the mask is static
mask = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
cast = []

xaxis = []

for x in range(1, 44):
    xaxis.append(str(x))

#fig = plt.figure()
plt.rcParams["figure.figsize"] = (20,5)
plt.bar(xaxis, mask)
plt.title("Mask")
plt.show()