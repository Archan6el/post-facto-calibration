import matplotlib.pyplot as plt
import random

#the mask is static
mask = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0,0]
shadow = []

xaxis = []

for x in range(1, 21):
    xaxis.append(str(x))

fig = plt.figure()
plt.bar(xaxis, mask)
plt.title("Mask")
plt.show()