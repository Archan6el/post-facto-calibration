import matplotlib.pyplot as plt
import numpy as np
import random

#the mask is static of 70 zeros and ones
mask = [1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1]
#for this case, the count or cast "shadow" is an exact shadow of the mask but shifted
count = []
#Contains our extrapolated mask
extrapMask = []

#Extrapolates our mask
for x in range(400):
    indexOfMask = int(x * 70 / 400)
    if(indexOfMask == 0):
        indexOfMask = 1
    if mask[indexOfMask] == 0:
        extrapMask.append(10)
    else:
        extrapMask.append(100)

#In this case, the count/shadow is perfectly aligned with the mask, so it would
#be identical to our extrapolated mask
count = extrapMask

#Randomly chooses to shift the count array left or right by varying degrees
direction = random.choice(range(0, 2))
if direction == 1:
    x = random.choice(range(len(count)))
    count = np.pad(count, (x, 0), mode='constant')[:-x]
else:
    x = random.choice(range(len(count)))
    count = np.pad(count, (0, x), mode='constant')[x:]

#Our cross correlation array
cc = np.correlate(extrapMask, count, "full")

#X axes of graphs
xaxisForMask = np.arange(0, len(mask), 1)
xaxisForCount = np.arange(0, len(count), 1)
xaxiscc = np.arange(0, len(cc), 1)

#Everything after this is plotting our graphs
figure, axis = plt.subplots(3)

axis[0].bar(xaxisForMask, mask)
axis[0].set_title("Mask")

axis[1].bar(xaxisForCount, count)
axis[1].set_title("Shifted Counts")

axis[2].bar(xaxiscc, cc)
axis[2].set_title("Cross Correlation")
axis[2].tick_params(axis='x', rotation=65)

figure.tight_layout()
figure.set_figheight(5)
figure.set_figwidth(25)
plt.show()
