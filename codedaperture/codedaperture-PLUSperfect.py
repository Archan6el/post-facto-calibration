import matplotlib.pyplot as plt
import numpy as np
import random

#the mask is static of 70 zeros and ones
mask = [1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1]
count = []

#Fills our count array perfectly. 10 signifies no x-ray hits
#100 signifies an x-ray hit
for x in range(400):
    indexOfMask = int(x * 70 / 400)
    if(indexOfMask == 0):
        indexOfMask = 1
    if mask[indexOfMask] == 0:
        count.append(10)
    else:
        count.append(100)

#Randomly chooses to shift the count array left or right by varying degrees
direction = random.choice(range(0, 2))
if direction == 1:
    x = random.choice(range(len(count)))
    count = np.pad(count, (x, 0), mode='constant')[:-x]
else:
    x = random.choice(range(len(count)))
    count = np.pad(count, (0, x), mode='constant')[x:]

#Our cross correlation array
cc = np.correlate(mask, count, "full")



bottom = 0 
width = 0.8 
#Used only for the x axis of our mask and count graphs
xaxisForMask = np.arange(0, len(mask), 1)
#for x in range(len(mask)):
#    xaxisForMask.append(str(x + 1))
xaxisForCount = []
for x in range(len(count)):
    xaxisForCount.append(str(x + 1))
#Used only for the x axis of our cross correlation graph
xaxiscc = []
for x in range(len(cc) ):
    xaxiscc.append(str(x + 1))
#Everything after this is plotting our graphs
figure, axis = plt.subplots(3)
axis[0].bar(xaxisForMask, mask)
axis[0].set_title("Mask")
axis[1].bar(xaxisForCount, count)
axis[1].set_title("Perfect Counts")
axis[2].bar(xaxiscc, cc)
axis[2].set_title("Cross Correlation")
axis[2].tick_params(axis='x', rotation=65)


figure.tight_layout()
figure.set_figheight(5)
figure.set_figwidth(25)
plt.show()
