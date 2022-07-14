# imperfection trial 4 - 7/14/2022, 12:54

import matplotlib.pyplot as plt
import numpy as np
import random
import math

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


#Creates a distorted version of our extrapolated mask (creates our count)
for x in range(400):
    indexOfMask = int(x * 70 / 400)
    if(indexOfMask == 0):
        indexOfMask = 1
    if mask[indexOfMask] == 0:
        #appends a random integer
        count.append( random.choice(range(1, 30)) )
        #adds a random decimal to make 
        count[x] = count[x] + random.choice(range(1, 99))/100
    else:
        count.append( random.choice(range(80, 120)) )
        count[x] = count[x] + random.choice(range(1, 99))/100



c = 1000

#adds more distortion with a function
for x in range(len(count)):
    print(count[x])
    count[x] = count[x] + c * math.sin(2*count[x]*math.pi)
    print(count[x])


#adds more distortion with bleeding
#Has x rays bleed into other bins in MORE MORE randomized way, creating MORE imperfections
for x in range(len(count)):
    pick = random.choice(range(4, 8))
    
    if(x != 0 & x != len(count) ):
        pickRan = random.choice(range(30, 60))
        add = count[x]*pick/10 * c
        #    add = -count[x]*pick/10*(-1)
        count[x-1] = count[x-1] + add
        count[x-1] = count[x-1] + add
    else:
        count[x] = count[x] + count[x]*0.3


for x in range (len(count)):
    if (count[x] < 0):
        count[x] = 0

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
axis[1].set_title("Imperfect Counts")

axis[2].bar(xaxiscc, cc)
axis[2].set_title("Cross Correlation")
axis[2].tick_params(axis='x', rotation=65)

figure.tight_layout()
figure.set_figheight(5)
figure.set_figwidth(25)
plt.show()
