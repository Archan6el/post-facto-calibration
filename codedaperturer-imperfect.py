import matplotlib.pyplot as plt
import numpy as np
import random

#the mask is static
mask = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
#for this case, the count or cast "shadow" is almost an exact shadow of the mask, with some imperfections
count = []

#Fills our count array perfectly. 10 signifies no x-ray hits
#100 signifies an x-ray hit
for x in range(len(mask)):
    if mask[x] == 0:

        count.append(10)
    else:
        count.append(100)

#Has x rays bleed into other bins in more randomized way, creating MORE imperfections
for x in range(len(count)):
    pick = random.choice(range(1, 4))
    
    if(x != 0 & x != len(count)):
        pickRan = random.choice(range(1, 10))
        add = count[x]*pick/10
        #if(pickRan < 5):
        #    add = -count[x]*pick/10*(-1)
        count[x] = count[x] + add
        count[x-1] = count[x-1] + add
        count[x-1] = count[x-1] + add
    else:
        count[x] = count[x] + count[x]*0.3
    


#calculates coefficient correlation of 2 arrays
def coef(arr1, arr2):
    coefficient = 0
    for x in range(len(arr1)):
        num = arr1[x] * arr2[x]
        coefficient += num

    return coefficient

#Our cross correlation array
cc = []

#Keeps shifting the count array left, calculates the correlation coefficient
#between the shifted to the left count array and the mask
#and appends it to the correlation array

for x in range(len(mask)):
    
    shifting = np.pad(count, (0, x), mode='constant')[x:]
    shifted = shifting.tolist()
    if (len(shifted) > 0):
        co = coef(mask, shifted)
        cc.append(co)
    else:
        co = coef(mask, count)
        cc.append(co)

#Reversing our cc array here ensures that the center of the bar graph is the case where there is no shift
temp = cc
cc = temp[::-1]

#Keeps shifting the count array right, calculates the correlation coefficient
#between the shifted to the right count array and the mask
#and appends it to the correlation array

for x in range(len(mask)):
    
    shifting = np.pad(count, (x, 0), mode='constant')[:-x]
    shifted = shifting.tolist()
    if (len(shifted) > 0):
        co = coef(mask, shifted)
        cc.append(co)
    else:
        co = coef(mask, count)
        cc.append(co)

#Used only for the x axis of our mask and count graphs
xaxis = []
for x in range(len(mask)):
    xaxis.append(str(x + 1))

#Used only for the x axis of our cross correlation graph
xaxiscc = []
for x in range(len(mask) * 2):
    xaxiscc.append(str(x + 1))

#Everything after this is plotting our graphs
figure, axis = plt.subplots(3)

axis[0].bar(xaxis, mask)
axis[0].set_title("Mask")

axis[1].bar(xaxis, count)
axis[1].set_title("Counts")

axis[2].bar(xaxiscc, cc)
axis[2].set_title("Cross Correlation")
axis[2].tick_params(axis='x', rotation=65)

figure.tight_layout()
figure.set_figheight(5)
figure.set_figwidth(25)
plt.show()
