import matplotlib.pyplot as plt
from collections import deque
import random

#the mask is static
mask = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
#for this case, the count or cast "shadow" is random or unrelated to the mask
count = []

#Fills our count array randomly. 100 signifies no x-ray hits
#10 signifies an x-ray hit
for x in range(len(mask)):
    pick = random.choice(range(0, 2))
    if pick == 1:
        count.append(10)
    else:
        count.append(100)


#calculates coefficient correlation of 2 arrays
def coef(arr1, arr2):
    coefficient = 0
    for x in range(len(arr1)):
        num = arr1[x] * arr2[x]
        coefficient += num

    return coefficient

#Our cross correlation array
cc = []

#Keeps shifting the count array, calculates the correlation coefficient
#between the shifted count array and the mask
#and appends it to the correlation array
shifting = deque(count)
for x in range(len(mask)):
    shifting.rotate(x)
    co = coef(mask, shifting)
    cc.append(co)

#Used only for the x axis of our graphs
xaxis = []
for x in range(len(mask)):
    xaxis.append(str(x + 1))

#Everything after this is plotting our graphs
figure, axis = plt.subplots(3)

axis[0].bar(xaxis, mask)
axis[0].set_title("Mask")

axis[1].bar(xaxis, count)
axis[1].set_title("Counts")

axis[2].bar(xaxis, cc)
axis[2].set_title("Cross Correlation")

figure.tight_layout()
figure.set_figheight(5)
figure.set_figwidth(25)
plt.show()