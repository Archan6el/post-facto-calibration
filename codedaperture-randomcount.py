import matplotlib.pyplot as plt
import random

#the mask is static
mask = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1]
#for this case, the count or cast "shadow" is random or unrelated to the mask
count = []

for x in range(len(mask)):
    pick = random.choice(range(0, 2))
    if pick == 1:
        count.append(10)
    else:
        count.append(100)

xaxis = []

for x in range(len(mask)):
    xaxis.append(str(x + 1))

figure, axis = plt.subplots(2)

axis[0].bar(xaxis, mask)
axis[0].set_title("Mask")

axis[1].bar(xaxis, count)
axis[1].set_title("Counts")

figure.tight_layout()
figure.set_figheight(5)
figure.set_figwidth(25)
plt.show()