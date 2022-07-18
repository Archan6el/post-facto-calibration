from math import log10, sqrt
#import cv2
import numpy as np
from skimage import io, img_as_float
import imquality.brisque as brisque

'''
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
'''

def brisquenr(image):
    return brisque.score(image)

def main():
     original = img_as_float(io.imread("mega_compressed.jpeg", as_gray=True)) #cv2.imread("original_image.jpeg") 
     #compressed = cv2.imread("compressed_image.jpeg", 1)
     #print(original)
     print(brisquenr(original))

     #value = PSNR(original, compressed)
     #print(f"PSNR value is {value} dB")
       
if __name__ == "__main__":
    main()