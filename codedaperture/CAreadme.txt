This folder is for files related to coded apertures

The 4 files within this folder are variations of each other. 
They all differ in terms of the count, or shadow cast from the mask, resulting in different cross correlation graphs. 
All of them share the same mask, which is where they are similar

codedaperture-perfect.py has its counts graph as a perfect shadow of the mask
codedaperture-imperfect.py has its counts graph as essentially a perfect shadow of the mask, but x-rays bleed into adjacent bins
codedaperture-zeroShifted has its counts graph as a perfect shadow of the mask, but shifted left or right to varying degrees
codedaperture-randomcount.py has its count graph completely random generated and not related to the mask in any way
