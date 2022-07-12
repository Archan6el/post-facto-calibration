# post-facto-calibration

**Quickly Getting Started:**<br />

Clone this repo with:

`git clone https://github.com/Archan6el/post-facto-calibration`
<br /><br />
The 4 "codedaperture" files are variations of each other. They all differ in terms of the count, or shadow cast from the mask, resulting in different cross correlation graphs. All of them share the same mask, which is where they are similar

Simply run any of the 4 files using python3 and 3 graphs will automatically be generated. The first graph is the mask, the second graph is the counts (cast shadow), and the third is the cross correlation graph between the mask and shadow. 

`python3 codedaperture-[VERSION].py` 
<br /><br />

There are python libraries needed to run the code. If not already installed, install them in a Linux terminal with

`pip3 install matplotlib`<br />
`pip3 install numpy` <br />
`pip3 install random`<br />
