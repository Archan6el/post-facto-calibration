#! /usr/bin/env python3
from numpy import exp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
import random
import numpy as np
from collections import OrderedDict

verbose = False

# the mask is static of some 70 or 72 zeros and ones. zero means the
# mask is closed, 1 means the mask is open.

# mask_real is something we feel is the real mask
mask_real = [1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1,
             0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0,
             1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0,
             1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1]

# some parameters of the run -- we might want to find a way to
# encapsulate them so that we can do sweeps by passing around a single
# object
n_photons = 100
noise_fraction = 0.0
distortion_dial = 0.9          # limits are [0, 1]

mask = mask_real
det_height_cm = 18.74
mask_width_cm = 20.8
det_width_cm = 12.0
det_pix_size_cm = 0.0495
n_det_pix = int(mask_width_cm / det_pix_size_cm + 0.5) # +0.5? or +1? or +0?
mask_det_offset_cm = (mask_width_cm - det_width_cm) / 2


real_vals = []
naive_vals = []

print(n_det_pix)

# random.seed(1234)               # FIXME: for debugging only

def main():
    # find the detector readout for a given angle
    theta_deg = -15              # incidence angle, 0 if perpendicular
    det_readout = gen_readout(mask, theta_deg)
    if verbose:
        print(det_readout)

    # to do cross-correlation we must resample the mask to have as
    # many pixels as the detector
    # resampled_mask = [int(x * len(mask) / len(det_redout)) for x in mask]
    resampled_mask = np.zeros(len(det_readout))
    for resampled_ind in range(len(det_readout)):
        mask_ind = int(resampled_ind * len(mask) / len(det_readout))
        val = mask[mask_ind]
        resampled_mask[resampled_ind] = val
    if verbose:
        print(resampled_mask)
        
    # Our cross correlation array
    # ccor = np.correlate(resampled_mask, det_readout, mode='full')
    ccor = np.correlate(resampled_mask, det_readout, mode='same')
    if verbose:
        print('ccor:', ccor)
    # trick to get the lag
    lag = ccor.argmax() - (len(resampled_mask) - 1)

    # finally we can plot what we've done
    #repare_plots(mask, resampled_mask, det_readout, ccor, lag, theta_deg)
    
    createScipy()


def gen_readout(mask, angle_deg):
    """Given a mask and an incidence angle, simulate photons hitting the
    detector through that mask and at that angle
    """
    # #Contains our extrapolated mask
    readout = np.zeros(n_det_pix)
    # simulate a bunch of photons hitting the detector
    for i in range(n_photons):
        # hypothetical position where it might have hit.  note that we
        # measure things starting at the left end of the mask, and the
        # detector valid range is smaller than that, so we must do
        # arithmetic with those offsets to generate a valid detector
        # position
        det_pos_cm = mask_det_offset_cm + random.random() * det_width_cm
        distorted_pos_cm = apply_distortion(det_pos_cm)
        if verbose:
            print('det_pos_cm:', det_pos_cm, distorted_pos_cm,
                  f'(in range of [{mask_det_offset_cm}, {mask_det_offset_cm+det_width_cm}])')
        # first sanity check: is the detector position < 0 or > width?
        # i.e. is it out of bounds?
        assert(det_pos_cm >= mask_det_offset_cm
               and det_pos_cm < mask_det_offset_cm + det_width_cm)
        assert(distorted_pos_cm >= mask_det_offset_cm
               and distorted_pos_cm < mask_det_offset_cm + det_width_cm)
        # now see if that position is consistent with the
        # mask and the angle it's coming from
        if random.random() < noise_fraction:
            is_valid = True     # noise is always valid
        else:
            is_valid = mask_pos_angle_consistent(mask, distorted_pos_cm,
                                                 angle_deg)
        if is_valid:
            readout_ind = int(n_det_pix * distorted_pos_cm / mask_width_cm)
            assert(readout_ind >= 0 and readout_ind < n_det_pix)
            readout[readout_ind] += 1 # yay! I got a count on the detector


            #updating array for scipy plotting
            real_vals.append(det_pos_cm)
            naive_vals.append(distorted_pos_cm)

        
    return readout


def mask_pos_angle_consistent(mask, det_pos_cm, angle_deg):
    """Determines if the mask and detector position pairs are compatible
    with a photon incident at angle_deg
    """
    # to do this geometry, first figure out the position under the
    # mask if that photon had been straight
    perp_pos_cm = det_pos_cm + det_height_cm * math.tan(angle_deg * math.pi / 180.0)
    if verbose:
        print('perp_pos_cm:', perp_pos_cm)
    # now find the mask slit index, so we can see if it is open or closed
    mask_slit_pos_cm = perp_pos_cm
    mask_slit_ind = int((mask_slit_pos_cm / mask_width_cm) * len(mask))
    if verbose:
        print('MASK_offset,perp,slitpos,width,ratio,slit_ind,len',
              mask_det_offset_cm, perp_pos_cm, 
              mask_slit_pos_cm, mask_width_cm, (mask_slit_pos_cm / mask_width_cm),
              mask_slit_ind, len(mask))
    if mask_slit_ind < 0 or mask_slit_ind >= len(mask):
        return False
    assert(mask_slit_ind >= 0 and mask_slit_ind < len(mask))
    valid = (mask[mask_slit_ind] == 1)
    if verbose:
        print(det_pos_cm, angle_deg, valid)
    return valid


def prepare_plots(mask, resampled_mask, det_readout, ccor, lag, theta_deg):
    """At this time all the plotting stuff is shoved here"""
    # X axes of graphs
    xaxisForMask = np.arange(0, len(mask), 1)
    # xaxisForResampledMask = np.arange(0, mask_width_cm, mask_width_cm / len(resampled_mask))
    xaxisForResampledMask = np.arange(0, len(resampled_mask), 1)
    # xaxisForCount = np.arange(0, mask_width_cm, mask_width_cm / len(det_readout))
    xaxisForCount = np.arange(0, len(det_readout), 1)
    xaxiscc = np.arange(0, len(ccor), 1) - len(ccor) // 2

    # Everything after this is plotting our graphs
    figure, ax = plt.subplots(5, constrained_layout=True, figsize=(15, 10))
    ax[0].axes.yaxis.set_ticklabels([])
    ax[1].axes.yaxis.set_ticklabels([])

    figure.set_figheight(14)
    figure.set_figwidth(25)
    figure.suptitle(f'Mask/readout cross-correlations, theta={theta_deg} deg')

    ax[0].bar(xaxisForMask, ((np.zeros(len(mask)) + 1) - mask), width=1.0,
              align='edge', color='black', edgecolor='white')
    ax[0].bar(xaxisForMask, mask, width=1.0,
              align='edge', color='orange', edgecolor='white', alpha=0.8)
    ax[0].set_title('mask (solid black is closed, orange is open)')

    ax[1].bar(xaxisForResampledMask, ((np.zeros(len(resampled_mask)) + 1) - resampled_mask),
              width=1.0, align='edge', color='black', edgecolor='white')
    ax[1].bar(xaxisForResampledMask, resampled_mask,
              width=1.0, align='edge', color='orange', edgecolor='white', alpha=0.8)
    ax[1].set_title('resampled mask (solid black is closed, orange means open)')

    ax[2].bar(xaxisForCount, det_readout)
    ax[2].set_title('detector readout (x-axis is in mask coordinate frame)')

    # OK, I confess I'm not sure how "lag" is measured
    ax[3].bar(xaxiscc, ccor, color='green')
    ax[3].set_title(f'Cross Correlation (bars) - lag is {lag}')

    ax[4].stem(xaxiscc, ccor)
    ax[4].set_title(f'Cross Correlation (stems) - lag is {lag}')

    # align.xaxes(ax[1], mask_width_cm/2, ax[2], mask_width_cm/2, 0.5)

    """
    for suffix in ['png', 'svg']:
        ofname = f'mask_readout_ccor.{suffix}'
        print(f'saving file {ofname}')
        plt.savefig(ofname)
    """
    plt.show()
    


def apply_distortion(x_real):
    """Applies a simple distortion to the signal -- in this case something
    like x -> x + c*sin(2*pi*x / scale)

    """
    xlow = 0
    xhigh = mask_width_cm
    x_bad = x_real + distortion_dial * math.sin(2*math.pi*(x_real / mask_width_cm))
    return x_bad


def createScipy():
    
    #create dictionary mapping x and y values (x-> key, y->value)
    #sort dict by x and store keys to x, values to y
    count = 0
    dict = {}
    for i in (real_vals):
        dict[i] = naive_vals[count]
        count += 1
    
    #Sort the dictionary so that all of our data is in order
    dictSorted = OrderedDict(sorted(dict.items()))

    #x and y for our scipy functions
    x = np.array(list (dictSorted.keys()) )
    y = np.array(list (dictSorted.values() ))


    #establishes different types of functions we want the regression line to be
    def func1(x, a):
        return a*x

    def func2(x, a):
        return a*x**2

    def func3(x, a, b, c):
        return a*x**3+b*x**2+c

    def sigmoid(x, a, b, c, d):
        return (a / (x - b)) + c + (d*x)

    #linear function
    params, _ = curve_fit(func1, x, y)
    a = params[0]
    yfit1 = a*x

    #quadratic function
    params, _  = curve_fit(func2, x, y)
    a = params[0]
    yfit2 = a*x**2

    #cubic function
    params, _  = curve_fit(func3, x, y)
    a, b, c = params[0], params[1], params[2]
    yfit3 = a*x**3+b*x**2+c

    #sigmoid function
    params, _  = curve_fit(sigmoid, x, y)
    a, b, c, d = params[0], params[1], params[2], params[3]
    yfitSig = (a / (x - b)) + c + (d*x)
    

    #plotting the graphs here
    #yfit# -> the # correlates to the highest degree of exponent of the highest x
    plt.plot(x, y, 'bo', label="y-original")
    plt.plot(x, yfit1, label="linear regression")
    plt.plot(x, yfit2, label="quadratic regression")
    plt.plot(x, yfit3, label="cubic regression")
    plt.plot(x, yfitSig, label="sigmoid")
   
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.show() 

if __name__ == '__main__':
    main()
