#! /usr/bin/env python3
#from re import X
from numpy import exp
#from pendulum import naive
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
import random
import numpy as np
from collections import OrderedDict
from scipy.stats import pearsonr
from sqlalchemy import true

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
distortion_dial = 1         # limits are [0, 1]

mask = mask_real
det_height_cm = 18.74
mask_width_cm = 20.8
det_width_cm = 12.0
det_pix_size_cm = 0.0495
n_det_pix = int(mask_width_cm / det_pix_size_cm + 0.5) # +0.5? or +1? or +0?
mask_det_offset_cm = (mask_width_cm - det_width_cm) / 2

real_vals = [6.469734334222604, 9.784052567573474, 11.474638288210137, 13.406001371509154, 5.124122238459028, 
8.670306055808428, 13.659288940313875, 5.297207584460752, 12.73294619565659, 6.319152276787557, 6.3742820516493985, 
13.39196114701655, 6.358093320772113, 8.71800295958431, 14.771520458874143, 8.783450052669053, 5.551411782503367, 12.484310979785706, 
6.333214380034148, 11.391418126730192, 5.294702883206989, 14.567437407495762, 14.432565049737184, 5.885980328261018, 13.282191152849387, 
13.821279973323424, 12.49730970949375, 13.359009180834827, 5.015696434882418, 14.287221887784723, 14.290726077821917, 12.680744592323386, 
5.8375652049353, 6.432723947587867, 11.447987933892616, 8.462971823896137, 13.782194899739583, 14.585071755827398, 8.597955794141917, 4.563227469719163, 
14.532076521708426, 13.45381226483366, 11.399196401468917, 5.609731014005759, 4.4128439741784025, 9.541330302988955, 14.857282524729818]

naive_vals = [7.3970739043175255, 9.969043927071004, 11.155687103627523, 12.617701152923114, 6.123859566884064, 9.16935366898281, 12.826263691058397, 
6.296776490272765, 12.085119883216763, 7.2625490530895425, 7.312024898348968, 12.606277592882027, 7.297523473333458, 9.204513512933907, 13.802673377001007, 
9.252594099342495, 6.5457828265494244, 11.89547268828387, 7.275193788274939, 11.096391068874949, 6.294293716908906, 13.615689034680528, 13.494105809955594, 
6.864587322512722, 12.51744878181858, 12.962190867512057, 11.905302281975082, 12.57952228073988, 6.0141470533911425, 13.36482637837911, 13.367922230572514, 
12.045010855316573, 6.819076373612105, 7.364189250750114, 11.136676975410673, 9.015279703393345, 12.929208492162473, 13.631702172380752, 9.11582195969817, 
5.544784441802859, 13.58366037204009, 12.656708210126384, 11.101925112961588, 6.6020812812407685, 5.384706992405885, 9.797815213631731, 13.882345225200464]

#print(pearsonr(real_vals, naive_vals))

#print(n_det_pix)

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
    resampled_mask = np.zeros(len(naive_vals))
    for resampled_ind in range(len(naive_vals)):
        mask_ind = int(resampled_ind * len(mask) / len(naive_vals))
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

    
    p, _ = optimize(real_vals)
    print(p, _)
    list = []
    for x in naive_vals:
        list.append(sigmoid(x, p[0], p[1], p[2], p[3]))
    

    #print(naive_vals)
    #print(len(resampled_mask))
   
    print(list)
    

    xcor_og = np.correlate(resampled_mask, naive_vals, 'full')
    xcor_new = np.correlate(resampled_mask, list, 'full')
    #print(len(xcor_og))
    #print(len(xcor_new))
    # finally we can plot what we've done
    prepare_plots(mask, resampled_mask, det_readout, xcor_og, xcor_new, lag, theta_deg)
    #print(xcor_og)
    #print(xcor_new)
    
    #print(pearsonr(real_vals, real_vals))
    #print(optimize(real_vals))
    #print(gen_metric(resampled_mask, real_vals))



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
            #real_vals.append(det_pos_cm)
            #naive_vals.append(distorted_pos_cm)

    #print(real_vals)
    #print(naive_vals)   
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


def prepare_plots(mask, resampled_mask, det_readout, ccor_original, ccor_new, lag, theta_deg):
    """At this time all the plotting stuff is shoved here"""
    # X axes of graphs
    xaxisForMask = np.arange(0, len(mask), 1)
    # xaxisForResampledMask = np.arange(0, mask_width_cm, mask_width_cm / len(resampled_mask))
    xaxisForResampledMask = np.arange(0, len(resampled_mask), 1)
    # xaxisForCount = np.arange(0, mask_width_cm, mask_width_cm / len(det_readout))
    xaxisForCount = np.arange(0, len(det_readout), 1)
    xaxiscc = [] #np.arange(0, len(ccor_original), 1) - len(ccor_original) // 2
    for x in range(0, len(ccor_original)):
        xaxiscc.append(x)
   
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
    ax[3].bar(xaxiscc, ccor_original, color='green')
    ax[3].set_title(f'Original Cross Correlation (bars) - lag is {lag}')

    ax[4].bar(xaxiscc, ccor_new)
    ax[4].set_title(f'New Cross Correlation (stems) - lag is {lag}')

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
    x_bad = x_real + distortion_dial * math.sin(2*math.pi*(x_real/ mask_width_cm))
    return x_bad


def sigmoid(x, a, b, c, d):
        return (a / (x - b)) + c + (d*x)

def gen_random(arr):
    """Generates a random array by running the given array through our
    sigmoid function with random parameters, each being a number from -5 and 5"""
    a = random.uniform(0, 2) #1.573374532385897
    b = random.uniform(3, 5) #4.019185691943018
    c = random.uniform(-5, -3) #-4.00502684639661
    d = random.uniform(0, 2) #1.359444812728886
    
    new = [0]*len(arr)

    for x in range(len(arr)):
        
        num = sigmoid(arr[x], a, b, c, d)
        if num < 0:
            #print("!", end="")
            new[x] = 0
        else:
            new[x] = num
    
    return new, a, b, c, d

def gen_metric(mask, count):
    matrix = np.corrcoef(mask, count)
    #num, _ = pearsonr(mask, count)
    num = matrix[0][1]
    if math.isnan(num):
        return 0
    else:
        return num

def possible_answers(mask, arr):
    answers = [0]*16
    arrs = [0]*16
    param_answers = [0]*16

    count = 0
    while count < len(answers):
        #print("#")
        randarr, a, b, c, d = gen_random(arr)
        params = [a, b, c, d]

        '''
        negative = False
        for x in randarr:
            if x < 0:
                negative = True
                break
        
        if negative != True:   
        '''   
        answers[count] = gen_metric(mask, randarr)
        param_answers[count] = params
        arrs[count] = randarr
        count += 1

    return answers, param_answers, arrs

def get_best(metric_arr, param_arr, arrs):
    max = metric_arr[0]
    #num = abs(max - 132)

    for x in range(len(metric_arr)):
        #numx = abs(metric_arr[x] - 132)
        if metric_arr[x] > max: #numx < num: 
            max = metric_arr[x]

    index = metric_arr.index(max)

    return max, param_arr[index], arrs[index]


def optimize(mask):
    
    #Actual answer
    #a, b, c, d = 1.573374532385897, 4.019185691943018, -4.00502684639661, 1.359444812728886

    current_metric = gen_metric(mask, naive_vals)
    #print(current_metric)
    current_arr, a, b, c, d = gen_random(naive_vals)
    parameters = [a, b, c, d]

    metrics, params, arrays = possible_answers(mask, naive_vals)
    better_metric, better_params, better_arr = get_best(metrics, params, arrays)
    print(current_metric, better_metric)
    #print(better_metric)
    #better_metric = gen_metric(mask, better_arr)

    while current_metric < better_metric:
        
        #current_arr = better_arr
        current_metric = better_metric
        parameters = better_params

        metrics, params, arrays = possible_answers(mask, naive_vals)

        better_metric, better_params, better_arr = get_best(metrics, params, arrays)
        print(current_metric, better_metric)
        
        #print("!")

        #if ():
        #    break
    
    return parameters, current_metric


if __name__ == '__main__':
    main()
