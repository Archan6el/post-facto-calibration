#! /usr/bin/env python3
#NOTE - All optimization related functions here will eventually be placed within the postfactolib library. They are not placed there yet
#due to parameter optimization not yet working

import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from collections import OrderedDict
import random
import pygad
import postfactolib

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
theta_deg = 0             # incidence angle, 0 if perpendicular


#print(pearsonr(real_vals, naive_vals))

#print(n_det_pix)

# random.seed(1234)               # FIXME: for debugging only

 #Initializing our calibrate class within the postfactolib module so that we can access the functions
pfl = postfactolib.calibrate(mask, mask_width_cm, mask_det_offset_cm, det_width_cm, det_height_cm, n_det_pix, n_photons, theta_deg)



def main():

    def fitness_function(solution, solution_idx):
        """Fitness function for our genetic algorithm. Incorporates our own defined metric (xcor peak - second highest peak)
        to create the metric that the genetic algorithm will use. The genetic algorithm will try to optimize our own defined
        metric and try to get it as close to the target metric as possible"""

        print(solution)
        new = [0]*len(det_poslist_naive)
        for x in range(len(det_poslist_naive)):
            new[x] = sigmoid(det_poslist_naive[x], solution[0], solution[1], solution[2], solution[3])
        metric = gen_metric(resampled_mask, new)
        output = 1 / abs(metric - target)
        return output

    # find the detector readout for a given angle
    det_readout = pfl.gen_readout()

    # to do cross-correlation we must resample the mask to have as
    # many pixels as the detector
    resampled_mask = pfl.resample_mask(det_readout)
    
    #Stuff to get the lag 
    ccor = np.correlate(resampled_mask, det_readout, mode='same')
    lag = ccor.argmax() - (len(resampled_mask) - 1)

    #Real and Naive positions if you are generating new ones each time
    det_poslist_real = pfl.get_real_poslist()
    det_poslist_naive = pfl.get_naive_poslist()

    #Get our target metric
    target = gen_metric(resampled_mask, det_poslist_naive)

    #Paramaters for our genetic algorithm
    sol_per_pop = 50 #Number of solutions per population
    num_genes = 4 #Number of genes (essentially parameters)

    init_range_low = -5 #Lowest value that a parameter can be
    init_range_high = 5 #Highest value that a parameter can be

    mutation_percent_genes = 10 #Percent chance that a gene is mutated (increases variety)

    #Create our genetic algorithm object
    ga = pygad.GA(num_generations=100,
                       num_parents_mating=2, 
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop, 
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       mutation_percent_genes=mutation_percent_genes,
                       crossover_probability=0.2)

    #Run the genetic algorithm
    ga.run()

    #Once we get our solution (our parameters), use the parameters to transform the naive positions into an estimate of the real positions
    solution, solution_fitness, solution_idx = ga.best_solution()
    estimated = [0]*len(det_poslist_naive)

    for x in range(len(det_poslist_naive)):
        estimated[x] = sigmoid(det_poslist_naive[x], solution[0], solution[1], solution[2], solution[3])

    #Get the metric of our estimated real positions
    metric = gen_metric(resampled_mask, estimated)

    # finally we can plot what we've done
    prepare_plots(mask, resampled_mask, det_readout, estimated, lag, theta_deg, target, metric, det_poslist_naive, det_poslist_real)

def prepare_plots(mask, resampled_mask, det_readout, estimated, lag, theta_deg, original_metric, generated_metric, naive_vals, real_vals):
    """At this time all the plotting stuff is shoved here"""
    # X axes of graphs
    xaxisForMask = np.arange(0, len(mask), 1)
    # xaxisForResampledMask = np.arange(0, mask_width_cm, mask_width_cm / len(resampled_mask))
    xaxisForResampledMask = np.arange(0, len(resampled_mask), 1)
    # xaxisForCount = np.arange(0, mask_width_cm, mask_width_cm / len(det_readout))
    xaxisForCount = np.arange(0, len(det_readout), 1)

    ccor_original = np.correlate(mask, naive_vals, 'full')
    ccor_new = np.correlate(mask, estimated, 'full')
    xaxiscc = [] #np.arange(0, len(ccor_original), 1) - len(ccor_original) // 2
    for x in range(0, len(ccor_original)):
        xaxiscc.append(x)
    
    # Everything after this is plotting our graphs
    figure, ax = plt.subplots(6, constrained_layout=True, figsize=(15, 10))
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
    #ax[3].scatter(real_vals, naive_vals)
    ax[3].set_title(f'Cross Correlation of Mask to Naive - lag is {lag}')

    ax[4].bar(xaxiscc, ccor_new)
    #ax[4].scatter(estimated, naive_vals)
    ax[4].set_title(f'Cross Correlation of Mask to Estimated - lag is {lag}')

    ax[5].scatter(real_vals, naive_vals, label='Real to Naive')
    ax[5].scatter(estimated, naive_vals, label = 'Estimated to Naive')
    ax[5].set_title(f'Lag is {lag} - Original Metric is {original_metric} - Generated Metric is {generated_metric}')
    ax[5].legend(loc = 'upper left')

    plt.show()
    
def sigmoid(x, a, b, c, d):
    """Our sigmoid function"""
    return (a / (x - b)) + c + (d*x)

def gen_metric(mask, estimated_real):
    """Generates our metric which is xcor peak - second highest peak
    Using this metric gives relatively precise results, though not accurate"""
    
    try:
        #Cross correlate the mask and whatever array is passed through
        xcor = np.correlate(mask, estimated_real, 'full')

        #Find the index of the highest peak and the second highest peak
        peak_indices, peak_dict = signal.find_peaks(xcor, height=0.2, distance=3)
        peak_heights = peak_dict['peak_heights']
        highest_peak_index = peak_indices[np.argmax(peak_heights)]
        second_highest_peak_index = peak_indices[np.argpartition(peak_heights,-2)[-2]]

        #Get values of the highest peak and second highest peak
        highest = xcor[highest_peak_index]
        second_highest = xcor[second_highest_peak_index]
        
        #Subtract the 2 and return the value
        num = highest - second_highest
        
        return num
    except:
        return 0

#These 2 functions are here in case you want to see our original graphs
def simphotons():
    """Uses postfactolib module to simulate photons"""
    det_readout = pfl.gen_readout()
    resampled_mask = pfl.resample_mask(det_readout)

    ccor = np.correlate(resampled_mask, det_readout, 'full')

    pfl.plot_xcor(resampled_mask, det_readout, ccor)

def curvefit(real, naive):
    """Uses postfactolib module to curvefit real and naive positions"""
    pfl.plot_curvefit(real, naive, fit='sigmoid')

if __name__ == '__main__':
    main()
