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

'''
#Positions of where x rays actually hit the detector
det_poslist_real = [6.469734334222604, 9.784052567573474, 11.474638288210137, 13.406001371509154, 5.124122238459028, 
8.670306055808428, 13.659288940313875, 5.297207584460752, 12.73294619565659, 6.319152276787557, 6.3742820516493985, 
13.39196114701655, 6.358093320772113, 8.71800295958431, 14.771520458874143, 8.783450052669053, 5.551411782503367, 12.484310979785706, 
6.333214380034148, 11.391418126730192, 5.294702883206989, 14.567437407495762, 14.432565049737184, 5.885980328261018, 13.282191152849387, 
13.821279973323424, 12.49730970949375, 13.359009180834827, 5.015696434882418, 14.287221887784723, 14.290726077821917, 12.680744592323386, 
5.8375652049353, 6.432723947587867, 11.447987933892616, 8.462971823896137, 13.782194899739583, 14.585071755827398, 8.597955794141917, 4.563227469719163, 
14.532076521708426, 13.45381226483366, 11.399196401468917, 5.609731014005759, 4.4128439741784025, 9.541330302988955, 14.857282524729818]

#Distorted positions of x ray hits
det_poslist_naive = [7.3970739043175255, 9.969043927071004, 11.155687103627523, 12.617701152923114, 6.123859566884064, 9.16935366898281, 12.826263691058397, 
6.296776490272765, 12.085119883216763, 7.2625490530895425, 7.312024898348968, 12.606277592882027, 7.297523473333458, 9.204513512933907, 13.802673377001007, 
9.252594099342495, 6.5457828265494244, 11.89547268828387, 7.275193788274939, 11.096391068874949, 6.294293716908906, 13.615689034680528, 13.494105809955594, 
6.864587322512722, 12.51744878181858, 12.962190867512057, 11.905302281975082, 12.57952228073988, 6.0141470533911425, 13.36482637837911, 13.367922230572514, 
12.045010855316573, 6.819076373612105, 7.364189250750114, 11.136676975410673, 9.015279703393345, 12.929208492162473, 13.631702172380752, 9.11582195969817, 
5.544784441802859, 13.58366037204009, 12.656708210126384, 11.101925112961588, 6.6020812812407685, 5.384706992405885, 9.797815213631731, 13.882345225200464]
'''
# find the detector readout for a given angle
det_readout = pfl.gen_readout()

# to do cross-correlation we must resample the mask to have as
# many pixels as the detector
resampled_mask = pfl.resample_mask(det_readout)

#Real and Naive positions if you are generating new ones each time
det_poslist_real = pfl.get_real_poslist()
det_poslist_naive = pfl.get_naive_poslist()

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

target = gen_metric(resampled_mask, det_poslist_naive)





def main():
    
    #Real and naive positions are currently hard coded to help troubleshoot problems. Normally the real and naive positions
    #change every time the program is run
    
    #Get the original metric
    og_metric = gen_metric(resampled_mask, det_poslist_naive)
        
    # Our cross correlation array
    ccor = np.correlate(resampled_mask, det_readout, mode='same')
    if verbose:
        print('ccor:', ccor)

    # trick to get the lag
    lag = ccor.argmax() - (len(resampled_mask) - 1)

    sol_per_pop = 50
    num_genes = 4

    init_range_low = -5
    init_range_high = 5

    mutation_percent_genes = 10


    ga = pygad.GA(num_generations=100,
                       num_parents_mating=2, 
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop, 
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       mutation_percent_genes=mutation_percent_genes)

    ga.run()

    estimated = [0]*len(det_poslist_naive)
    solution, solution_fitness, solution_idx = ga.best_solution()
    print(solution_fitness)
    
    for x in range(len(det_poslist_naive)):
        estimated[x] = sigmoid(det_poslist_naive[x], solution[0], solution[1], solution[2], solution[3])

    print(det_poslist_naive)
    #Run our optimize function, which will get us our parameters, our final metric, and our final array
    #p, metric, generated_real = optimize(resampled_mask, det_poslist_naive, 500)
    #print(p, metric)
    #print(generated_real)
    metric = gen_metric(resampled_mask, estimated)
    # finally we can plot what we've done
    prepare_plots(mask, resampled_mask, det_readout, estimated, lag, theta_deg, og_metric, metric, det_poslist_naive, det_poslist_real)
    #simphotons()
    #curvefit(det_poslist_real, det_poslist_naive)
    
    #54.17

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

def gen_estimated(naive_pos_list):
    """Generates estimated real values by running the given array through our
    sigmoid function with random parameters. Commented values are the real parameters
    found while curve fitting"""

    #Randomly generate parameters
    a = random.uniform(-5, 5) #1.573374532385897
    b = random.uniform(-5, 5) #4.019185691943018
    c = random.uniform(-5, 5) #-4.00502684639661
    d = random.uniform(-5, 5) #1.359444812728886
   
    estimated_pos_list = [0]*len(naive_pos_list)

    #Takes the values of the array passed through and runs it through our sigmoid function using the parameters that 
    #were randomly generated above
    for x in range(len(naive_pos_list)):
        
        num = sigmoid(naive_pos_list[x], a, b, c, d)

        #Prevent negatives
        if num < 0:
            estimated_pos_list[x] = 0
        else:
            estimated_pos_list[x] = num
 
    #Returns the freshly made array (which are estimated real values) as well as the parameters
    return estimated_pos_list, a, b, c, d

def fitness_function(solution, solution_idx):
    print(solution)
    new = [0]*len(det_poslist_naive)
    for x in range(len(det_poslist_naive)):
        new[x] = sigmoid(det_poslist_naive[x], solution[0], solution[1], solution[2], solution[3])
    metric = gen_metric(resampled_mask, new)
    output = 1 / abs(metric - target)
    return output
    
def possible_answers(mask, naive_list):
    """Generates 200 possible answers (Estimated real-value arrays)"""

    metric_array = [0]*200
    estimated_real_array = [0]*200
    parameter_array = [0]*200

    #Continuously generates estimated real values. Stores those values and their 
    #corresponding parameters. Generates the metric using those estimated real values and store that as well
    count = 0
    while count < len(metric_array):
        estimated_real, a, b, c, d = gen_estimated(naive_list)
        params = [a, b, c, d]
        
        metric_array[count] = gen_metric(mask, estimated_real)
        parameter_array[count] = params
        estimated_real_array[count] = estimated_real
        count += 1

    return metric_array, parameter_array, estimated_real_array

def get_best(metric_arr, param_arr, estimated_real_arr, target_metric):
    """Retrieves the parameters and estimated values associated with the highest metric from the output of our "possible_answers" function"""

    #Finds the index the highest metric
    max = metric_arr[0]
    num = abs(target_metric - max)
    for x in range(len(metric_arr)):
        numx = abs(target_metric - metric_arr[x])
        if numx < num: #metric_arr[x] > max: 
            num = numx
            max = metric_arr[x]

    index = metric_arr.index(max)

    #Returns the highest metric and its corresponding parameters and real values
    return max, param_arr[index], estimated_real_arr[index]

def optimize(mask, naive_vals, target_metric):
    """Attempt at hill climbing. Probably doesn't work correctly"""
    # The best fit values of (a, b, c, d) are 
    # (1.573374532385897, 4.019185691943018, -4.00502684639661, 1.359444812728886) 
    # whichwe obtained with a "in the lab" calibration process where we did
    # a curve fit.  This curve fit is described in the section "In-lab
    # calibration" in the book, and is implemented with the program
    # "do_in_lab_calibration.py"

    #Set the current array and parameters 
    checker = [0]*len(naive_vals)
    current_array = [0]*len(naive_vals)
    a, b, c, d = 0, 0, 0, 0
    while current_array == checker:
        current_array, a, b, c, d = gen_estimated(naive_vals)
    
    #parameters = [0, 0, 0, 0]
    parameters = [a, b, c, d]

    #Gets the current metric, which is the highest peak of the cross correlation
    #between the mask and naive minus the second highest peak
    current_metric = gen_metric(mask, current_array)
    print("Current: ", current_metric)
    #Generate 200 possible answers (possible "real values"). This comes with the metrics and parameters for each
    metrics, params, estimated_real_list = possible_answers(mask, naive_vals)

    #Get the best metric and its associating parameters and estimated real values from our list of possible answers 
    better_metric, better_params, better_array = get_best(metrics, params, estimated_real_list, target_metric)
    print("Better: ", better_metric)
    
    #Essentially repeats what is done above until none of the newly generated possible real values has a higher 
    #metric than the current
    while current_metric > better_metric:
        print("!")
        
        current_array = better_array
        current_metric = better_metric
        parameters = better_params

        metrics, params, estimated_real_list = possible_answers(mask, naive_vals)

        better_metric, better_params, better_array = get_best(metrics, params, estimated_real_list, target_metric)
        print(better_metric)

    #Returns our parameters, metric, and estimated real values
    return parameters, current_metric, current_array

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
