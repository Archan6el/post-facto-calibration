#! /usr/bin/env python3

#import our module
import postfactolib

# mask_real is something we feel is the real mask
mask_real = [1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1,
             0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0,
             1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0,
             1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1]

# some parameters of the run -- Based on the measurements of the proportional counter on the HETE-2
n_photons = 100
noise_fraction = 0.0
distortion = 1        # limits are [0, 1]

mask = mask_real
det_height_cm = 18.74
mask_width_cm = 20.8
det_width_cm = 12.0
det_pix_size_cm = 0.0495
n_det_pix = int(mask_width_cm / det_pix_size_cm + 0.5) 
mask_det_offset_cm = (mask_width_cm - det_width_cm) / 2
theta_deg = 0              # incidence angle, 0 if perpendicular

def main():

    #Initialize our constructor using our run parameters
    pfl = postfactolib.calibrate(mask, mask_width_cm, mask_det_offset_cm, det_width_cm, det_height_cm, n_det_pix, n_photons, theta_deg)
    
    #NOTE - This process of generating a readout and getting our real position list and naive position list
    #is not required. You can curve fit with hard coded position lists as well. Just create the 2 arrays and pass them
    #through the plot_curvefit function
    
    #Generate a detector read out which is based on the mask, angle, and number of photons.
    #You can also add distortion if you wish. Do this by passing in the argument.
    #For example with our distortion variable: pfl.gen_readout(distortion_dial=distortion)
    pfl.gen_readout(postfactolib.calibration_none, None, distortion_dial=distortion)

    #After generating our readout, we can then fetch our real and naive positions
    det_poslist_naive = pfl.get_naive_poslist()
    det_poslist_real = pfl.get_real_poslist()

    '''
    for x in range(len(det_poslist_naive)):
        det_poslist_naive[x] = (det_poslist_naive[x] - mask_det_offset_cm) / det_width_cm
        det_poslist_real[x] = (det_poslist_real[x] - mask_det_offset_cm) / det_width_cm
    '''

    #print(det_poslist_real)
    #print(det_poslist_naive)
    
    #Plot our data, curve fit the data, and print the parameters

    #Executing while verbose will print our parameters.
    #The function by default curve fits with multiple functions, so it will print
    #parameters associated with linear, cubic, etc functions as well
    pfl.plot_curvefit(det_poslist_real, det_poslist_naive, fit='sigmoid', verbose=True) 

if __name__ == '__main__':
    main()
