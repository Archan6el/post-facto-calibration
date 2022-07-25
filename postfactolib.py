import numpy as np
import matplotlib.pyplot as plt
import math
import random
from collections import OrderedDict
from scipy.optimize import curve_fit

class calibrate:

    def __init__(self, mask_param, mask_width_param, mask_det_offset, det_width, det_height, n_pix, photons, theta=None, noise_num=None):
        """Constructor for all the values needed to create the mask, detector readout, etc"""
        if theta is None:
            theta = 0

        if noise_num is None:
            noise_num = 0

        self.mask = mask_param
        self.mask_width_cm = mask_width_param
        self.mask_det_offset_cm = mask_det_offset
        self.det_width_cm = det_width
        self.det_height_cm = det_height
        self.n_det_pix = n_pix
        self.n_photons = photons
        self.angle_deg = theta
        self.noise = noise_num
        self.det_poslist_real = []
        self.det_poslist_naive = []

    def resample_mask(self, det_readout):
        """Resamples the mask based on the length of the detector readout"""
        resampled_mask = np.zeros(len(det_readout))
        for resampled_ind in range(len(det_readout)):
            mask_ind = int(resampled_ind * len(self.mask) / len(det_readout))
            val = self.mask[mask_ind]
            resampled_mask[resampled_ind] = val
        
        return resampled_mask

    def mask_pos_angle_consistent(self, det_pos, verbose=None):
        """Determines if the mask and detector position pairs are compatible
        with a photon incident at angle_deg
        """
        # to do this geometry, first figure out the position under the
        # mask if that photon had been straight
        perp_pos_cm = det_pos + self.det_height_cm * math.tan(self.angle_deg * math.pi / 180.0)
        if verbose:
            print('perp_pos_cm:', perp_pos_cm)

        # now find the mask slit index, so we can see if it is open or closed
        mask_slit_pos_cm = perp_pos_cm
        mask_slit_ind = int((mask_slit_pos_cm / self.mask_width_cm) * len(self.mask))
        if verbose:
            print('MASK_offset,perp,slitpos,width,ratio,slit_ind,len',
                self.mask_det_offset_cm, perp_pos_cm, 
                mask_slit_pos_cm, self.mask_width_cm, (mask_slit_pos_cm / self.mask_width_cm),
                mask_slit_ind, len(self.mask))
        if mask_slit_ind < 0 or mask_slit_ind >= len(self.mask):
            return False
        assert(mask_slit_ind >= 0 and mask_slit_ind < len(self.mask))
        valid = (self.mask[mask_slit_ind] == 1)
        if verbose:
            print(det_pos, self.angle_deg, valid)
        return valid

    def gen_readout(self, distortion_dial=0, verbose=None):
        """Given a mask and an incidence angle, simulate photons hitting the
        detector through that mask and at that angle
        """
        # #Contains our extrapolated mask
        readout = np.zeros(self.n_det_pix)
        # simulate a bunch of photons hitting the detector
        for i in range(self.n_photons):
            # hypothetical position where it might have hit.  note that we
            # measure things starting at the left end of the mask, and the
            # detector valid range is smaller than that, so we must do
            # arithmetic with those offsets to generate a valid detector
            # position
            det_pos_cm = self.mask_det_offset_cm + random.random() * self.det_width_cm
            distorted_pos_cm = self.apply_distortion(det_pos_cm, distortion_dial)

            
            if verbose:
                print('det_pos_cm:', det_pos_cm, distorted_pos_cm,
                    f'(in range of [{self.mask_det_offset_cm}, {self.mask_det_offset_cm+ self.det_width_cm}])')
            

            # first sanity check: is the detector position < 0 or > width?
            # i.e. is it out of bounds?
            assert(det_pos_cm >= self.mask_det_offset_cm
                and det_pos_cm < self.mask_det_offset_cm + self.det_width_cm)
            assert(distorted_pos_cm >= self.mask_det_offset_cm
                and distorted_pos_cm < self.mask_det_offset_cm + self.det_width_cm)
            # now see if that position is consistent with the
            # mask and the angle it's coming from
            if random.random() < self.noise:
                is_valid = True     # noise is always valid
            else:
                is_valid = self.mask_pos_angle_consistent(distorted_pos_cm, verbose)
            if is_valid:
                readout_ind = int(self.n_det_pix * distorted_pos_cm / self.mask_width_cm)
                assert(readout_ind >= 0 and readout_ind < self.n_det_pix)
                readout[readout_ind] += 1 # yay! I got a count on the detector

                self.det_poslist_real.append(det_pos_cm)
                self.det_poslist_naive.append(distorted_pos_cm)

        return readout

    def apply_distortion(self, x_real, distortion_dial=0):
        """Applies a simple distortion to the signal -- in this case something
        like x -> x + c*sin(2*pi*x / scale)

        """
        x_bad = x_real + distortion_dial * math.sin(2*math.pi*(x_real / self.mask_width_cm))
        return x_bad


    def plot_xcor(self, resampled_mask, det_readout, ccor, lag=None, save=None):
        
        """Plots cross correlation between resampled mask and detector readout"""
        # X axes of graphs
        xaxisForMask = np.arange(0, len(self.mask), 1)
        # xaxisForResampledMask = np.arange(0, mask_width_cm, mask_width_cm / len(resampled_mask))
        xaxisForResampledMask = np.arange(0, len(resampled_mask), 1)
        # xaxisForCount = np.arange(0, mask_width_cm, mask_width_cm / len(det_readout))
        xaxisForCount = np.arange(0, len(det_readout), 1)
        xaxiscc = np.arange(0, len(ccor), 1) - len(ccor) // 2

        # Everything after this is plotting our graphs
        figure, ax = plt.subplots(4, constrained_layout=True, figsize=(15, 10))
        ax[0].axes.yaxis.set_ticklabels([])
        ax[1].axes.yaxis.set_ticklabels([])

        figure.set_figheight(14)
        figure.set_figwidth(25)
        figure.suptitle(f'Mask/readout cross-correlations, theta={self.angle_deg} deg')

        ax[0].bar(xaxisForMask, ((np.zeros(len(self.mask)) + 1) - self.mask), width=1.0,
                align='edge', color='black', edgecolor='white')
        ax[0].bar(xaxisForMask, self.mask, width=1.0,
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
        ax[3].bar(xaxiscc, ccor, color='blue')
        ax[3].set_title(f'Cross Correlation (bars) - lag is {lag}')

        '''
        ax[4].stem(xaxiscc, ccor)
        ax[4].set_title(f'Cross Correlation (stems) - lag is {lag}')
        '''

        # align.xaxes(ax[1], mask_width_cm/2, ax[2], mask_width_cm/2, 0.5)

        if save:
            for suffix in ['png', 'svg']:
                ofname = f'mask_readout_ccor.{suffix}'
                print(f'saving file {ofname}')
                plt.savefig(ofname)
        plt.show()

    def plot_curvefit(self, real_pos, naive_pos, fit=None, verbose=None):
        if fit is None:
            fit = 'all'

        """Uses scipy to plot our data and curve fitting lines"""
        
        #create dictionary mapping x and y values (x-> key, y->value)
        count = 0
        dict = {}
        for i in (real_pos):
            dict[i] = naive_pos[count]
            count += 1
        
        #Sort dictionary by x so that all of our data is in order. Store keys to x and values to y
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
        linear_a = params[0]
        yfit1 = linear_a*x

        #quadratic function
        params, _  = curve_fit(func2, x, y)
        quadratic_a = params[0]
        yfit2 = quadratic_a*x**2

        #cubic function
        params, _  = curve_fit(func3, x, y)
        cubic_a, cubic_b, cubic_c = params[0], params[1], params[2]
        yfit3 = cubic_a*x**3+cubic_b*x**2+cubic_c

        #sigmoid function
        params, _  = curve_fit(sigmoid, x, y)
        sig_a, sig_b, sig_c, sig_d = params[0], params[1], params[2], params[3]
        yfitSig = (sig_a / (x - sig_b)) + sig_c + (sig_d*x)
        
        if verbose:
            print('Parameters:')
            print('Linear: ', linear_a)
            print('Quadratic: ', quadratic_a)
            print('Cubic: ', cubic_a, cubic_b, cubic_c)
            print('Sigmoid: ', sig_a, sig_b, sig_c, sig_d)
        #plotting the graphs here
        #yfit# -> the # correlates to the highest degree of exponent of the highest x
        plt.plot(x, y, 'bo', label="y-original")

        if (fit == 'linear' or fit == 'all'):
            plt.plot(x, yfit1, label="linear regression")

        if (fit == 'quadratic' or fit == 'all'):
            plt.plot(x, yfit2, label="quadratic regression")

        if (fit == 'cubic' or fit == 'all'):
            plt.plot(x, yfit3, label="cubic regression")

        if (fit == 'sigmoid' or fit == 'all'):
            plt.plot(x, yfitSig, label="sigmoid")
    
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='best', fancybox=True, shadow=True)
        plt.grid(True)
        plt.show() 

    def get_real_poslist(self):
        """Returns the real positions of where x rays hit"""
        return self.det_poslist_real

    def get_naive_poslist(self):
        """Returns the naive (distorted) positions of where x rays hit"""
        return self.det_poslist_naive

    def set_theta(self, angle):
        """Allows user to set the theta angle. Used for animating plots"""
        self.angle_deg = angle

    def set_noise(self, noise_val):
        """Allows user to set the noise value. Used for animating plots"""
        self.noise = noise_val

    def animate(self, range_vals, mode):
        """Plots cross correlation but animates either the change in angle or change in noise"""
        figure, ax = plt.subplots(4, constrained_layout=True, figsize=(20, 10))
        
        if mode == 'theta':
            range_nums = range(range_vals[0], range_vals[1])
        if mode == 'noise':
            range_nums = [x * 0.1 for x in range(range_vals[0], range_vals[1])]

        for num in range_nums: #Starts at -1 because the first angle is not animated due to slow run time. If start at 0, it'll show 1
            
            if mode =='theta':
                self.set_theta(num)
            if mode == 'noise':
                self.set_noise(num)

            # find the detector readout for a given angle
            det_readout = self.gen_readout()
            resampled_mask = self.resample_mask(det_readout)
        
            # Our cross correlation array
            # ccor = np.correlate(resampled_mask, det_readout, mode='full')
            ccor = np.correlate(resampled_mask, det_readout, mode='same')
           
            # trick to get the lag
            lag = ccor.argmax() - (len(resampled_mask) - 1)

            #Plot our data
            # X axes of graphs
            xaxisForMask = np.arange(0, len(self.mask), 1)
            # xaxisForResampledMask = np.arange(0, mask_width_cm, mask_width_cm / len(resampled_mask))
            xaxisForResampledMask = np.arange(0, len(resampled_mask), 1)
            # xaxisForCount = np.arange(0, mask_width_cm, mask_width_cm / len(det_readout))
            xaxisForCount = np.arange(0, len(det_readout), 1)
            xaxiscc = np.arange(0, len(ccor), 1) - len(ccor) // 2
            
            # Everything after this is plotting our graphs
            
            #Clears our changing plots
            ax[2].clear()
            ax[3].clear()
         

            ax[0].axes.yaxis.set_ticklabels([])
            ax[1].axes.yaxis.set_ticklabels([])

            #figure.set_figheight(14)
            #figure.set_figwidth(25)
            if mode == 'theta':
                figure.suptitle(f'Mask/readout cross-correlations, theta={num} deg')
            
            if mode == 'noise':
                figure.suptitle(f'Mask/readout cross-correlations, noise={num}')

            ax[0].bar(xaxisForMask, ((np.zeros(len(self.mask)) + 1) - self.mask), width=1.0,
                    align='edge', color='black', edgecolor='white')
            ax[0].bar(xaxisForMask, self.mask, width=1.0,
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

            figure.canvas.draw_idle()
            plt.pause(0.1)
