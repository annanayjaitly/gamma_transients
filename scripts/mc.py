import random
import numpy as np

from astropy.table import Table
from scipy import stats
from scipy.interpolate import RectBivariateSpline

################## SAMPLER #######################################

def exposure_data_maker(dataset):
    
    exposure_data = dataset.exposure.data
    exposure_data_summed_bins = np.sum(exposure_data, axis=0)
    exposure_data_summed_normed = exposure_data_summed_bins / np.sum(exposure_data_summed_bins)
    exposure_map_coords = dataset.exposure.geom.get_coord().skycoord[0]
    ra, dec = np.flip(exposure_map_coords[0].ra.degree), exposure_map_coords[:,0].dec.degree
    
    return ra, dec, np.fliplr(exposure_data_summed_normed).T


def ra_unwrapper(ra_array):
    discont = int(np.argwhere(np.diff(ra_array)<0))
    ra_unwrapped = np.concatenate((ra_array[:discont+1], ra_array[discont+1:]+360))
    return ra_unwrapped, discont

def ra_wrapper(ra_array, discont):
    ra_wrapped = np.concatenate((ra_array[:discont+1], ra_array[discont+1:]-360))
    return ra_wrapped

class Sampler(object):
    """Sampler object. To be instantiated with an (x,y) grid and a PDF
    function z = z(x,y).
    """

    def __init__(self, stacked_dataset, events_lists, m=0.99, cond=None):
        """Create a sampler object from data.
        Parameters
        ----------
        x,y : arrays
            1d arrays for x and y data.
        z : array
            PDF of shape [len(x), len(y)]. Does not need to be normalized
            correctly.
        m : float, optional
            Number in [0; 1). Used as new maximum value in renormalization
            of the PDF. Random samples (x,y) will be accepted if
            PDF_renormalized(x, y) >= Random[0; 1). Low m values will create
            more values regions of low PDF.
        cond : function, optional
            A boolean function of x and y. True if the value in the x,y plane
            is of interest.
        Notes
        -----
        To restrict x and y to the unit circle, use
        cond=lambda x,y: x**2 + y**2 <= 1.
        For more information on the format of x, y, z see the docstring of
        scipy.interpolate.interp2d().
        Note that interpolation can be very, very slow for larger z matrices
        (say > 100x100).
        """
        self.dataset = stacked_dataset
        x, y, z = exposure_data_maker(stacked_dataset)
        
        if np.any(np.diff(x)<0):
            x, self.discont = ra_unwrapper(x)
        else:
            self.discont = None
        #print(x)
        #z = np.fliplr(z)

        maxVal = np.max(z)
        z *= m/maxVal  # normalize maximum value in z to m
        
        self.m = m

        print("Preparing interpolating function")
        self._interp = RectBivariateSpline(x, y, z)  # TODO FIXME: why .transpose()?
        print("Interpolation done")
        
        print("Scanning events for dt_list \n")
        self.events_lists = events_lists
        self.dt_list = self.dt_list_maker()
        
        self._xRange = (x[0], x[-1])  # set x and y ranges
        self._yRange = (y[0], y[-1])
        
        self.t0 = random.choice(events_lists).time.datetime64[0]

        self._cond = cond
    
    def dt_list_maker(self):
    
        dt_list = np.array([])

        for counter, event_list in enumerate(self.events_lists):

                time_list = np.sort(event_list.time.datetime64)

                diff_list = np.diff(time_list) / np.timedelta64(1, 'ns')

                dt_list = np.append(dt_list, diff_list, axis = 0)

                print(f"\rScan no. {counter+1}/{len(self.events_lists)} ({100*(counter+1)/len(self.events_lists):2.2f}%)\r",end='\r')

        return np.sort(dt_list)
            
    def mc_dt_timestamp_maker(self, size):
        
        size = int(size)

        loc, scale = stats.expon.fit(self.dt_list, floc = 0)
        dt_mc = stats.expon.rvs(loc = 0, scale = scale, size = size)

        return self.t0 + np.cumsum(dt_mc)*np.timedelta64(1,'ns')

    def sample(self, size=1):
        """Sample a given number of random numbers with following given PDF.
        Parameters
        ----------
        size : int
            Create this many random variates.
        Returns
        -------
        vals : list
            List of tuples (x_i, y_i) of samples.
        """

        vals = []
        val_buffer = []

        while(len(vals) < size):

            # first create x and y samples in the allowed ranges (shift from [0, 1)
            # to [min, max))
            while(True):
                x, y = np.random.rand(2)
                
                x = (self._xRange[1]-self._xRange[0])*x + self._xRange[0]
                y = (self._yRange[1]-self._yRange[0])*y + self._yRange[0]

                # additional condition true? --> use these values
                if(self._cond is not None):
                    if(self._cond(x, y)):
                        break
                    else:
                        continue
                else:  # no condition -> use values immediately
                    break

            # to decide if the values are to be kept, sample the PDF there and
            # decide about rejection
            chance = np.random.ranf()
            PDFsample = self._interp(x, y)

            # keep or reject sample? if at (x,y) the renormalized PDF is >= than
            # the random number generated, keep the sample
            if(PDFsample >= chance):
                vals.append((x, y))
                print(f"\rSampling ({100*len(vals)/size:3.2f}% done)\r", end='\r')
            
            val_buffer.append((x,y))
        
        #print("sampling iterations ",len(val_buffer))
        
        vals = np.array(vals)
        table = Table()
        table['RA'], table['DEC'] = vals[:,0], vals[:,1]
        table['TIME'] = self.mc_dt_timestamp_maker(size)

        return table
    
    def sample_vectorized(self, size=1):
        """Sample a given number of random numbers with following given PDF.
        Parameters
        ----------
        size : int
            Create this many random variates.
        Returns
        -------
        vals : list
            List of tuples (x_i, y_i) of samples.
        """

        size_upsampled = int(size*2)
        
        ra = np.random.uniform(low = self._xRange[0], high = self._xRange[1], size = size_upsampled)
        dec = np.random.uniform(low = self._yRange[0], high = self._yRange[1], size = size_upsampled)
        radec = np.c_[ra,dec]
        
        radec_prob = self._interp.ev(ra, dec)
        
        rand_arr = np.random.uniform(0.0, self.m, size_upsampled)

        negs = np.empty(size_upsampled)
        
        negs.fill(-1)

        selector = np.where(rand_arr <= radec_prob, radec_prob, negs)
        
        accept = radec[selector >= 0]
        accept = accept[:int(size)]
        
        #print(radec[:10], '\n',len(ra), '\n',radec_prob, '\n',accept[:,0], '\n',accept[:,1])
        ra, dec = accept[:,0], accept[:,1]
        table = Table()
        table['RA'], table['DEC'] = ra, dec
        table['TIME'] = self.mc_dt_timestamp_maker(len(accept))
        
        return table[:int(size)]
