import numpy as np
import math
import warnings
import scipy.signal as signal

class GlobalScaler:
    """
    Scale arrays based on a stored global min/max to ±output_range.
    """
    def __init__(self):
        self.global_min = None
        self.global_max = None
        

    def fit(self, array: np.ndarray):
        """Store min and max from array for future scaling."""
        self.global_min = np.min(array)
        self.global_max = np.max(array)
        if self.global_min == self.global_max:
            self.global_max += 1e-8  # avoid divide-by-zero
        return self

    def scale(self, array: np.ndarray, output_range: float = 1.0) -> np.ndarray:
        """
        Scale array based on stored min/max to a range of ±output_range 
        eg: a output_range=2 means range expands ±200% of original range.
        """
        if self.global_min is None or self.global_max is None:
            if self.global_min is None or self.global_max is None:
                warnings.warn("Scaler not fitted yet; returning input unchanged", UserWarning)
                return array  
        mid = (self.global_max + self.global_min) / 2
        spread = (self.global_max - self.global_min) / 2
        scaled = (array - mid) / spread  # normalize to [-1,1]
        return scaled * output_range

    def descale(self, scaled_array: np.ndarray, output_range: float = 1.0) -> np.ndarray:
        """Reverse the scaling to recover the original values."""
        if self.global_min is None or self.global_max is None:
            warnings.warn("Scaler not fitted yet; returning input unchanged", UserWarning)
            return scaled_array  
        mid = (self.global_max + self.global_min) / 2
        spread = (self.global_max - self.global_min) / 2
        array = scaled_array / output_range
        array = array * spread + mid
        return array
    
    def info(self) -> str:
        """Return a string with current min/max info."""
        return f"GlobalScaler(min={self.global_min}, max={self.global_max})"




class TimeCompressor:
    """
    Temporally compress and uncompress sequences based on comparison to a stored low-res sequence
    """

    ## Pseudo-code:
    # store init_hi_res: np.ndarray
    # store init_lo_res: np.ndarray = downsample(init_hi_res, factor)

    # get new_lo_res: np.ndarray from some input
    # calculate change_lo_res = compare(new_lo_res, init_lo_res) difference or scale
    # upsample change_lo_res to change_hi_res = upsample(change_lo_res, factor)
    # new_hi_res = init_lo_res + change_hi_res


    def __init__(self):
        self.init_hi_res = None
        self.init_lo_res = None
        self.factor = 10  # downsampling/upsampling factor
        

    def set_factor(self, factor: int):
        """Set the downsampling/upsampling factor."""
        self.factor = factor 

    # def fit(self, array: np.ndarray):
    #     """Store init array for future upsampling."""
    #     self.init_hi_res = array
    #     # print("Fitting TimeCompressor with array shape:", array.shape)
    #     self.init_lo_res = signal.decimate(array, self.factor, axis=2)
    #     return self
    
    def down_scale(self, array: np.ndarray) -> np.ndarray:
        self.init_hi_res = array
        print("Fitting TimeCompressor with array shape:", array.shape)
        self.init_lo_res = signal.decimate(array, self.factor, axis=2)
        print("Downscaled shape:", self.init_lo_res.shape)
        return self.init_lo_res

    def up_scale(self, new_lo_res: np.ndarray) -> np.ndarray:
        """
        Scale array based on stored min/max to a range of ±output_range 
        eg: a output_range=2 means range expands ±200% of original range.
        """

        # compare
        diff_lo_res = new_lo_res - self.init_lo_res # or could be ratio?
        # upsample
        diff_hi_res = signal.resample(diff_lo_res, self.init_hi_res.shape[2], axis=2) # might not apply if not periodic
        # add
        new_hi_res = self.init_hi_res + diff_hi_res
        return new_hi_res

    
    # def info(self) -> str:
    #     """Return a string with current min/max info."""
    #     return f"GlobalScaler(min={self.global_min}, max={self.global_max})"
