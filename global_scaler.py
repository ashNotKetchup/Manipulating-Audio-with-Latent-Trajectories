import numpy as np
import math
import warnings

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
