import librosa as li
import numpy as np
from typing import List
from functools import reduce
from logger import log
import os
import soundfile as sf


class BufferManager:
    """
    Manages reading and writing audio buffers to and from Max.
    TODO: Pad input before encoding, and crop again on export (because latents arent giving same length)
    TODO: Allow for cropping and reinsertion

    Private variables:
        - _input_buffer (np.ndarray): Last read input buffer.
        - _output_buffer (np.ndarray): Last written output buffer.

    Public methods:
        - load_buffer(sample_file: str, buffer_name: str = 'input') -> np.ndarray:
            Loads an audio file into memory as a NumPy array.
        - write_buffer(buffer_name: str = 'output', data: np.ndarray = None):
            Writes a NumPy array to an audio file.
        - get_input_buffer() -> np.ndarray:
            Returns the last read input buffer.
        - set_input_buffer(data: np.ndarray):
            Sets and updates the last read input buffer.
        - get_output_buffer() -> np.ndarray:
            Returns the last written output buffer.
        - set_output_buffer(data: np.ndarray):
            Sets and updates the last written output buffer.
    """
    def __init__(self):
        self._input_audio = None
        self._output_audio = None

    def load_buffer(self, sample_file_path: str, buffer_name: str = 'input') -> bool:
        """
        Load an audio file into a numpy array

        Parameters:
        file_path (str): path to the audio file

        Returns:
        A np array of shape [], where...

        Test:
        test_loading
        """

        try:
             x, sr = li.load(sample_file_path,sr=44100)
             self.set_input_buffer(np.asarray(x))
        except Exception as e:
            log(f"Error creating buffer '{buffer_name}' from file '{sample_file_path}': {e}")
        log(f"created buffer name: '{buffer_name}' sample_file: '{sample_file_path}, with shape: '{self._input_audio.shape}'")
        
        return True


    def write_buffer(self, audio_array:np.array = None, buffer_name: str = 'output', folder_name:str = (os.getcwd()+'/audio')) -> bool:
        """Write an audio np:array to a named file. TODO: Probably needs a better name
        
        Test:
        test_writing

        """

        sr = 44100
        audio = audio_array

        filepath = os.path.join(folder_name, buffer_name) + ".wav"
        sf.write(filepath, audio, sr)
        # log(f"created buffer name: '{buffer_name}' w/ framecount: {audio.shape[0]}")
        return True


    def get_input_buffer(self) -> np.ndarray:
        """Get the last read input buffer."""
        return self._input_audio
    
    def get_output_buffer(self) -> np.ndarray:
        """Get the last written output buffer."""
        return self.output_buffer

    def set_input_buffer(self, data: np.ndarray) -> bool:
        """Set the last read input buffer. Update max buffer too"""
        self._input_audio = data # store internally
        # self.write_buffer("input", self._input_audio) # update max buffer
        return True


    def set_output_buffer(self, data: np.ndarray) -> bool:
        """Set the last written output buffer. Internally, and optionally tell max to update"""
        self.output_buffer = data
        self.write_buffer(self.output_buffer, "output")
        return True