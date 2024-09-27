import torch
import librosa as li
import numpy as np


class latent_model:
    """
    Container for generative models which use a latent representation to encode and decode audio

    Parameters:
    file_path (str): path to model

    Methods:

    """
    def __init__(self, model_location:str) -> None:
        self.__model = self.__load_model(model_location)
        self.__model.double()
        self.number_of_dimensions: int = self.__get_shape()[1]

    def __get_shape(self):
         """Generate random audio to get the shape of its latent representation
         """
         return [0,4]
    
    #TODO: put this back in. chop audio in half. figure out double/float issue. 
        #  duration_in_samps = 2*44100  # Duration in samples 2 seconds * sample rate

        #  generated_audio = np.random.uniform(low=-1.0, high=1.0, size=int(duration_in_samps)) # Generate noise between -1.0 and 1.0
         
        #  latent_rep = self.encode_audio(generated_audio) # Encode audio into latent representation

        #  return latent_rep.size()
            
    

    def __load_model(self, model_location: str):
        """
        Loads a model from a location. Model must be a '.TS' file, and must have 'encode' and 'decode' methods

        Parameters:
        file_path (str): path to model

        Returns:
        PyTorch model
        """
        return torch.jit.load(model_location) # Load Model

    def encode_audio(self, audio_array: np.ndarray) -> torch.Tensor:
        """
        Encode an audio file into its latent representation

        Parameters:
        audio_array (x): numpy array of a loaded audio file

        Returns:
        Latent representation (y) a tensor of shape [], where
        """
        # check type, convert np to torch, so we can take both....
        audio_torch: torch.Tensor = torch.from_numpy(audio_array).reshape(1,1,-1) #.double()
        with torch.no_grad():
            y: torch.Tensor = self.__model.encode(audio_torch)

        return y

    def decode_audio(self, latent_representation: torch.Tensor) -> np.ndarray:
        """
        Decode a latent representation into audio

        Parameters:
        latent_representation: tensor representing path in latent space

        Returns:
        Audio file (x_hat): np array of shape [], where
        """
        with torch.no_grad():
            decoded_audio=self.__model.decode(latent_representation).numpy().reshape(-1)
            decoded_audio = decoded_audio[:len(decoded_audio) // 2]
            return decoded_audio


def load_audio(file_path: str) -> np.ndarray:
        """
        Load an audio file into a numpy array

        Parameters:
        file_path (str): path to the audio file

        Returns:
        A np array of shape [], where...
        """

        x, sr = li.load(file_path,sr=44100)
        return x