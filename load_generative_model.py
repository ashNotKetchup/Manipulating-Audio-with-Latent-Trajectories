import torch
import librosa as li
import numpy as np
from typing import List, Union, Tuple
from functools import reduce
import json
import math
from global_scaler import GlobalScaler, TimeCompressor

# -------------------------------
# AI model class
# -------------------------------

class Model:
    """
    Manage internal state of AI model, load by name, expose encode/decode methods if they exist. Raise errors if they dont. Also give info about the nature of the encoding (eg dimensions, mode, etc)
    
    TODO: make everything operate at higher precision. Currently super noisy and that is probably because of low resoulution latents

    internal variables:
    Q: do I assign and return variables as in the others, or do I take a functional approach and just spit out the result when asked?
    A (ANH): No, just pass the representations to be handled elsewhere
    - model_path (str): Name and path of the model to use.
    - model: the loaded model, this should have the encode and decode function
    - dimension_count: the number of latent dimensions in the latent dimension section
    public functions:
    - encode : takes audio as a numpy array, returns a numpy array of latent embeddings (or text...)
    - decode: takes latent embeddings as a numpy array, returns audio as numpy array
    - get_info: gives information about the shape of the model
    """
    def __init__(self, model_path: Union[str, List[str]]):
        if isinstance(model_path, str):
            self.model_paths: List[str] = [model_path] # make sure everything is a list
        else:
            self.model_paths = model_path

        self.models = [self.__load_model(model_location) for model_location in self.model_paths] # array of loaded models

        self.dimension_count = None

    def __load_model(self, model_location: str):
        """
        Loads a model from a location. Model must be a '.TS' file, and must have 'encode' and 'decode' methods

        Parameters:
        file_path (str): path to model

        Returns:
        PyTorch model with encode and decode
        """
        loaded_model = torch.jit.load(model_location) # Load Model
        # loaded_model.double() # change weights type
        if loaded_model.encode == None:
            return 'Model Needs Encode'
        if loaded_model.decode == None:
            return 'Model Needs Decode'
        return loaded_model

    def encode(self, audio_array:np.ndarray, use_text:bool = False) -> Tuple[np.ndarray, str]:
        """
        Encode an audio file into its latent representation

        Parameters:
        audio_array (x): numpy array of a loaded audio file
        use_text: bool declaring weather or not to recalculate text embedding

        Returns:
        Latent representation (y) a np array of shape []
        """
        # check type, convert np to torch, so we can take both....
        # print(type(audio_array))
        audio_torch: torch.Tensor = torch.from_numpy(audio_array).reshape(1,1,-1) #.double()
        
        with torch.no_grad():
            # Use reduce to recursively apply the encode() method of each object
            encoding_torch:torch.Tensor = reduce(lambda acc, each_encoder: each_encoder.encode(acc), self.models, audio_torch) #does g.encode(f.encode(x)) for a list of models [f,g] and input audio_torch
    
        
        latent_vector:np.ndarray = encoding_torch.numpy(force=True)
        latent_text:str = 'I havent implemented text embeddings yet!'
        # log(f"Generated latent array of shape {latent_vector.shape}, max {latent_embedding.max()}, min {latent_embedding.min()}")
        
        return latent_vector, latent_text

    def decode(self, latent_vector:np.ndarray, latent_text:str, use_text:bool=False) -> np.ndarray:
        """
        Decode a latent representation into audio

        Parameters:
        latent_representation(z): tensor representing path in latent space

        Returns:
        Audio_file (x_hat): np array of shape [], where
        """
        latent_torch: torch.Tensor = torch.Tensor(latent_vector)

        with torch.no_grad():
            # Use reduce to recursively apply the encode() method of each object
            decoded_audio_torch: torch.Tensor = reduce(lambda z, each_decoder: each_decoder.decode(z), reversed(self.models), latent_torch) #does f.decode(g.decode(x)) for a list of models [f,g] and input laternt_representation
            # y = reduce(self.__encode, )

        decoded_audio = decoded_audio_torch.numpy(force=True) # size (1,2,length), which we don't want

        #average the two channels
        decoded_audio_mono = np.average(decoded_audio,(0,1))

        return decoded_audio_mono


        
    #     # dummy code to simulate decoding
    #     t = np.linspace(0, 1, 44100, endpoint=False, dtype=np.float32)
    #     audio_array = 0.5 * np.sin(t * 2 * math.pi * 440)  # 440 Hz sine wave

    #     return audio_array

    # # def __init__(self, model_locations:List[str]) -> None:
    # #     self.__models: list = [self.__load_model(model_location) for model_location in model_locations] # array of loaded models
    # #     self.number_of_dimensions: int = self.__get_shape()[1]
    # def __get_shape(self):
    #      """Generate random audio to get the shape of its latent representation
    #      """
    #      return [0,4]
    
    # #TODO: put this back in. chop audio in half. figure out double/float issue. 
    #     #  duration_in_samps = 2*44100  # Duration in samples 2 seconds * sample rate

    #     #  generated_audio = np.random.uniform(low=-1.0, high=1.0, size=int(duration_in_samps)) # Generate noise between -1.0 and 1.0
         
    #     #  latent_rep = self.encode_audio(generated_audio) # Encode audio into latent representation

    #     #  return latent_rep.size()
            
    

    # def decode_audio(self, latent_representation: torch.Tensor) -> np.ndarray:
    #     """
    #     Decode a latent representation into audio

    #     Parameters:
    #     latent_representation: tensor representing path in latent space

    #     Returns:
    #     Audio file (x_hat): np array of shape [], where
    #     """
    #     with torch.no_grad():
    #         # Use reduce to recursively apply the encode() method of each object
    #         decoded_audio: np.ndarray = reduce(lambda acc, each_decoder: each_decoder.decode(acc), reversed(self.__models), latent_representation).numpy() #does f.decode(g.decode(x)) for a list of models [f,g] and input laternt_representation
    #         # y = reduce(self.__encode, )

    #     decoded_audio= decoded_audio.reshape(-1) # reshape to match audio output shape
    #     decoded_audio = decoded_audio[:len(decoded_audio) // 2] # half the length, covering a bug somewhere i think
    #     return decoded_audio
        
    
class LatentRepresentation:
    """
    Manages the internal state and conversion of latent representations, including JSON serialization/deserialization,
    Stores the latent representation as a NumPy array (length x channel_count)
    can also pass json scaled to a certain range  if needed
    Public methods:
    - fit(): set the scaler
    - set_latent_representation(np.ndarray): Set the latent representation from a NumPy array.
    - get_latent_representation() -> np.ndarray: Get the current latent representation as a NumPy array.
    - to_json() -> str: Serialize the latent representation to a JSON string, with optional stringified floats.
    - from_json(str or dict): Deserialize a JSON string or dict into a latent representation
    - sanitize_fragments(parts): Clean up fragmented input from Max for proper JSON parsing.
    """
    def __init__(self, buffer_manager=None, model_manager=None, logger=None):
        self.buffer_manager = buffer_manager
        self.model_manager = model_manager
        self.logger = logger

        self._audio_input = None
        self._audio_output = None
        self._latent_vector = None
        self._latent_text = None
        self._labels = None
        self._scaler = GlobalScaler()
        self._time_compressor = TimeCompressor()
        self.json_value_range = 1

    # # -------------------------------
    # # Internal helpers
    # # -------------------------------
    # def sanitize_fragments(self, parts):
    #     cleaned = []
    #     for x in parts:
    #         if isinstance(x, tuple) and len(x) == 1 and isinstance(x[0], (int, float)):
    #             cleaned.append(str(x[0]))
    #             cleaned.append(",")
    #         else:
    #             cleaned.append(str(x))
    #     s = "".join(cleaned)
    #     s = s.replace("][", "],[")
    #     return s

    def fit(self, output_range: float = 2):
        """
        fits a scaler to the currently held latent representation. 
        This scaler is applied to json input and output only.
        """ 
       
        ## fit scaler
        self._scaler.fit(self._latent_vector)
        self.json_value_range = output_range

        ## Time compression
        # self._time_compressor.fit(self._latent_vector)
        

    
    # -------------------------------
    # JSON serialization
    # -------------------------------
    def to_json(self, use_string:bool = False, re_scale:bool=True, dimension_labels:list = None) -> dict:
        """
        Convert latent array (length x channel_count) into JSON string containing:
        - latent_vector: could be scaled values or stringified floats.
        - latent_text: text from latent encoding
        - dimension_labels: a list of labels for the axes of the latent representation, in order. are applied until they run out
        """
        # self._scaler.fit(self._latent_representation)
        # scaled_data = self._scaler.scale(self._latent_representation) # scale latent representation to managaleable range


        if re_scale:
            latent_vector_scaled = self._scaler.scale(self._latent_vector,output_range=self.json_value_range)
            latent_vector_scaled = self._time_compressor.down_scale(latent_vector_scaled)
        else:
            latent_vector_scaled =self._latent_vector


        # TODO: implement string conversion
        if use_string:
            latent_vector = latent_vector_scaled
        else:
            latent_vector = latent_vector_scaled
        

        # Handle 3D latent arrays (batch, channels, length). 2nd dim -> index labels, values come from 3rd dim in order.
        # if latent_vector is None:
        #     latent_json = {"vector": {}, "text": self._latent_text}

        # elif getattr(latent_vector, "ndim", None) == 3:
        batches, channels, length = latent_vector.shape

        def get_label(label_list:list[str], channel:int) -> str:
            "Returns label if it exists and is not empty"
            if isinstance(label_list, (list, tuple)) and channel < len(label_list) and label_list[channel]:
                return label_list[channel]
            else:
                label = f"Dimension {channel+1}"
                return label

        # formatted_vector = {channel: {'label': label, 'data'}}
        formatted_vector = {channel: {'label': get_label(dimension_labels,channel), 
                                      'data': latent_vector[:, channel, :].reshape(-1).tolist()} for channel in range(channels)}


        latent_json = {"vector": formatted_vector, "text": self._latent_text}


        
            
        # if use_string:
        #     def str_float(x): return format(float(x), ".17g")
        #     latent_json = {
        #         "latent_representation": {
        #             f"dim{ch+1}": [str_float(val) for val in scaled_data[:, ch]]
        #             for ch in range(scaled_data.shape[1])
        #         }
        #     }

        
            # Refactor above so that it assigns only to the latent representation part of key, regardless of other bits?
        # return json.dumps(latent_json)
        return latent_json

    def from_json(self, json_in: dict, use_string:bool = False, re_scale:bool=True):
        """Deserialize JSON string into latent dictionary and set it.
        Convert a latent JSON string or dict back into a NumPy array (length x channel_count), text, and any labels given.
    
        """

        if isinstance(json_in, dict):
            loaded_json = json_in
        else:
            loaded_json = json.loads(json_in)

        print("Loaded JSON type:", type(loaded_json))
        loaded_text  = loaded_json['text']
        print("Loaded text type:", type(loaded_text), loaded_text)
        # print('text: ', loaded_text)
  
        items = sorted(loaded_json['vector'].items(), key=lambda kv: int(kv[0]))
  
        loaded_labels = [value['label'] for key, value in items]
        # print('labels: ', loaded_labels)

        loaded_data = np.array([value['data'] for key, value in items]).astype(np.float32)[np.newaxis, ...]
        # print('data: ', loaded_data.shape)
        
        if re_scale:
            loaded_data = self._scaler.descale(
                self._time_compressor.up_scale(loaded_data), output_range=self.json_value_range)

        # Store parsed values
        self._latent_vector = loaded_data
        self._latent_text = loaded_text
        self._labels = loaded_labels

        return self._latent_vector, self._latent_text, self._labels

        # ignore/comment below



        
        # # If input is not stringified floats, sanitize fragments (for Max)
        # if not use_string:
        #     json_in = self.sanitize_fragments(json_in)

        # # If input is a list of strings/fragments, join into a single string
        # if isinstance(json_in, list):
        #     log("Joining list of strings into single JSON string")
        #     log(repr(json_in))
        #     json_in = "".join(json_in)

        # # If input is a string, parse JSON (sometimes double-encoded from Max)
        # if isinstance(json_in, str):
        #     log("Parsing JSON string input...")
        #     json_in = json_in.strip()
        #     try:
        #         # Try double-decoding (addressing bs from Max API)
        #         json_in = json.loads(json.loads(json_in))
        #     except Exception as e:
        #         log(f"JSON decode error: {e}")
        #         raise

        # # TODO: STRETCH: Extract latent representation dictionary
        # # self.logger.log("from_json", {
        # #     "file": json_in.get("file_name"),
        # #     "model": json_in.get("model_name")
        # # })
        # latent_dict = json_in.get("latent_representation", {})
        # dims = sorted(latent_dict.keys())  # Ensure consistent dimension order

        # # Convert values to float and stack into array (shape: length x channel_count)
        # latent_array = np.array([[float(val) for val in latent_dict[dim]] for dim in dims]).T

        # # Descale using global scaler to recover original values
        # if self._scaler.global_min is not None and self._scaler.global_max is not None:
        #     latent_array = self._scaler.descale(latent_array)
        # else:
        #     log("Warning: Scaler not fitted yet, skipping descaling")
        # self.set_latent_representation(latent_array)
        # log(f"Converted JSON to array of shape {latent_array.shape}, max {latent_array.max()}, min {latent_array.min()}")
        # return

    # -------------------------------
    # Public getters/setters
    # -------------------------------   
    
    def set_latent_representation(self, latent_vector:np.ndarray, latent_text:str='not set yet') -> bool:
        """Set latent representation from a NumPy array."""
        self._latent_vector = latent_vector
        self._latent_text = latent_text
        return True
    
    def get_latent_representation(self) -> tuple[np.ndarray,str]:
        """Get the current latent representation as a NumPy array."""
        return self._latent_vector, self._latent_text

