# In[0]:
import torch
from ipywidgets import VBox
# from load_generative_model import torch, encode_audio, decode_audio
import ipywidgets as widgets
from IPython.display import Audio, display, clear_output
from trajectory_canvas import trajectory_canvas
import numpy as np
from load_generative_model import latent_model # for typing hints
from typing import Callable

class interface:
    """User interface for manipulating a latent trajectory

    Typical use:
        user_interface = interface(model,audio) #Initialise the interface with a model and audio   
        display(user_interface.app) #Pass the interface to a display object

    Parameters:
        model: Generative model. Must have an encode and decode method. Depends on torch tensors as latent representations I think
        audio: An audio file, as an numpy array
    
    Attributes:
        audio_playback_canvas: Designated region for audio playback
        latent_representation: Current latent representation of audio
        dimension_dropdown: Dropdown for selecting the dimension which is displayed
        guis: list of gui objects which hold canvases and their data
        app: single unified app for all guis and one audio out. Pass to display to render

    Methods:
        refresh_audio: Re-calculates audio from latent representation
        on_dropdown_change: Updates gui on dropdown change


    """
    def __init__(self, model, audio: np.ndarray) -> None:

        # Encode Audio:
        self.model = model
        self.audio = audio
        self.latent_representation:torch.Tensor = self.model.encode_audio(audio)
        self.number_of_dimensions = self.model.number_of_dimensions
        
        # Frontend stuff
        self.audio_playback_canvas = widgets.Output()
        self.dimension_dropdown = widgets.Dropdown(
            options= range(self.number_of_dimensions), 
            description='Latent Dimension:',
            disabled=False, 
        )
        self.dimension_dropdown.observe(self.on_dropdown_change, names='value') # Attach the callback to observe the 'value' property of the dropdown

        # Make and display multiple canvases
        self.guis = [trajectory_canvas(self.latent_representation, self.refresh_audio, dimension) for dimension in range(0,self.number_of_dimensions)]
        self.app = VBox([self.dimension_dropdown, self.guis[0].canvas, self.audio_playback_canvas])
        # self.refresh_audio(None, 0,self.latent_representation)

    # TODO: add a reset dimension button

    # Callback function to refresh the canvas and create an audio object
    def refresh_audio(self, event, dimension_to_update: int, new_latent_data):
        """Re-calculates audio from latent representation

        Parameters: 
        dimension_to_update: the dimension of the latent representation which needs to be updated
        new_latent_data: Incoming data for updating
        """
        with self.audio_playback_canvas:
            clear_output()
            print(f"Rendered Audio:")

            # Generate audio buffer
            self.latent_representation[0,dimension_to_update,:] = new_latent_data
            new_audio = self.model.decode_audio(self.latent_representation)
            audio_buffer = new_audio
            
            # Create an IPython Audio object and display it
            audio_widget = Audio(audio_buffer, rate=44100, autoplay=True)
            display(audio_widget)

    # Define the callback function
    def on_dropdown_change(self,change):
        """Updates gui on dropdown change

        Parameters: 
        change: new dimension to show
        """
        self.app.children = [self.dimension_dropdown, self.guis[change['new']].canvas, self.audio_playback_canvas]

# %%
