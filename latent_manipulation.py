# In[0] # Load Utility Functions:
from load_generative_model import latent_model, load_audio
from IPython.display import Audio, display
from gui import interface
import librosa as li

# Working with trajectories in latent audio models

# In[1] # Pick Model:
model_name: str = 'percussion'
model_location:str = 'generative_models/'+model_name+'.ts'
model = latent_model([model_location])
sr: int =44100

# In[2] # Pick Audio Sample:
audio_location: str = 'audio/SOPHIE_fx_27.wav'
# audio_location: str = 'audio/THE_KOUNT_tambourine_loop_loose_77.wav'
audio, sr = li.load(audio_location,sr=44100)
Audio(audio, rate=sr)

# In[3] # Mess with Latent Space:
user_interface = interface(model,audio)   
display(user_interface.app)
# %%
