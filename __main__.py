import sys
import numpy as np

from load_generative_model import latent_model
from load_audio import load_audio
from IPython.display import Audio, display
from gui import interface
import librosa as li

# Working with trajectories in latent audio models

# -------------------------------
# GLOBAL VARIABLES
# -------------------------------
debug: bool = True
use_string: bool = False
# latent_array = np.array([])  # Placeholder for latent array

# api.post(f'python is located at {sys.executable}')
# api.post('globals and imports are coo')




# In[1] # Pick Model:
model_name: str = 'percussion'
model_location:str = 'generative_models/'+model_name+'.ts'

# control_model_location = 'control_models/vae_scripted_model.ts'
# model = latent_model([model_location])
# sr: int =44100


# # In[2] # Pick Audio Sample:
# # Thanks to freesound user Jovica (https://freesound.org/people/Jovica/sounds/2267/)
# audio_location: str = 'audio/2267__jovica__90-bpm-attack-loop-3-hihats-mastered-16-bit.wav'
# audio, sr = li.load(audio_location,sr=44100)
# Audio(audio, rate=sr)

# In[3] # Mess with Latent Space:
# user_interface = interface(model,audio)   
# display(user_interface.app)


def readInput():
	text = ''
	for word in sys.argv[1:]:
		text += f'{word}'
	return text

def translate():
	text = readInput()

	if text =='':
		return

	print('eyo that worked!')

if __name__ == '__main__':
	translate()
