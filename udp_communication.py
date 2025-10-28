import socket
import threading
from dummy_json import dummy_json
from load_audio import BufferManager
from load_generative_model import Model, LatentRepresentation
import numpy as np
import json
from max_utils import MaxUtils

# --- CONFIG ---
MAX_IP = "127.0.0.1"
MAX_SEND_PORT = 9999   # Max is listening here
PYTHON_LISTEN_PORT = 9997  # Python listens here
# large buffer to accommodate larger UDP datagrams (max UDP payload ~65507)
BUFFER_SIZE = 65535

# --- SETUP DEPENDENCIES/CLASSES –––
latent_representation = LatentRepresentation()
gen_model = Model('generative_models/percussion.ts')
audio_handler = BufferManager()

# --- SETUP SOCKET ---
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((MAX_IP, PYTHON_LISTEN_PORT))
print(f"Listening on {MAX_IP}:{PYTHON_LISTEN_PORT}, Sending on: {MAX_SEND_PORT}")

def listen():
    while True:
        data, addr = sock.recvfrom(BUFFER_SIZE)
        try:
            msg = data.decode("utf-8")
        except UnicodeDecodeError:
            msg = data.decode("utf-8", errors="replace")
        parts = msg.split(maxsplit=1)
        prepend = parts[0] if parts else ""
        message = parts[1] if len(parts) > 1 else ""
        print(f"From Max {addr}: {prepend}")
        if (prepend == 'sending_latent'):
            clean_message = message.strip()    
#            print(f'Message = {clean_message}')
            try:
                # parse and apply latent representation, then generate audio and set buffer
                latent_representation.from_json(clean_message)
                audio_out = gen_model.decode(*latent_representation.get_latent_representation())
                audio_handler.set_output_buffer(audio_out)
            except Exception as e:
                print(f"Error handling latent message: {e}")
#            print(f'Message = {msg}')
#            latent_representation.from_json(clean_message)
#            audio_out = gen_model.decode(*latent_representation.get_latent_representation())
#            audio_handler.set_output_buffer(audio_out)
        # Example reply
#        reply = f"Python got: {msg}"
#        sock.sendto(reply.encode("utf-8"), (MAX_IP, MAX_SEND_PORT))

#        loading audio
        if (prepend == 'load_audio'):
            audio_path = message.strip('""')
            try:
                audio_handler.load_buffer(audio_path)
                audio_in = audio_handler.get_input_buffer()
                audio_handler.set_output_buffer(audio_in)
                assert isinstance(audio_in,np.ndarray), 'internal buffer has wrong type, not numpy'
            except Exception as e:
                print(f"Error loading audio: {e}")

            try:
                # Encode
                latent_vector, latent_text  = gen_model.encode(audio_in)
                assert isinstance(latent_vector,np.ndarray) and isinstance(latent_text, str), 'encodings not right type'
                # print('Encoded latent vector has size: ', latent_vector.shape)
                
                # to json
                latent_representation.set_latent_representation(latent_vector,latent_text)

                # Set scale (only do this once per model + audio combo)
                latent_representation.fit(output_range=4)
                
                latent_json = latent_representation.to_json() 
                assert isinstance(latent_json, str), 'json not a string'

            except Exception as e:
                print(f"Error encoding latent: {e}")

            print(latent_json)
            sock.sendto(latent_json.encode("utf-8"), (MAX_IP, MAX_SEND_PORT))
        
threading.Thread(target=listen, daemon=True).start()

# --- SENDER LOOP (optional) ---
print("Type messages to send to Max (Ctrl+C to quit)")
try:
#    sock.sendto(dummy_json.encode("utf-8"), (MAX_IP, MAX_SEND_PORT))
    while True:
        msg = input("> ")
        sock.sendto(msg.encode("utf-8"), (MAX_IP, MAX_SEND_PORT))
except KeyboardInterrupt:
    print("Exiting.")
    sock.close()
