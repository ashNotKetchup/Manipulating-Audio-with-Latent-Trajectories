from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import random
# from dummy_json import dummy_json
from load_audio import BufferManager
from load_generative_model import Model, LatentRepresentation
import numpy as np
import json

# --- SETUP DEPENDENCIES/CLASSES –––
latent_representation = LatentRepresentation()
gen_model = Model('generative_models/percussion.ts')
audio_handler = BufferManager()

def handle_request_latent(message):
    filepath = message['content']
    print("Python received file path to encode:", filepath)

    audio_path = filepath.strip('""')
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
        print('Encoded latent vector has size: ', latent_vector.shape)
        
        # to json
        latent_representation.set_latent_representation(latent_vector,latent_text)

        # Set scale (only do this once per model + audio combo)
        latent_representation.fit(output_range=4)
        
        latent_json = latent_representation.to_json() 
        assert isinstance(latent_json, dict), 'json not a dictionary'

    except Exception as e:
        print(f"Error encoding latent: {e}")

    # print(latent_json)

    dummy_json = {
        "id": random.randint(1, 10),
        "name": "Test Object",
        "tags": ["alpha", "beta", "gamma"],
        "scores": [10, 20, 30, 40],
        "position": { "x": 12.5, "y": -3.7 },
        "items": [
            {"id": 1, "label": "One"},
            {"id": 2, "label": "Two"}
        ]
    }

    return {"type": "latent", "content": latent_json}
    # latent_json}


def handle_request_audio(message):
    latent_data = message['content']
    print("Python received latent data:", latent_data)
    try:
        # parse and apply latent representation, then generate audio and set buffer
        latent_representation.from_json(latent_data)
        audio_out = gen_model.decode(*latent_representation.get_latent_representation())
        audio_handler.set_output_buffer(audio_out)
    except Exception as e:
        print(f"Error handling latent message: {e}")
#            print(f'Message = {msg}')
#            latent_representation.from_json(clean_message)
#            audio_out = gen_model.decode(*latent_representation.get_latent_representation())
#            audio_handler.set_output_buffer(audio_out)
    return {"type": "decoded", "content": "Decoded successfully"}


# --- Switch map/dictionary ---
handlers = {
    "request_latent": handle_request_latent,
    "request_audio": handle_request_audio
}


class SimpleHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        message = json.loads(body.decode())

        # init reply
        reply = {"type": "None", "content": "None"}

        # --- Switch dispatcher ---
        msg_type = message['type']
        print("Python received message of type:", msg_type)
        
        if msg_type in handlers:
            reply = handlers[msg_type](message)
        else:
            print(f"Unknown message type: {msg_type}")
            reply = {"type": "error", "content": f"Unknown message type: {msg_type}"}
        
        reply_bytes = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply_bytes)))
        self.end_headers()
        self.wfile.write(reply_bytes)

        print("Python sent reply of type:", reply['type'])


server = HTTPServer(("127.0.0.1", 5000), SimpleHandler)
print("Python server running on port 5000")
server.serve_forever()
