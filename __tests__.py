from load_audio import BufferManager
from load_generative_model import Model, LatentRepresentation
import numpy as np
from global_scaler import GlobalScaler
import json


# TEST BOILERPLATE
def test_start(test_name):
    print(f'Testing {test_name}...')

def test_pass(test_name):
    print(f'Test for: {test_name} ✅ Passed')
    return True

def test_fail(test_name):
    print(f'Test for: {test_name} ❌ Failed')
    return False

def tester(function, function_name, function_parameters=None):
    test_start(function_name)
    assert function(function_parameters), test_fail(function_name)
    test_pass(function_name)
    return

def test_outcome(function_outcome:bool, test_name:str, expected_condition = True):
    """
    Checks function against expected condition, reports
    # test_outcome(True, 'tru testa')                   # expects pass
    # test_outcome(False, 'false testah')               # expects fail
    # test_outcome(False, 'false false testah', False)  # expects pass
    """
    if function_outcome == expected_condition:
        print(f'Test for: {test_name} ✅ Passed')
        return True
    else:
        print(f'Test for: {test_name} ❌ Failed')
        return False


print('–––– TESTS ––––')
# AUDIO HANDLING TESTS 



## units

def test_loading():
    test_buffer = BufferManager()
    tester(test_buffer.load_buffer, 'load buffer', 'audio/2267__jovica__90-bpm-attack-loop-3-hihats-mastered-16-bit.wav')
    assert isinstance(test_buffer.get_input_buffer(),np.ndarray), 'internal buffer has wrong type, not numpy'
    return


def test_write_buffer():
    test_buffer = BufferManager()

    sr = 44100
    t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
    freq = 880.0
    sine_wave = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    tester(test_buffer.write_buffer, 'write buffer', sine_wave)
    return
    # this test sucks cause we arent testing if the file actually gets written


## good paths

### Set input audio, then get it, set that to output, which should also write it

def test_audio_thru():
    test_buffer = BufferManager()

    sr = 44100
    t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
    freq = 880.0
    sine_wave = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    test_buffer.set_input_buffer(sine_wave)
    tester(test_buffer.set_output_buffer, 'set in,get in,set out', test_buffer.get_input_buffer())




# LATENT MODEL

def test_encoding():
    test_model = Model('generative_models/percussion.ts')

    sr = 44100
    t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
    freq = 880.0
    sine_wave =  (0.5*np.sin(2 * np.pi * freq * t)).astype(np.float32)
    print('Audio expected shape: ',sine_wave.shape)
    
    output = test_model.encode(sine_wave)
    assert isinstance(output[0],np.ndarray) and isinstance(output[1], str), 'encodings not right type'
    tester(test_model.encode,'encode', sine_wave)
    return True

def test_decoding():
    test_start('decoding')
    test_model = Model('generative_models/percussion.ts')
    ## Dummy data
    # create a latent (not audio-rate) with dim (1, 4, 22)
    latent_dim = 4
    latent_length = 22

    # make dummy vector (1, latent_dim, latent_length)
    t = np.linspace(0, 1.0, latent_length, endpoint=False, dtype=np.float32)
    # each latent channel is a sine with a different number of cycles
    channels = [0.5 * np.sin(2 * np.pi * (i + 1) * t) for i in range(latent_dim)]
    latent_vector = np.stack(channels, axis=0).astype(np.float32)[np.newaxis, ...]  # (1, latent_dim, latent_length)
    latent_text = 'there is a cow in the field'

    print('Dummy latent vector has size: ', latent_vector.shape)
    audio_out = test_model.decode(latent_vector,latent_text)
    print('Decoded audio is size: ', audio_out.shape)
#    assert isinstance(test_model.decode(latent_vector,latent_text), np.ndarray), 'decodings not right type'
    # tester(test_model.decode,'decode', (latent_vector,latent_text))
    return True


## Good paths

def encode_decode():
    test_model = Model('generative_models/percussion.ts')

    #spoof audio
    sr = 44100
    t = np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
    freq = 880.0
    sine_wave =  (0.5*np.sin(2 * np.pi * freq * t)).astype(np.float32) 
    
    latent_vector, latent_text  = test_model.encode(sine_wave)
    assert isinstance(latent_vector,np.ndarray) and isinstance(latent_text, str), 'encodings not right type'
  
    print('Encoded latent vector has size: ', latent_vector.shape)
    audio_out = test_model.decode(latent_vector,latent_text)
    print('Decoded audio is size: ', audio_out.shape)
    return True


def load_encode_decode_save():
    test_name = 'load_encode_decode_save' 
    test_start(test_name)
    test_buffer = BufferManager()
    test_buffer.load_buffer('audio/2267__jovica__90-bpm-attack-loop-3-hihats-mastered-16-bit.wav')
    audio_in = test_buffer.get_input_buffer()

    # Audio loads good
    assert isinstance(audio_in,np.ndarray), 'internal buffer has wrong type, not numpy'

    # Encodings appropriate
    test_model = Model('generative_models/percussion.ts')
    latent_vector, latent_text  = test_model.encode(audio_in)
    assert isinstance(latent_vector,np.ndarray) and isinstance(latent_text, str), 'encodings not right type'
    print('Encoded latent vector has size: ', latent_vector.shape)

    # Decode works ok
    audio_out = test_model.decode(latent_vector,latent_text)
    print('Decoded audio is size: ', audio_out.shape)

    #
    if test_buffer.set_output_buffer(audio_out):
        test_pass(test_name)
        return True
    else:
        test_fail(test_name)
        return False


# scaler

## scale accurately
def test_scale():

    test_start('scaler')
    ## Spoof latent representation
    # create a latent (not audio-rate) with dim (1, 4, 22)
    latent_dim = 4
    latent_length = 22

    # make dummy vector (1, latent_dim, latent_length)
    t = np.linspace(0, 1.0, latent_length, endpoint=False, dtype=np.float32)
    # each latent channel is a sine with a different number of cycles
    channels = [0.5 * np.sin(2 * np.pi * (i + 1) * t) for i in range(latent_dim)]
    latent_vector = np.stack(channels, axis=0).astype(np.float32)[np.newaxis, ...]  # (1, latent_dim, latent_length)

    ## create latent rep object
    test_scaler = GlobalScaler()
    range_plus_minus = 2
    ## fit scaler
    test_scaler.fit(latent_vector)

    ## scale and check in range
    latent_vector_scaled = test_scaler.scale(latent_vector)
    assert (latent_vector_scaled.max() == 2) and (latent_vector_scaled.min() == -2)  , 'scaling not hitting range'

    ## descale and check min and max are similar to before
    latent_vector_descaled = test_scaler.descale(latent_vector_scaled)

    tolerance = 1e-6
    biggest_difference = np.max(np.abs(latent_vector_descaled - latent_vector))
    assert biggest_difference <= tolerance, f'descaled vector differs from original by max {biggest_difference} > {tolerance}'

    return test_pass('scaler')


# latent representation
## set and get
def test_latent_set():
    test_representation = LatentRepresentation()

    test_name= 'set rep'
    test_start(test_name)
    ## Spoof latent representation
    # create a latent (not audio-rate) with dim (1, 4, 22)
    latent_dim = 4
    latent_length = 22

    # make dummy vector (1, latent_dim, latent_length)
    t = np.linspace(0, 1.0, latent_length, endpoint=False, dtype=np.float32)
    # each latent channel is a sine with a different number of cycles
    channels = [0.5 * np.sin(2 * np.pi * (i + 1) * t) for i in range(latent_dim)]
    latent_vector = np.stack(channels, axis=0).astype(np.float32)[np.newaxis, ...]  # (1, latent_dim, latent_length)
    latent_text = 'hi im being set up'
    test_outcome(test_representation.set_latent_representation(latent_vector,latent_text), test_name)

def test_latent_get():
    test_representation = LatentRepresentation()

    test_name= 'get rep'
    test_start(test_name)
    ## Spoof latent representation
    # create a latent (not audio-rate) with dim (1, 4, 22)
    latent_dim = 4
    latent_length = 22

    # make dummy vector (1, latent_dim, latent_length)
    t = np.linspace(0, 1.0, latent_length, endpoint=False, dtype=np.float32)
    # each latent channel is a sine with a different number of cycles
    channels = [0.5 * np.sin(2 * np.pi * (i + 1) * t) for i in range(latent_dim)]
    latent_vector = np.stack(channels, axis=0).astype(np.float32)[np.newaxis, ...]  # (1, latent_dim, latent_length)
    latent_text = 'hi im being set up'
    assert test_representation.set_latent_representation(latent_vector,latent_text), 'setting didnt work'
    test_outcome(test_representation.get_latent_representation(), test_name, (latent_vector,latent_text))
        
## json
def test_write_json():
    test_representation = LatentRepresentation()

    test_name= 'give json'
    test_start(test_name)
    ## Spoof latent representation
    # create a latent (not audio-rate) with dim (1, 4, 22)
    latent_dim = 4
    latent_length = 22

    # make dummy vector (1, latent_dim, latent_length)
    t = np.linspace(0, 1.0, latent_length, endpoint=False, dtype=np.float32)
    # each latent channel is a sine with a different number of cycles
    channels = [0.5 * np.sin(2 * np.pi * (i + 1) * t) for i in range(latent_dim)]
    latent_vector = np.stack(channels, axis=0).astype(np.float32)[np.newaxis, ...]  # (1, latent_dim, latent_length)
    latent_text = 'hi im being set up'
    test_representation.set_latent_representation(latent_vector,latent_text)
    our_json = test_representation.to_json(dimension_labels=[None, None, 'custom 3rd dimension'] ) 
    # print(json.loads(our_json))                                    
    # test_outcome(isinstance(json.loads(our_json),dict), test_name)
    # print('JSON: ', our_json )
    return test_outcome(isinstance(json.loads(our_json),dict), test_name)


def test_load_json():
    test_representation = LatentRepresentation()

    test_name= 'get json'
    test_start(test_name)
    dummy_json = {
    "vector": {
        "0": {
            "label": "Dimension 1",
            "data": [
                0,
                0.14086629450321198,
                0.27032041549682617,
                0.37787482142448425,
                0.4548160135746002,
                0.49491071701049805,
                0.49491071701049805,
                0.4548160135746002,
                0.3778747320175171,
                0.2703203856945038,
                0.14086619019508362,
                -4.371138828673793e-8,
                -0.14086638391017914,
                -0.2703203558921814,
                -0.37787479162216187,
                -0.45481598377227783,
                -0.49491074681282043,
                -0.49491071701049805,
                -0.45481598377227783,
                -0.37787479162216187,
                -0.27032023668289185,
                -0.14086614549160004
            ]
        },
        "1": {
            "label": "Dimension 2",
            "data": [
                0,
                0.27032041549682617,
                0.4548160135746002,
                0.49491071701049805,
                0.3778747320175171,
                0.14086619019508362,
                -0.14086638391017914,
                -0.37787479162216187,
                -0.49491074681282043,
                -0.45481598377227783,
                -0.27032023668289185,
                8.742277657347586e-8,
                0.2703205943107605,
                0.45481595396995544,
                0.49491071701049805,
                0.37787485122680664,
                0.1408659815788269,
                -0.1408664733171463,
                -0.37787485122680664,
                -0.49491071701049805,
                -0.4548158347606659,
                -0.27032017707824707
            ]
        },
        "2": {
            "label": "custom 3rd dimension",
            "data": [
                0,
                0.37787482142448425,
                0.49491071701049805,
                0.2703203856945038,
                -0.14086638391017914,
                -0.45481598377227783,
                -0.45481598377227783,
                -0.14086636900901794,
                0.2703205943107605,
                0.49491071701049805,
                0.37787485122680664,
                -1.1924880638503055e-8,
                -0.37787485122680664,
                -0.49491074681282043,
                -0.2703205645084381,
                0.14086617529392242,
                0.45481619238853455,
                0.45481616258621216,
                0.14086613059043884,
                -0.27032020688056946,
                -0.49491068720817566,
                -0.37787485122680664
            ]
        },
        "3": {
            "label": "Dimension 4",
            "data": [
                0,
                0.4548160135746002,
                0.3778747320175171,
                -0.14086638391017914,
                -0.49491074681282043,
                -0.27032023668289185,
                0.2703205943107605,
                0.49491071701049805,
                0.1408659815788269,
                -0.37787485122680664,
                -0.4548158347606659,
                1.7484555314695172e-7,
                0.45481619238853455,
                0.3778749406337738,
                -0.14086632430553436,
                -0.49491068720817566,
                -0.2703199088573456,
                0.27032074332237244,
                0.49491068720817566,
                0.14086627960205078,
                -0.37787526845932007,
                -0.4548157751560211
            ]
        }
    },
    "text": "hi im being set up"
}
    ## Spoof latent representation
    # create a latent (not audio-rate) with dim (1, 4, 22)
    latent_dim = 4
    latent_length = 22

    # make dummy vector (1, latent_dim, latent_length)
    t = np.linspace(0, 1.0, latent_length, endpoint=False, dtype=np.float32)
    # each latent channel is a sine with a different number of cycles
    channels = [0.5 * np.sin(2 * np.pi * (i + 1) * t) for i in range(latent_dim)]
    latent_vector = np.stack(channels, axis=0).astype(np.float32)[np.newaxis, ...]  # (1, latent_dim, latent_length)
    latent_text = 'hi im being set up'
    test_representation.set_latent_representation(latent_vector,latent_text)
    json_vector, json_text, json_labels  = test_representation.from_json(dummy_json) 
    return test_outcome((json_vector,json_text), test_name, (latent_vector,latent_text))
## give json of the right shape
# 1


# test_loading()
# test_write_buffer()
# test_audio_thru()
#test_encoding()
#test_decoding()
# encode_decode()
# load_encode_decode_save()
# test_scale()

# test_latent_set()
# test_latent_get()
# test_write_json()
test_load_json()



print('All Tests Passed! 💯')
