# Test API. These are also prototypes for a faceade class wrapping all these Max(or other ext) API wrappers within public functions
# -------------------------------
def get_dict():
    """
    Test the creation and manipulation of a random dictionary in Max using the API.
    Outputs a serialized JSON string for use in Max.
    """
    latent_json_str = tester.to_json()
    log(f"Output-ing JSON string of length: {len(latent_json_str)}")
    api.out(latent_json_str)

def set_dict(latent_json_str: str):
    """
    Parse a serialised JSON string coming from Max and output as numpy array.
    """
    log("Parsing latent JSON string back to array...")
    tester.from_json(latent_json_str)
    arr = tester.get_latent_representation()
    log(f"Wrote internally, a latent array of shape: {arr.shape}, max {arr.max()}, min {arr.min()}")
    # global latent_array
    # latent_array = json_to_latent_array(latent_json_str)


    # Tests
def test_buffer_get_input(sample_file: str = "jongly.aif"):
    # Load audio to a buffer by name, pass input of the format: call test_buffer_get_input 'jongly.aif'

    buffer_manager = BufferManager() # create instance
    buffer_manager.load_buffer(sample_file, "input")
    input_audio = buffer_manager.get_input_buffer()
    log(f"Loaded input buffer of shape {input_audio.shape[0]}, max {input_audio.max()}, min {input_audio.min()}")


def test_buffer_set_output():
    # Should write a buffer into the max buffer named 'output'
    # generate a test array
    t = np.linspace(0, 1, 44100, endpoint=False, dtype=np.float32)
    test_array = 0.5 * np.sin(t * 2 * math.pi * 440)  # 440 Hz sine wave
    buffer_manager = BufferManager() # create instance
    log("Writing test buffer to 'output'")
    buffer_manager.write_buffer("output", test_array)