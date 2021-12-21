"""
Connect a resistor and LED to board pin 8 and run this script.
Whenever you say "stop", the LED should flash briefly
"""

import sounddevice as sd
import numpy as np
import scipy.signal
import timeit
import python_speech_features
import RPi.GPIO as GPIO
import tensorflow as tf

from tflite_runtime.interpreter import Interpreter

# Parameters
debug_time = 1
debug_acc = 0
led_pin = 8
word_threshold = 0.5
rec_duration = 0.5
window_stride = 0.5
sample_rate = 48000
resample_rate = 8000
num_channels = 1
num_mfcc = 16
model_path = 'wake_word_stop_lite.tflite'

# Sliding window
window = np.zeros(int(rec_duration * resample_rate) * 2)

# GPIO 
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(8, GPIO.OUT, initial=GPIO.LOW)

# Load model (interpreter)
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)

# Decimate (filter and downsample)
def decimate(signal, old_fs, new_fs):
    
    # Check to make sure we're downsampling
    if new_fs > old_fs:
        print("Error: target sample rate higher than original")
        return signal, old_fs
    
    # We can only downsample by an integer factor
    dec_factor = old_fs / new_fs
    if not dec_factor.is_integer():
        print("Error: can only decimate by integer factor")
        return signal, old_fs

    # Do decimation
    resampled_signal = scipy.signal.decimate(signal, int(dec_factor))

    return resampled_signal, new_fs

# mfcc by tensorflow
def get_mfcc(waveform):
    # sample rate = 16000, 1s have 16000 samples
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros( [16000] - tf.shape(waveform), dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    
    # FFT
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft( equal_length, frame_length=255, frame_step=128 )
    # Obtain the magnitude of the STFT.
    # spectrogram_abs shape = (124, 129)
    spectrogram_abs = tf.abs(spectrogram)
    
    # Triangular Bandpass Filters
    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = spectrogram.shape[-1]
    # generate linear_to_mel_weight_matrix 
    num_mel_bins=80
    sample_rate=16000
    lower_edge_hertz=80.0
    upper_edge_hertz=7600.0
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
    # tensordot operation
    # spectrogram_abs shape = (124, 129)
    # linear_to_mel_weight_matrix shape = (129, 80)
    mel_spectrograms = tf.tensordot( spectrogram_abs, linear_to_mel_weight_matrix, 1)
    # mel_spectrograms shape = (124, 80)
    mel_spectrograms.set_shape( spectrogram_abs.shape[:-1].concatenate( linear_to_mel_weight_matrix.shape[-1:] ) )
    
    # Log energy
    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    # log_mel_spectrograms shape = (124, 80)
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    # mfccs shape = (124, 13)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :13]  
    
    # Train shape = (124, 13, 1)
    mfccs = mfccs[..., tf.newaxis]
    return mfccs 
  

# This gets called every 0.5 seconds
def sd_callback(rec, frames, time, status):

    GPIO.output(led_pin, GPIO.LOW)

    # Start timing for testing
    start = timeit.default_timer()
    
    # Notify if errors
    if status:
        print('Error:', status)
    
    # Remove 2nd dimension from recording sample
    rec = np.squeeze(rec)
    
    # Resample
    rec, new_fs = decimate(rec, sample_rate, resample_rate)
    
    # Save recording onto sliding window
    window[:len(window)//2] = window[len(window)//2:]
    window[len(window)//2:] = rec

    # Compute features
    mfccs = get_mfcc(window)
    

    # Make prediction from model
    in_tensor = np.float32(mfccs.reshape(1, mfccs.shape[0], mfccs.shape[1], 1))
    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    val = output_data[0][0]
    if val > word_threshold:
        print('stop')
        GPIO.output(led_pin, GPIO.HIGH)

    if debug_acc:
        print(val)
    
    if debug_time:
        print(timeit.default_timer() - start)

# Start streaming from microphone
with sd.InputStream(channels=num_channels,
                    samplerate=sample_rate,
                    blocksize=int(sample_rate * rec_duration),
                    callback=sd_callback):
    while True:
        pass
