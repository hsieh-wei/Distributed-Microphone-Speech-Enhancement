# Overview

This folder is an example of the LA-DDAE model. The main function is to train the input noisy speech and clean speech with location information, and output the enhancement speech. For more details, please refer to 
<span style="color:red">
Chapter 2 of DOC/Distributed Microphone Speech Enhancement.pdf
</span>

# Prerequisites

- Keras 1.2
- Tensorflow 1.x as backend
- h5py
- librosa
- scipy

# Main Parameters and Function

- <code>wav2spec</code> : Through the Hanmin window (512 is selected here) and STFT short-time Fourier transform, the data in the .wav file is converted into spectral data.

- <code>spec2wav</code> : Through the Hanmin window (512 is selected here), and ISTFT inverse short-time Fourier transform, the spectral data is converted into a .wav file.

- <code>ddae_generator_noVAD_1</code>：Onydata_1 will access the eigenvalues of the training audio files recorded by the microphone number 1 after converting them into spectrums. The following is to convert the location information into features and add them to Onydata_1. Onydata_2 is recorded by microphone number 2 and so on... The experiment uses three microphones, so there will be Onydata_1, Onydata_2, Onydata_3

- <code>test</code>：Onydata_1 will access the eigenvalues of the training audio files recorded by the microphone number 1 after converting them into spectrums. The following is to convert the location information into features and add them to Onydata_1. Onydata_2 is recorded by microphone number 2 and so on... The experiment uses three microphones, so there will be Onydata_1, Onydata_2, Onydata_3

- <code>Trpath_x</code>：Loading train audio files

- <code>Vapath_x</code>：Loading test audio files 

- <code>Neural Network</code> : There will be 257 eigenvalues converted from STFT to spectrum. In order to make the sentence data complete, the two sound frames before and after will be strung together and sent to the neural layer for training. In this demonstration, the eigenvalues of 3 positions are added, so the total input will be: 
<span style="color:yellow">
(257+3) (spectral feature + position feature) x5 (sound frame series) x3 (number of microphones)
</span>