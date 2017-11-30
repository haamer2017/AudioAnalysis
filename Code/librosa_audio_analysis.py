import os
import librosa 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
from librosa import display

def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds

def mono_waveform(f,duration):
	y, sr = librosa.load(f, duration=10) 	
	plt.figure()
	plt.subplot(5,1,1)
	librosa.display.waveplot(y, sr=sr)
	plt.title('Monophonic')
	

def hp_waveform(f):
	y, sr = librosa.load(f, duration=10)
	y_harm, y_perc = librosa.effects.hpss(y)
	plt.subplot(5, 1, 3)
	librosa.display.waveplot(y_harm, sr=sr, alpha=0.25)
	librosa.display.waveplot(y_perc, sr=sr, color='r', alpha=0.5)
	plt.title('Harmonic + Percussive')
	plt.tight_layout()

def stereo_waveform(f):
	y, sr = librosa.load(f,
                     mono=False, duration=10)
	plt.subplot(5, 1, 2)
	librosa.display.waveplot(y, sr=sr)
	plt.title('Stereo')


def minima_energy(f):
	y, sr = librosa.load(f,offset=30, duration=2.0)
	oenv= librosa.onset.onset_strength(y=y, sr=sr)
	# Detect events without backtracking
	onset_raw = librosa.onset.onset_detect(onset_envelope=oenv,backtrack=False)
	# Backtrack the events using the onset envelope
	onset_bt = librosa.onset.onset_backtrack(onset_raw, oenv)
	# Backtrack the events using the RMS energy
	rmse = librosa.feature.rmse(S=np.abs(librosa.stft(y=y)))
	onset_bt_rmse = librosa.onset.onset_backtrack(onset_raw, rmse[0])

	plt.subplot(5, 1, 4)
	plt.plot(oenv, label='Onset strength')
	plt.vlines(onset_raw, 0, oenv.max(), label='Raw onsets')
	plt.vlines(onset_bt, 0, oenv.max(), label='Backtracked', color='r')
	plt.legend(frameon=True, framealpha=0.25)
	plt.subplot(5, 1, 5)
	plt.plot(rmse[0], label='RMSE')
	plt.vlines(onset_bt_rmse, 0, rmse.max(), label='Backtracked (RMSE)', color='r')
	plt.legend(frameon=True, framealpha=0.25)

if __name__ == '__main__':

	file_name = ['C:\\Python\\Python36\\DATA\\sampleData\\Cheapest 100Hz Curved Ultrawide Gaming Monitor - Massdrop Vast.wav', 'C:\\Python\\Python36\\DATA\\sampleData\\Craig_Wireless_Netbook.wav']
	for f in file_name:
		mono_waveform(f,10)
		stereo_waveform(f)
		hp_waveform(f)
		
		minima_energy(f)
		plt.show()