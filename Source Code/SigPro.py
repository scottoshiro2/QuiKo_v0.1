
import numpy as np
from numpy import fft
import matplotlib.pyplot as plt

from scipy.signal import argrelextrema

import librosa
from librosa import display

filename = 'Samples/Anga.wav' ## Test Signal

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_audiofile(filename):
    audio, Fs = librosa.load(filename)
    return audio, Fs

def Ry_Base(Beats, Rx, Rz): ## Ry: onset strengths --> Theta, Rz: the frequency components --> Psi
    Ry_Base = {}
    Ry_subase = {}
    
    for index, beat in enumerate(Beats):
        if index < len(Beats)-1:
            Ry_subase['Rx ' + str(index+1)] = Rx[index]
            Ry_subase['Rz ' + str(index+1)] = Rz[index]

            Ry_Base['Beat' + str(index+1)] = Ry_subase
            Ry_subase = {}
    
    return Ry_Base

class audiofile:

    def __init__(self, audio, samplerate, duration):

        self.fs = samplerate
        self.dur = duration
        self.audio = audio
        
    def beat_detect(self):
        tempo, beats = librosa.beat.beat_track(y=self.audio, sr=self.fs)
        beat_times = librosa.frames_to_time(beats, sr=self.fs)

        B_Intv = np.linspace(0, len(beat_times)-2, num=len(beat_times)-1)

        for index, beat in enumerate(beat_times):
            if index != 0:
                bintv = beat_times[index] - beat_times[index-1]
                B_Intv[index-1] = round(bintv, 2)

        avg_beat = round(np.sum(B_Intv) / len(B_Intv), 2)
        B_time = np.linspace(0, len(beat_times)-2, num=len(beat_times)-1)

        for i, val in enumerate(B_time):
            B_time[i] = round(i * avg_beat, 2)

        sub_div = np.linspace(0, 5*(len(beat_times)-2), num=5*(len(beat_times)-2))

        for div, sub in enumerate(sub_div):
            sub_div[div] = round(div * (avg_beat/4), 2)

        return B_time, sub_div, tempo

    def tatum_detect(self):
        onset = librosa.onset.onset_strength(y=self.audio, sr=self.fs)
        times = librosa.times_like(onset, sr=self.fs)
        onset_detect = librosa.onset.onset_detect(onset_envelope=onset, sr=self.fs)

        tatum_times = times[onset_detect]

        return tatum_times, onset

class Beat:

    def __init__(self, audio_blk, beat_time, tatum_times, sub_div, sample_rate): ## Tatum is type np.array

        self.audio_blk = audio_blk
        self.fs = sample_rate

        self.beat_time = beat_time
        self.sub_div = sub_div
        self.tatum_times, self.tatum_strength = tatum_times

        self.beat_samp = np.linspace(0, len(self.beat_time)-1, num=len(self.beat_time)-1)      ## Beats in samples
        for element, time in enumerate(self.beat_samp):
            self.beat_samp[element] = int(self.beat_time[element] * self.fs)
            
        self.tatum_samp = np.linspace(0, len(self.tatum_times)-1, num=len(self.tatum_times)-1) ## Tatum in samples
        for element, time in enumerate(self.tatum_samp):
            self.tatum_samp[element] = int(self.tatum_times[element] * self.fs)

    def Transisent_strength(self):
        
        tat_S = []
        offset = 0
        for index, beat in enumerate(self.beat_time):
            tat_subT = self.tatum_times[(self.tatum_times < beat)&
                                        (self.tatum_times >= self.beat_time[index-1])]

            tat_sub2 = np.zeros(len(tat_subT))

            if (index != 0) & (tat_subT != []):
                for i, j in enumerate(tat_subT):
                    #tat_samp = int(j * self.fs)
                    tat_sub2[i] = self.tatum_strength[i + offset]#; print('check: ', self.tatum_strength[i + offset])
                    offset += len(tat_subT)
                tat_element = tat_sub2 / np.max(tat_sub2)#; print('check: ', tat_element)
                tat_element2 = np.linspace(0, len(tat_element), num=len(tat_element))
                
                for x, y in enumerate(tat_element):
                    y = round(y, 2)
                    tat_element2[x] = y

                tat_S.append(tat_element2)
            else:
                tat_S.append([])
        return tat_S
        
    def Harmonic(self): ## Q: How does the perceived perception range affect the result of the qunatum circuit? (reformulate and clean up question)
        audio = self.audio_blk
        stft = librosa.stft(np.array(audio))
        harm, perc = librosa.decompose.hpss(stft)
        h_audio = librosa.core.istft(harm)

        freq_array = []; beat_fa = []
        prev_beat = 0
        for element, beat in enumerate(self.beat_samp):

            tat_subT = self.tatum_samp[(self.tatum_samp < self.beat_samp[element])&
                                        (self.tatum_samp >= self.beat_samp[element-1])]
            
            for index, val in enumerate(self.tatum_samp):
                if (val >= prev_beat) & (val < beat):
            
                    if (index != 0) & (index < len(self.tatum_samp)):

                        sub_blk = h_audio[int(self.tatum_samp[index-1]):int(self.tatum_samp[index])]
                        fft_subblk = np.real(np.fft.fft(sub_blk))

                        fft_subblk = self.fs / (2 * np.array(argrelextrema(fft_subblk, np.greater)))
                        pos_fft = fft_subblk[fft_subblk >= 0]
                        pos_fft = pos_fft[(pos_fft < 1000) & (pos_fft > 100)]
                        freq_pitch = pos_fft[:len(tat_subT)] ##give me the first number of qubits out, so for testing 3, experiment with the perceptual rang of human hearing and where music lies!!!
                        freq_array.append(freq_pitch) ## This is going to be deafult rang for now....
                        #print(freq_array, np.max(freq_array),np.min(freq_array), len(freq_array))
            
            beatfa = freq_array
            prev_beat = beat
        
        return beatfa

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
