
import numpy as np
from SigPro import *

filename = 'Samples/Anga.wav' ## test file

def Qstate_dict(filename, duration=30): ## duration is set to 30 as default
    audio, fs = load_audiofile(filename)

    Anga = audiofile(audio, fs, duration)
    Beats, sub_div, tempo = Anga.beat_detect()
    Tatum = Anga.tatum_detect()

    tat = Beat(audio, Beats, Tatum, sub_div, fs)
    Rx = tat.Transisent_strength()
    Rz = tat.Harmonic() 

    return Ry_Base(Beats, Rx, Rz), tempo

#print(Qstate_dict(filename))