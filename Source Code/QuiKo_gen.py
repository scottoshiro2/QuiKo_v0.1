
import numpy as np
from mido import Message, MidiFile, MidiTrack
from Qalgorithm import QuiKo_Algorithm, prob_dist

import librosa
from librosa import display

def midi_info(loops, num_tats, prob_dist, states):

	midi_trackinfo = []
	midi_pattern = []; pitch_info   = []

	for num_l in np.arange(loops):		

		for beat, dist in enumerate(zip(prob_dist, states)):

			for tat in np.arange(num_tats[beat]):
				pro = np.random.choice(dist[1], 1, p=dist[0])
				midi_pattern.append(int(pro[0][:4], 2))

			for pitch in np.arange(num_tats[beat]):
				pro = np.random.choice(dist[1], 1, p=dist[0])
				pro = int((int(pro[0][5:], 2) / np.power(2, 4)) * 1000)
				pitch_info.append(pro)

			midi_trackinfo.append([midi_pattern, pitch_info])
			midi_pattern = []; pitch_info   = []

	return midi_trackinfo

def save_to_midi(midi_info, sub_div, filename) -> None:
	mid = MidiFile()
	track = MidiTrack()
	mid.tracks.append(track)

	Rhythm = []
	note_event = []; BEATS = []
	for beat, event in enumerate(midi_info):
		note    = event[1]
		tat_loc = event[0]

		for i in zip(tat_loc, note):
			note_event.append(i)
		BEATS.append(note_event); note_event = []


	## iterate over the arrays
	for event, decode in enumerate(BEATS):
		Beat_frame = np.zeros(16)

		for element in decode:
			Beat_frame[element[0]] = element[1]

		Rhythm.append(Beat_frame)
	
	for Ry in Rhythm:
		for y in Ry:
			y = int((y / 1000) * 127)
			message = 'note_{}'.format('on' if y != 0 else 'off')
			track.append(Message(message, note=y, time=sub_div))

	mid.save(filename)

if __name__ == "__main__":

	audiosample = 'Anga'; outputfile = 'Samples/TESTing!.mid'
	audio, Fs = librosa.load('Samples/'+audiosample+'.wav'); length = len(audio); print(length/Fs)
    
	result_dict, num_tats, tempo = QuiKo_Algorithm(audiosample, simulator=True)

	prob_dist, states = prob_dist(result_dict)

	loops=1 ## have this be an input
	sub_div = int((tempo / (1920)) * 1000)

	midi_info = midi_info(loops, num_tats, prob_dist, states)
	save_to_midi(midi_info, sub_div, outputfile)


	
