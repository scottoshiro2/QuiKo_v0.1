# QuiKo_v0.1 (Still Under Construction...)

QuiKo is a quantum computing music generation application that takes in an raw audio file and generates a MIDI file based on the audio file's percussive and harmonic content. It performs feature extraction utilizing Python's Librosa package to perform beat tracking, Harmonic Percussive source separation (HPSS) and onset detection. This information is then mapped to a X and Z rotation angle to prepare allocated qubits into a corresponding state based off these features. 

Once the allocated qubits are prepared into a specific quantum state a 8 qubit system us set up and phase estimation is performed. The estimated phased gives us the 

Output:
The circuit for each corresponding beat within the input audio file
