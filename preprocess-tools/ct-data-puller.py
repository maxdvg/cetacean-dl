# Max Van Gelder
# 6/29/20

# Uses classification trees to pull data which best represents the Humpback
# whale songs which we want to target

# sys.argv[1] is a file which contains .wav files of whale songs

import copy
import math
import numpy as np
from scipy import signal
from scipy.io import wavfile
from select_and_search import DenseArchipelago, archipelago_expander, regenerate_from_archipelagos, colormesh_spectrogram
import sys


class SongChunk:
    def __init__(self, samples, sample_rate):
        self.frequencies, self.times, self.spectrogram = signal.spectrogram(samples, sample_rate)
        self.archipelagos = []
        self.archipelagos_initialized = False

    def populate_archipelagos(self, min_land_mass=12, max_gap=3):
        spec = copy.deepcopy(self.spectrogram)
        for row_idx, row in enumerate(spec):
            for point_idx, point in enumerate(row):
                if point:
                    arch = DenseArchipelago((point_idx, row_idx))
                    spec[row_idx][point_idx] = 0
                    # Allow for a gap between islands in archipelago of size at most 3
                    archipelago_expander(arch, (point_idx, row_idx), spec, max_gap)
                    # Only pick islands with more than 12 pixel
                    if arch.size() > min_land_mass:
                        self.archipelagos.append(arch)
        self.archipelagos_initialized = True

    def display(self):
        colormesh_spectrogram(self.spectrogram, self.times, self.frequencies, "Hi")
        if self.archipelagos_initialized:
            regenerate_from_archipelagos(self.archipelagos, self.spectrogram, self.times, self.frequencies)

class Recording:
    def __init__(self, song_wav, chunk_len=20):
        """
        Initializes Recording object by reading in the .wav file 'song_wav' and breaking it into
        several SongChunks with recording length of 'chunk_len'
        :param song_wav:
        :param chunk_len: The length, in seconds, that each 'chunk' of the recording should be
        """
        #TODO: Decimate .wav for faster processing

        # Read in the song_wav
        sample_rate, samples = wavfile.read(song_wav)

        # Populate array of SongChunk objects which as a whole represent the entire song_wav recording
        self.chunks = []
        total_length = float(samples.shape[0] / sample_rate)
        for i in range(math.floor(total_length / chunk_len)):
            self.chunks.append(SongChunk(samples[i * sample_rate * chunk_len:(i+1) * sample_rate * chunk_len],
                                         sample_rate))

    def restrict_chunk_frequencies(self, freq_lo, freq_hi):
        for chunk in self.chunks:
            # Only keep frequencies between freq_lo and freq_hi Hz
            desired_frequencies = ((chunk.frequencies >= freq_lo) & (chunk.frequencies <= freq_hi))
            chunk.frequencies = chunk.frequencies[desired_frequencies]
            chunk.spectrogram = chunk.spectrogram[desired_frequencies, :]

    def threshold_chunks(self, cutoff_fraction):
        for chunk in self.chunks:
            flat = chunk.spectrogram.flatten()
            cutoff_pos = int(len(flat) * cutoff_fraction)
            cutoff = np.partition(flat, cutoff_pos)[cutoff_pos]
            chunk.spectrogram = chunk.spectrogram * (chunk.spectrogram >= cutoff)

    def locate_archipelagos(self, min_land_mass=12, max_gap=3):
        for chunk in self.chunks:
            chunk.populate_archipelagos(min_land_mass, max_gap)

    def display_spectrograms(self):
        for chunk in self.chunks:
            chunk.display()


if __name__ == "__main__":
    # Load in all of the song
    hi = Recording(sys.argv[1])
    # Restrict the frequencies of the song between 200 and 900 Hz
    hi.restrict_chunk_frequencies(200, 900)
    # Threshold at 70%
    hi.threshold_chunks(.9)
    # Pull out the features (# of archipelagos, distance between archipelagos, etc...)
    hi.locate_archipelagos()

    for chunk in hi.chunks:
        print(len(chunk.archipelagos))

    hi.display_spectrograms()