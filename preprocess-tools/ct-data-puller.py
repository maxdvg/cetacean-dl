# Max Van Gelder
# 6/29/20

# Uses classification trees to pull data which best represents the Humpback
# whale songs which we want to target

# sys.argv[1] is a file which contains .wav files of whale songs

import copy
import math
import numpy as np
from os import listdir
from os.path import dirname, join as pjoin
from scipy import signal
from scipy.io import wavfile
from select_and_search import DenseArchipelago, archipelago_expander, regenerate_from_archipelagos, colormesh_spectrogram
import sys


class SongChunk:
    def __init__(self, samples, sample_rate):
        """
        :param samples: The second parameter returned by scipy.wavfile.read() on a wav file
        :param sample_rate: The first parameter returned by scipy.wavfile.read() on a wav file
        """
        self.frequencies, self.times, self.spectrogram = signal.spectrogram(samples, sample_rate)
        self.archipelagos = []
        self.archipelagos_initialized = False

    def populate_archipelagos(self, min_land_mass=12, max_gap=3):
        """
        Finds all of the archipelagos of size > 'min_land_mass' and with gaps of size at most 'max_gap' contained
        in the song chunk
        :param min_land_mass: The minimum size of an archipelago for it to qualify for addition to self.archipelagos
        :param max_gap: the maximum space between land masses in an archipelago which is permissable
        :side effects self.archipelagos_initialized: After successful execution, sets self.archipelagos_initialized to
        True
        :side effects self.archipelagos: Populates self.archipelagos with all of the archipelagos found which satisfy
        the criteria fo min_land_mass and max_gap
        """
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
        """
        Display the spectrogram which corresponds to the SongChunk, and the archipelagos if they have been populated
        """
        colormesh_spectrogram(self.spectrogram, self.times, self.frequencies, "")
        if self.archipelagos_initialized:
            regenerate_from_archipelagos(self.archipelagos, self.spectrogram, self.times, self.frequencies)

    def num_archipelagos(self):
        if self.archipelagos_initialized:
            return len(self.archipelagos)
        else:
            return None


class RecordingIterator:
    """ Iterator """
    def __init__(self, record):
        self._recording = record
        self._idx = 0

    def __next__(self):
        if self._idx < len(self._recording.chunks):
            result_chunk = self._recording.chunks[self._idx]
            self._idx += 1
            return result_chunk
        raise StopIteration


class Recording:
    def __init__(self, song_wav, chunk_len=20):
        """
        Initializes Recording object by reading in the .wav file 'song_wav' and breaking it into
        several SongChunks with recording length of 'chunk_len'
        :param song_wav: wav file containing the whale song
        :param chunk_len: The length, in seconds, that each 'chunk' of the recording should be
        """
        #TODO: Decimate .wav for faster processing

        # Read in the song_wav
        sample_rate, samples = wavfile.read(song_wav)

        # Populate array of SongChunk objects which as a whole represent the entire song_wav recording
        self.chunks = []
        total_length = float(samples.shape[0] / sample_rate)
        for i in range(math.floor(total_length / chunk_len)):
            # Downsamples to 4096 samples/second (seems to be a good rate)
            self.chunks.append(SongChunk(samples[i * sample_rate * chunk_len:(i+1) * sample_rate * chunk_len
                                                 :int(sample_rate/4096)],
                                         4096))

    def restrict_chunk_frequencies(self, freq_lo, freq_hi):
        """
        Eliminates all data which is not between freq_lo and freq_hi from every chunk's spectrogram
        :param freq_lo: The lowest frequency which will be conserved in the spectrogram
        :param freq_hi: The highest frequency which will be conserved in the spectrogram
        :side effects: Deletes all data < freq_lo and > freq_hi from every chunk in self.chunks
        :return: None
        """
        for chunk in self.chunks:
            # Only keep frequencies between freq_lo and freq_hi Hz
            desired_frequencies = ((chunk.frequencies >= freq_lo) & (chunk.frequencies <= freq_hi))
            chunk.frequencies = chunk.frequencies[desired_frequencies]
            chunk.spectrogram = chunk.spectrogram[desired_frequencies, :]

    def threshold_chunks(self, cutoff_fraction):
        """
        Zeros out every element in all of the chunks spectrogram's which is not in the top (1 - cutoff_fraction) portion
        of the spectrogram
        :param cutoff_fraction: What fraction of every chunk's spectrogram should be zeroed out, eg .8 zeros out the
        lowest 80% of every chunk's spectrogram
        :side effects: Every chunk's spectrogram get's the bottom cutoff_fraction zeroed out
        :return: None
        """
        for chunk in self.chunks:
            flat = chunk.spectrogram.flatten()
            cutoff_pos = int(len(flat) * cutoff_fraction)
            cutoff = np.partition(flat, cutoff_pos)[cutoff_pos]
            chunk.spectrogram = chunk.spectrogram * (chunk.spectrogram >= cutoff)

    def locate_archipelagos(self, min_land_mass=12, max_gap=3):
        """
        Locates all of the archipelagos in each of the chunks in self.chunks
        :param min_land_mass: the minimum land mass for an archipelago
        :param max_gap: the maximum gap allowed for an archipelago
        :side effects: Every chunk gets its archipelagos populated
        :return: None
        """
        for chunk in self.chunks:
            chunk.populate_archipelagos(min_land_mass, max_gap)

    def display_spectrograms(self):
        for chunk in self.chunks:
            chunk.display()

    def __iter__(self):
        return RecordingIterator(self)


if __name__ == "__main__":
    wavs = listdir(sys.argv[1])
    recordings = []
    num_chunks = 0


    # for wav_file in wavs:

    wavs_read = 0
    while wavs and wavs_read < 5:
        wav_file = wavs[wavs_read]
        # Load in all of the song
        recording = Recording(pjoin(sys.argv[1], wav_file))
        # Restrict the frequencies of the song between 200 and 900 Hz
        recording.restrict_chunk_frequencies(200, 900)
        # Threshold at 70%
        recording.threshold_chunks(.9)
        # Pull out the features (# of archipelagos, distance between archipelagos, etc...)
        recording.locate_archipelagos(max_gap=2)

        recordings.append(recording)
        num_chunks += len(recording.chunks)
        wavs_read += 1

    good_data = np.zeros(num_chunks, dtype=bool)
    for recording in recordings:
        for chunk in recording:
            chunk.display()
            ui = input("Is this a good chunk? (y/n) ")