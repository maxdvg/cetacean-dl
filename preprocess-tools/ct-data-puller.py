# Max Van Gelder
# 6/29/20

# Uses classification trees to pull data which best represents the Humpback
# whale songs which we want to target

# sys.argv[1] is a directory which contains (exclusively) .wav files of whale songs
# sys.argv[2] is a .json file for storing SongRecords

import copy
from collections import namedtuple
import json
import math
import numpy as np
from os import listdir
from os.path import join as pjoin
from scipy import signal
from scipy.io import wavfile
from select_and_search import DenseArchipelago, archipelago_expander, regenerate_from_archipelagos, colormesh_spectrogram
from sklearn.cluster import KMeans
import sys

# Default values for processing recordings
Config = namedtuple('Config', ['chunk_len', 'lo_freq', 'hi_freq', 'min_land_mass',
                               'max_gap', 'threshold_cutoff', 'recording_rate'])
cfg_default = Config(
    chunk_len=5,
    lo_freq=200,
    hi_freq=950,
    min_land_mass=12,
    max_gap=3,
    threshold_cutoff=.85,
    recording_rate=2048
)


class AlreadyInitializedError(Exception):
    """
    Raised when user tries to change a variable which is already set
    """
    def __init__(self, value, attempted_value, message):
        self.value = value
        self.attempted_value = attempted_value
        self.message = message

    def __str__(self):
        return "{}: Attempted to reset value {} to {}".format(self.message, self.value, self.attempted_value)


class SongChunk:
    def __init__(self, samples, sample_rate):
        """
        :param samples: The second parameter returned by scipy.wavfile.read() on a wav file
        :param sample_rate: The first parameter returned by scipy.wavfile.read() on a wav file
        """
        self.frequencies, self.times, self.spectrogram = signal.spectrogram(samples, sample_rate)
        self.archipelagos = []
        self._archipelagos_initialized = False
        self._min_land_mass = None
        self._max_gap = None
        self._threshold_cutoff = None
        self._lo_freq = None
        self._hi_freq = None

    def restrict_frequencies(self, freq_lo, freq_hi):
        """
        Restrict the frequencies of the spectrogram to be between 'freq_lo' and 'freq_hi'
        :param freq_lo: The lowest frequency we want maintained in the spectrogram
        :param freq_hi: The highest frequency we want maintained in the spectrogram
        :raises AlreadyInitializedError: if user attempts to restrict frequencies to a superset of the current
        current frequencies
        """
        if self._lo_freq is not None:
            if self._lo_freq > freq_lo:
                raise AlreadyInitializedError(self._lo_freq, freq_lo, "Invalid attempt to restrict low frequencies")
            elif self._hi_freq < freq_hi:
                raise AlreadyInitializedError(self._hi_freq, freq_hi, "Invalid attempt to restrict high frequencies")

        # Only keep frequencies between freq_lo and freq_hi Hz
        desired_frequencies = ((self.frequencies >= freq_lo) & (self.frequencies <= freq_hi))
        self.frequencies = self.frequencies[desired_frequencies]
        self.spectrogram = self.spectrogram[desired_frequencies, :]

    def populate_archipelagos(self, min_land_mass=cfg_default.min_land_mass, max_gap=cfg_default.max_gap):
        """
        Finds all of the archipelagos of size > 'min_land_mass' and with gaps of size at most 'max_gap' contained
        in the song chunk
        :param min_land_mass: The minimum size of an archipelago for it to qualify for addition to self.archipelagos
        :param max_gap: the maximum space between land masses in an archipelago which is permissable
        :side effects self.archipelagos_initialized: After successful execution, sets self.archipelagos_initialized to
        True
        :side effects self.archipelagos: Populates self.archipelagos with all of the archipelagos found which satisfy
        the criteria fo min_land_mass and max_gap
        :raises AlreadyInitializedError: if archipelagos have already been populated
        """
        if self._archipelagos_initialized:
            raise AlreadyInitializedError(self._archipelagos_initialized, True, "Archipelagos already initialized")

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

        self._archipelagos_initialized = True
        self._min_land_mass = min_land_mass
        self._max_gap = max_gap

    def threshold(self, cutoff_fraction=cfg_default.threshold_cutoff):
        """
        Zeros out every element in the chunks spectrogram which is not in the top (1 - cutoff_fraction) fraction
        of the spectrogram's values
        :requires: self.threshold_cutoff = None, i.e. this function hasn't been called on this instance before
        :param cutoff_fraction: What fraction of every chunk's spectrogram should be zeroed out, eg .8 zeros out the
        lowest 80% of every chunk's spectrogram
        :side effects: The chunk's spectrogram gets the bottom cutoff_fraction zeroed out
        :side effects: sets self.threshold_cutoff to the cutoff_fraction
        :raises AlreadyInitializedError: if the chunk has already been thresholded
        """
        if self._threshold_cutoff is not None:
            raise AlreadyInitializedError(self._threshold_cutoff, cutoff_fraction, "You can't threshold again")

        flat = self.spectrogram.flatten()
        cutoff_pos = int(len(flat) * cutoff_fraction)
        cutoff = np.partition(flat, cutoff_pos)[cutoff_pos]
        self.spectrogram = self.spectrogram * (self.spectrogram >= cutoff)
        self._threshold_cutoff = cutoff_fraction

    def display(self):
        """
        Display the spectrogram which corresponds to the SongChunk, and the archipelagos if they have been populated
        """
        colormesh_spectrogram(self.spectrogram, self.times, self.frequencies, "")
        if self._archipelagos_initialized:
            regenerate_from_archipelagos(self.archipelagos, self.spectrogram, self.times, self.frequencies)

    def num_archipelagos(self):
        """
        :return: The number of archipelagos detected in the SongChunk from when populate_archipelagos was called
        None if populate_archipelagos hasn't yet been called
        """
        if self._archipelagos_initialized:
            return len(self.archipelagos)
        return None

    def avg_archipelago_size(self):
        """
        :return: The average size of an archipelago in the SongChunk, if there are archipelagos
         from when populate_archipelagos was called. 0 if there are no archipelagos but populate_archipelagos was
         called. None if populate_archipelagos hasn't yet been called.

        """
        if self._archipelagos_initialized:
            if self.num_archipelagos() == 0:
                return 0
            else:
                tot_land_mass = 0
                for archipelago in self.archipelagos:
                    tot_land_mass += archipelago.size()
                return tot_land_mass / self.num_archipelagos()
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
    def __init__(self, song_wav, chunk_len=cfg_default.chunk_len):
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
            # Downsamples to cfg_default.recording_rate
            self.chunks.append(SongChunk(samples[i * sample_rate * chunk_len:(i+1) * sample_rate * chunk_len
                                                 :int(sample_rate/cfg_default.recording_rate)],
                                         cfg_default.recording_rate))

    def restrict_chunk_frequencies(self, freq_lo, freq_hi):
        """
        Eliminates all data which is not between freq_lo and freq_hi from every chunk's spectrogram
        :param freq_lo: The lowest frequency which will be conserved in the spectrogram
        :param freq_hi: The highest frequency which will be conserved in the spectrogram
        :side effects: Deletes all data < freq_lo and > freq_hi from every chunk in self.chunks
        :return: None
        """
        for chunk in self.chunks:
            chunk.restrict_frequencies(freq_lo, freq_hi)

    def threshold_chunks(self, cutoff_fraction):
        """
        Zeros out every element in all of the chunks spectrogram's which is not in the top (1 - cutoff_fraction) portion
        of the spectrogram
        :param cutoff_fraction: What fraction of every chunk's spectrogram should be zeroed out, eg .8 zeros out the
        lowest 80% of every chunk's spectrogram
        :side effects: Every chunk's spectrogram get's the bottom cutoff_fraction zeroed out
        """
        for chunk in self.chunks:
            chunk.threshold(cutoff_fraction)

    def locate_archipelagos(self, min_land_mass=cfg_default.min_land_mass, max_gap=cfg_default.max_gap):
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
        recording.restrict_chunk_frequencies(cfg_default.lo_freq, cfg_default.hi_freq)
        # Threshold at 70%
        recording.threshold_chunks(cfg_default.threshold_cutoff)
        # Pull out the features (# of archipelagos, distance between archipelagos, etc...)
        recording.locate_archipelagos()

        recordings.append(recording)
        num_chunks += len(recording.chunks)
        wavs_read += 1

    # good_data = np.zeros(num_chunks, dtype=bool)

    chunk_data = []
    for recording in recordings:
        for chunk in recording:
            chunk_data.append([chunk.num_archipelagos(), chunk.avg_archipelago_size()])

    # K-Means
    kmeans = KMeans(n_clusters=2, random_state=0).fit(chunk_data)

    i = 0
    for recording in recordings:
        for chunk in recording:
            chunk.display()
            print(kmeans.labels_[i])
            i += 1
            input("Next?")
