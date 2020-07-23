# Max Van Gelder
# 6/29/20

# Loads whale song data into database

# sys.argv[1] is a directory which contains (exclusively) .wav files of whale songs
# sys.argv[2] is a .db SQLite file which contains the database of information
# sys.argv[3] is the path where the spectrograms which are generated should be saved

# TODO: Choose a better way to interact with conn
# WARNING: THE REMAINDER OF THE RECORDING (RECORDING-LENGTH % CHUNK-LENGTH) IS SIMPLY THROWN AWAY.
# AS WRITTEN, THIS PROGRAM IS FOR PULLING GOOD WHALE DATA, NOT AUTOMATICALLY DETECTING WHALES. ANY
# WHALE CALLS WHICH APPEAR IN THE REMAINDER OF THE RECORDING AREN'T DETECTED AS THE PROGRAM STANDS

import argparse
import copy
from collections import namedtuple
import math
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import join as pjoin
from os.path import basename, dirname
from scipy import signal
from scipy.io import wavfile
from select_and_search import DenseArchipelago, archipelago_expander,\
    regenerate_from_archipelagos, colormesh_spectrogram, denoise
import sqlite3
import time

# Default values for processing recordings
Config = namedtuple('Config', ['chunk_len', 'lo_freq', 'hi_freq', 'min_land_mass',
                               'max_gap', 'threshold_cutoff', 'recording_rate',
                               'save_format'])
cfg_default = Config(
    chunk_len=20,
    lo_freq=140,
    hi_freq=950,
    min_land_mass=12,
    max_gap=3,
    threshold_cutoff=.85,
    recording_rate=2048,
    save_format=".png",
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


def bool_to_sqlite(val):
    """
    :param val: A boolean
    :return: 1 if val is True, 0 if val is False
    """
    if val:
        return 1
    return 0


class SongChunk:
    def __init__(self, samples=None, sample_rate=None, chunk_pos=None, parent_path=None):
        """
        :param samples: The second parameter returned by scipy.wavfile.read() on a wav file
        :param sample_rate: The first parameter returned by scipy.wavfile.read() on a wav file
        :param chunk_pos: The position of the chunk within the greater recording
        :param parent_path: The full path of the greater recording which the songchunk is a subsection of
        """
        # If a field is None then the instance is being built from the database,
        # therefore it may not be desirable to reconstruct the spectrogram and other fields
        if samples is not None and sample_rate is not None:
            self.frequencies, self.times, self.spectrogram = signal.spectrogram(samples, sample_rate)
            self.height = self.spectrogram.shape[0]
            self.width = self.spectrogram.shape[1]
        if parent_path is not None and chunk_pos is not None:
            self.specname = "{}-{}{}".format(pjoin(dirname(args.spec_path), basename(parent_path).split('.')[0]),
                                            chunk_pos,
                                            cfg_default.save_format)
        self.archipelagos = []
        self._archipelagos_initialized = False
        self._min_land_mass = None
        self._max_gap = None
        self._threshold_cutoff = None
        self._lo_freq = None
        self._hi_freq = None
        self.spec_in_memory = False

    @classmethod
    def from_database_id(cls, record_id, db_connection):
        """
        Create a SongChunk instance from data already stored in the database. Does NOT load spectrogram into memory,
        only the archipelagos and data about where the spectrogram image is stored are loaded into the instance
        :param record_id: The RecordID of the SongChunk in the database that should be turned into an instance
        :param db_connection: A connection to the SQLite3 database
        :return: A SongChunk instance which has its specname, archipelagos, and spec_in_memory fields populated
        using the data in the database pointed to by db_connection
        """
        chunk = SongChunk()

        # Get the specpath
        db_connection.execute("SELECT SpecPath FROM chunks where RecordID={}".format(record_id))
        path = db_connection.fetchone()
        # Couldn't find chunk with RecordID record_id in the database
        if path is None:
            raise sqlite3.ProgrammingError("The RecordID ({}) of the chunk you tried to load wasn't"
                                           " found in the database".format(record_id))
        chunk.specname = path[0]

        # Fetch if the spectrogram has actually been stored at the Specpath
        db_connection.execute("SELECT SpecWritten FROM chunks where RecordID={}".format(record_id))
        chunk.spec_in_memory = bool(db_connection.fetchone()[0])

        # Check if there are archipelagos for the chunk
        db_connection.execute("SELECT NumACP FROM chunks where RecordID={}".format(record_id))
        num_acp = db_connection.fetchone()[0]
        if num_acp:
            # Fetch the archipelagos if there are supposed to be archipelagos
            db_connection.execute("SELECT ArchID FROM archs where ParentChunk={}".format(record_id))
            archs = db_connection.fetchall()
            if len(archs) != num_acp:
                raise sqlite3.ProgrammingError("Did not find the expected number of archipelagos in the database. "
                                               "Expected: {}, Found: {}".format(num_acp, len(archs)))
            for arch in archs:
                chunk.archipelagos.append(DenseArchipelago.from_database_id(arch[0], db_connection))
        chunk._archipelagos_initialized = True

        db_connection.execute("SELECT Height, Width FROM chunks where RecordID={}".format(record_id))
        chunk.height, chunk.width = db_connection.fetchone()

        return chunk

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

        self._hi_freq = freq_hi
        self._lo_freq = freq_lo

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

        # The process of finding the archipelagos in the spectrogram takes about .25 seconds per chunk on ACS2
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

    def display_archipelagos(self):
        img = np.zeros((self.height, self.width))
        for arch in self.archipelagos:
            for land in arch.land:
                img[self.height - 1 - land[1]][land[0]] = 1
        plt.imshow(img)
        plt.show()

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

    def rowwise_statistical_threshold(self):
        self.spectrogram = denoise(self.spectrogram, display=False)

    def save_chunk(self, c):
        """
        Save the spectrogram which corresponds to the SongChunk, if it isn't already saved in the database
        """
        # First check whether the spectrogram already exists in the location through the database
        c.execute("SELECT SpecWritten FROM chunks where SpecPath='{}'".format(self.specname))
        db_entry = c.fetchone()
        # db_entry == 0 implies chunk is in database but spectrogram not yet written to memory
        # db_entry is None implies the chunk is not yet in the database
        if db_entry is None or db_entry[0] == 0:
            # Save spectrogram image without axes or any surrounding whitespace
            plt.pcolormesh(self.times, self.frequencies, self.spectrogram, shading='gouraud')
            plt.axis("off")
            plt.savefig(self.specname, bbox_inches="tight")
            self.spec_in_memory = True

            # Update the database entry to show that the spectrogram has been written to memory
            # if the chunk is already in the database
            if db_entry is not None:
                c.execute("UPDATE chunks SET SpecWritten=1 where SpecPath='{}'".format(self.specname))

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

    def avg_archipelago_density(self):
        """
        :return: The average density of the archipelagos in the sound chunk, i.e. the ratio
        of the number of pixels in the archipelago to the number of pixels in the bounding rectangle
        of the archipelago
        """
        if self._archipelagos_initialized:
            if self.num_archipelagos() == 0:
                return 0
            else:
                total_density = 0
                for archipelago in self.archipelagos:
                    total_density += archipelago.density()
                return total_density / self.num_archipelagos()
        return None

    def insert_to_database(self, c, fid):
        """
        Insert the SongChunk into the database. This insertion involves a) inserting metadata like num_archipelagos
        and b) inserting all of the archipelagos in the SongChunk into the database as well. See the EXPLANATORY_FILE
        for more details on the database
        :param c: Cursor for SQLite3 database
        :param fid: The FileID field in the database of the Recording which is the parent of this SongChunk
        :return:
        """
        # Insert the SongChunk into the chunks table
        c.execute("INSERT INTO chunks (SpecPath, ParentRecording, NumACP, AvgACPSize, AvgACPDensity, SpecWritten, Width, Height) "
                  "VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}')".format(self.specname,
                                                                             fid,
                                                                             self.num_archipelagos(),
                                                                             self.avg_archipelago_size(),
                                                                             self.avg_archipelago_density(),
                                                                             bool_to_sqlite(self.spec_in_memory),
                                                                             self.spectrogram.shape[1],
                                                                             self.spectrogram.shape[0]))
        # Get the RowID of the SongChunk we just inserted
        c.execute("SELECT last_insert_rowid()")
        chunk_id = c.fetchone()[0]
        # Insert all of the archipelagos in the SongChunk into the Archs table
        for archipelago in self.archipelagos:
            archipelago.insert_to_database(c, chunk_id)


class RecordingIterator:
    """ Iterator """
    def __init__(self, record):
        self._recording = record
        self._idx = 0

    def __next__(self):
        if self._idx < len(self._recording.song_chunks):
            result_chunk = self._recording.song_chunks[self._idx]
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
        # Check if it already exists in the database, and if so abort creation of new object
        c.execute("SELECT * FROM recordings where Path='{}'".format(song_wav))
        if c.fetchone() is not None:
            self.load = False
        else:
            self.song_wav = song_wav
            self.chunk_len = chunk_len
            # Read in the song_wav
            sample_rate, samples = wavfile.read(song_wav)

            # Populate array of SongChunk objects which as a whole represent the entire song_wav recording
            self.song_chunks = []
            total_length = float(samples.shape[0] / sample_rate)
            for i in range(math.floor(total_length / chunk_len)):
                # Downsamples to cfg_default.recording_rate
                self.song_chunks.append(SongChunk(samples[i * sample_rate * chunk_len:(i+1) * sample_rate * chunk_len:
                                                     int(sample_rate/cfg_default.recording_rate)],
                                             cfg_default.recording_rate, i, song_wav))
            self.load = True

    def __bool__(self):
        """
        :return: False if user attempts to initialize a Record which has already been added to the database
        True otherwise
        """
        return self.load

    def restrict_chunk_frequencies(self, freq_lo, freq_hi):
        """
        Eliminates all data which is not between freq_lo and freq_hi from every chunk's spectrogram
        :param freq_lo: The lowest frequency which will be conserved in the spectrogram
        :param freq_hi: The highest frequency which will be conserved in the spectrogram
        :side effects: Deletes all data < freq_lo and > freq_hi from every chunk in self.chunks
        :return: None
        """
        for chunk in self.song_chunks:
            chunk.restrict_frequencies(freq_lo, freq_hi)

    def threshold_chunks(self, cutoff_fraction):
        """
        Zeros out every element in all of the chunks spectrogram's which is not in the top (1 - cutoff_fraction) portion
        of the spectrogram
        :param cutoff_fraction: What fraction of every chunk's spectrogram should be zeroed out, eg .8 zeros out the
        lowest 80% of every chunk's spectrogram
        :side effects: Every chunk's spectrogram get's the bottom cutoff_fraction zeroed out
        """
        for chunk in self.song_chunks:
            chunk.threshold(cutoff_fraction)

    def locate_archipelagos(self, min_land_mass=cfg_default.min_land_mass, max_gap=cfg_default.max_gap):
        """
        Locates all of the archipelagos in each of the chunks in self.chunks
        :param min_land_mass: the minimum land mass for an archipelago
        :param max_gap: the maximum gap allowed for an archipelago
        :side effects: Every chunk gets its archipelagos populated
        :return: None
        """
        for chunk in self.song_chunks:
            chunk.populate_archipelagos(min_land_mass, max_gap)

    def save_spectrograms(self):
        for chunk in self.song_chunks:
            chunk.save_chunk(c)

    def standard_process(self, write_spectrograms=True):
        """
        Performs the standard processing steps on each chunk which is in the recording
        :param write_spectrograms:
        :return:
        """
        # Restrict the frequencies of the song between lo_freq and hi_freq Hz
        # TIME < 1/100th of a second
        self.restrict_chunk_frequencies(cfg_default.lo_freq, cfg_default.hi_freq)
        # Rowwise threshold
        # TIME < 1/100th of a second
        for chunk in self.song_chunks:
            chunk.rowwise_statistical_threshold()
        # Pull out the features (# of archipelagos, distance between archipelagos, etc...)
        # TIME ~5 seconds
        self.locate_archipelagos()
        # Save the spectrogram images
        # TIME 1-2 MINUTES! save_spectrograms is the bottleneck
        if write_spectrograms:
            self.save_spectrograms()

    def insert_to_database(self, c):
        """
        Insert the Recording and all of its SongChunks into the database if there is not already a recording
        in the database which was generated from the same song_wav
        :param c: The database cursor
        :return: False if an Recording which has the same song_wav is already in the database, True if there wasn't
        already a Recording with the song_wav in the database and so this Recording was added to the database
        """
        # Check if a recording with the same file path already exists in the database
        c.execute("SELECT * FROM recordings WHERE Path='{}'".format(self.song_wav))
        if c.fetchone() is not None:
            print("{} already in database".format(self.song_wav))
            return False

        # Insert the overall recordings object into recordings table first
        c.execute("INSERT INTO recordings (Path) VALUES ('{}')".format(self.song_wav))

        # Get the FileID of the recording in the recordings table
        c.execute("SELECT last_insert_rowid()")
        fid = c.fetchone()[0]
        # Insert each of the song chunks into the chunks table
        for chunk in self.song_chunks:
            chunk.insert_to_database(c, fid)

        return True

    def __iter__(self):
        return RecordingIterator(self)


def db_check(c):
    """
    Verifies that all of the expected tables are in the database that cursor 'c' points to
    :param c: A SQLite3 cursor object pointing to a database
    :return: True if all of the tables already existed in the database, false otherwise
    """
    db_exists = True
    # Check that the FileTable exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='recordings'")
    if c.fetchone() is None:
        # Create FileTable table, it didn't exist before
        c.execute("CREATE TABLE recordings (FileID INTEGER PRIMARY KEY, Path TEXT NOT NULL)")
        db_exists = False

    # Check that RecordingTable exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'")
    if c.fetchone() is None:
        # Create RecordingTable
        c.execute("CREATE TABLE chunks (RecordID INTEGER PRIMARY KEY, SpecPath TEXT NOT NULL,"
                  "ParentRecording INTEGER, NumACP INTEGER, AvgACPSize REAL, Label INTEGER,"
                  " AvgACPDensity REAL, SpecWritten INTEGER, Width INTEGER, Height INTEGER, "
                  "FOREIGN KEY(ParentRecording) REFERENCES recordings(FileID))")
        db_exists = False

    # Check that ArchipelagoTable exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='archs'")
    if c.fetchone() is None:
        # Create ArchipelagoTable
        c.execute("CREATE TABLE archs (ArchID INTEGER PRIMARY KEY, "
                  "ParentChunk INTEGER, LeftBd INTEGER, RightBd INTEGER, UpBd INTEGER,"
                  "LowBd INTEGER, FOREIGN KEY(ParentChunk) REFERENCES chunks(RecordID))")
        db_exists = False

    # Check that LandTable exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='land'")
    if c.fetchone() is None:
        # Create ArchipelagoTable
        c.execute("CREATE TABLE land (ParentArchipelago INTEGER, X INTEGER, Y INTEGER,"
                  " FOREIGN KEY(ParentArchipelago) REFERENCES archs(ArchID))")
        db_exists = False

    return db_exists


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_directory", help="The directory containing the .wav files you wish to load")
    parser.add_argument("db", help="The .db for the song database (creates new db here if non-existent)")
    parser.add_argument("spec_path", help="The path to which any spectrograms generated will be written")
    args = parser.parse_args()

    # Get handle for working with database
    conn = sqlite3.connect(args.db)
    c = conn.cursor()
    if not db_check(c):
        print("{} did not contain all of the expected database tables when examined".format(args.db))
    conn.commit()

    wavs = listdir(args.wav_directory)
    recordings = []
    num_chunks = 0

    # for wav_file in wavs:

    wavs_read = 0
    while wavs and wavs_read < 120:
        start = time.time()
        wav_file = wavs[wavs_read]
        # Load in the song
        recording = Recording(pjoin(args.wav_directory, wav_file))
        # If the recording hasn't already been parsed into the database
        if recording:
            # Do all of the preprocessing and feature extraction
            recording.standard_process(write_spectrograms=False)  # NOT WRITING SPECTROGRAMS TO SAVE TIME!
            # Save the information to the database
            # TIME < .5 seconds
            recording.insert_to_database(c)
            conn.commit()
            print("Added new record to database")
            print("It took %.3f seconds to process" % (time.time() - start))

            recordings.append(recording)
            num_chunks += len(recording.song_chunks)
        wavs_read += 1

    # Commit any changes and close connection with the database
    conn.commit()
    conn.close()
