# Max Van Gelder 6/19/2020
# sys.argv[1] is a directory containing any number of .wav files for which
# spectrograms are to be generated

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import copy
import numpy as np
import sys
import math
import time
from os import listdir


def plot_spectrogram(wav_file, spectrogram_title, display=True):
    """
    Plots a of wav_file with title spectrogram_title
    :param wav_file: The .wav file to be plotted
    :param spectrogram_title: The title of the spectrogram to be plotted
    :param display: If True, displays the spectrogram as a matplotlib colormesh. Otherwise, does not display image
    :return: a 3-tuple which consists of:
        - a 2d ndarray of float32s which represents the spectrogram
        - an ndarray of float64s which represents the times of each spectrum
        - an ndarray of float64s which represents the frequencies in the spectrogram
    """
    sample_rate, samples = wavfile.read(wav_file)
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    # Only keep frequencies between 100 and 900 Hz
    desired_frequencies = ((frequencies >= 100) & (frequencies <= 900))
    frequencies = frequencies[desired_frequencies]
    spectrogram = spectrogram[desired_frequencies, :]

    if display:
        colormesh_spectrogram(spectrogram, times, frequencies, spectrogram_title)

    return spectrogram, times, frequencies


def colormesh_spectrogram(spectrogram, times, frequencies, title, save=False):
    plt.pcolormesh(times, frequencies, spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(title)

    if save:
        plt.savefig("{}.png".format(title.split('.')[0]))

    plt.show()


def denoise(image, times, frequencies, display=True):
    """
    (Intended for use on noisy spectrograms)
    Denoise an image by removing all pixels from each row which are less than the (mean + std deviation) of that
    row.
    :param image: A 2d numpy array representing an image
    :param display: If True, then displays the image as a matplotlib plot. Otherwise, does not display image
    :return: The denoised image.
    """
    denoised_img = copy.deepcopy(image)

    for row in denoised_img:
        mean = row.mean()
        std_dev = row.std()

        # Select the top 10 percent
        dropped_indices = row < mean + std_dev
        row[dropped_indices] = 0
        row[np.invert(dropped_indices)] = 1

    if display:
        colormesh_spectrogram(denoised_img, times, frequencies, "Denoised")

    return denoised_img


class DenseArchipelago:
    def __init__(self, *seed):
        if seed:
            # Bounding box for the dense archipelago is initialized to only
            # include the archipelago's seed if seed was specified on init
            self.lower_bd = self.upper_bd = seed[0][1]
            self.left_bd = self.right_bd = seed[0][0]
        else:
            # If seed wasn't specified on init, initialize bounding box to none
            self.lower_bd = self.upper_bd = None
            self.left_bd = self.right_bd = None
        # List containing all of the points within the archipelago
        self.land = []

    def add_point(self, location):
        """
        Add a point to the archipelago, updating the bounding box of the
        archipelago in the process
        :param location: A tuple containing the x and y coordinates of the location
        which is being added
        :return: None
        """
        if self.lower_bd is None:
            self.left_bd = self.right_bd = location[0]
            self.lower_bd = self.upper_bd = location[1]
        else:
            self.left_bd = min(self.left_bd, location[0])
            self.lower_bd = min(self.lower_bd, location[1])
            self.right_bd = max(self.right_bd, location[0])
            self.upper_bd = max(self.upper_bd, location[1])
        self.land.append(location)

    def size(self):
        """
        :return: Number of points contained in the archipelago
        """
        return len(self.land)

    def density(self):
        """
        Calculates the proportion of the bounding box which is occupied by
        points in the archipelago
        :requires: Archipelago must have at least one element in it already
        :return: a float representing the proportion of the bounding box occupied
        by points in the archipelago
        """
        volume = (self.right_bd - self.left_bd) * (self.upper_bd - self.lower_bd)
        return self.size() / float(volume)


def archipelago_expander(archipelago, focus, cp_spectrogram, cur_gap):
    """
    :side_effects cp_spectrogram: DESTROYS cp_spectrogram during the search process
    :side_effects archipelago: Adds any newly discovered landmasses to the archipelago
    :param archipelago: A DenseArchipelago object to be populated with any landmasses found
    :param focus: The 'point of expansion', from which nearby land masses will be explored
    :param cp_spectrogram: The spectrogram to search through for finding new land masses
    :param cur_gap: The acceptable distance of water to traverse before giving up looking for land
    :return: None
    """
    potential_neighbors = []
    spec_height = cp_spectrogram.shape[0]
    spec_width = cp_spectrogram.shape[1]

    # Left neighbor
    if focus[0] > 0:
        potential_neighbors.append((focus[0] - 1, focus[1]))
    # Upper neighbor
    if focus[1] > 0:
        potential_neighbors.append((focus[0], focus[1] - 1))
    # Right neighbor
    if focus[0] < spec_width - 1:
        potential_neighbors.append((focus[0] + 1, focus[1]))
    # Lower neighbor
    if focus[1] < spec_height - 1:
        potential_neighbors.append((focus[0], focus[1] + 1))

    while potential_neighbors:
        potential_neighbor = potential_neighbors.pop()
        if cp_spectrogram[potential_neighbor[1]][potential_neighbor[0]]:
            # Add the land mass to the archipelago if it exists
            archipelago.add_point(potential_neighbor)
            # Remove the land mass from the spectrogram so it can't be "found"
            # again by this algorithm
            cp_spectrogram[potential_neighbor[1]][potential_neighbor[0]] = 0
            archipelago_expander(archipelago, potential_neighbor, cp_spectrogram, cur_gap)
        elif cur_gap > 0:
            archipelago_expander(archipelago, potential_neighbor, cp_spectrogram, cur_gap - 1)


def regenerate_from_archipelagos(archipelago_list, original_spectrogram, times, frequencies, display=True):
    """
    Converts a list of archipelagos back into an image of said archipelagos
    :param archipelago_list: A list of DenseArchipelagos which were pulled out of 'original_spectrogram'
    :param original_spectrogram: The spectrogram which was originally used to generate the archipelagos in
    'archipelago_list'
    :param display: If True, then displays a matplotlib colormesh of the regenerated archipelago spectrogram,
    otherwise does not display anything
    :return: A 2d numpy array of float32s which represents the spectrogram containing only the archipelagos from
    'archipelago_list'
    """
    regenerated = np.zeros(original_spectrogram.shape, dtype=np.float32)
    for archipelago in archipelago_list:
        for land in archipelago.land:
            regenerated[land[1]][land[0]] = 1

    if display:
        colormesh_spectrogram(regenerated, times, frequencies, "Archipelago Regeneration")

    return regenerated


if __name__ == "__main__":
    wavs = listdir(sys.argv[1])
    for wav in wavs:
        # Generate plain spectrogram
        audio, times, frequencies = plot_spectrogram(sys.argv[1] + wav, wav)
        # Generate denoised spectrogram
        test_spectrogram = denoise(audio, times, frequencies)
        # Find archipelagos in denoised spectrogram
        archipelagos = []
        for row_idx, row in enumerate(test_spectrogram):
            for point_idx, point in enumerate(row):
                if point:
                    arch = DenseArchipelago((point_idx, row_idx))
                    test_spectrogram[row_idx][point_idx] = 0
                    # Allow for a gap between islands in archipelago of size at most 3
                    archipelago_expander(arch, (point_idx, row_idx), test_spectrogram, 3)
                    # Only pick islands with more than 12 pixel
                    if arch.size() > 12:
                        archipelagos.append(arch)

        # Regenerate and display cleaned up spectrogram from denoised spectrogram
        audio_regen = regenerate_from_archipelagos(archipelagos, audio, times, frequencies)

        # Draw in bounding boxes on cleaned up and denoised spectrogram
        for archipelago in archipelagos:
            # Print the times that the bounding box is bounding
            print("Sample starting at {} ending at {}".format(
                time.strftime('%H:%M:%S', time.gmtime(math.floor(times[archipelago.left_bd]))),
                time.strftime('%H:%M:%S', time.gmtime(math.floor(times[archipelago.right_bd])))))

            # draw left and right lines
            for i in range(archipelago.lower_bd, archipelago.upper_bd):
                audio[i][archipelago.left_bd] = audio[i][archipelago.right_bd] = audio.max()

            # draw upper and lower lines
            for i in range(archipelago.left_bd, archipelago.right_bd):
                audio[archipelago.lower_bd][i] = audio[archipelago.upper_bd][i] = audio.max()

        colormesh_spectrogram(audio, times, frequencies, "Bounding Boxes {}".format(wav))

        input("Ready for the next spectrograms?")
