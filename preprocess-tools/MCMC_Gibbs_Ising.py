# Max Van Gelder
# 6/25/20
# Denoising with Gibbs sampling on a Ising model which is really just a thresholded spectrogram

# Inspired by Pranav Prathvikumar's post on Medium:
# https://towardsdatascience.com/image-denoising-by-mcmc-fc97adeaba9b

# sys.argv[1] is a directory containing .wav files of whale songs

from select_and_search import plot_spectrogram, DenseArchipelago
from select_and_search import colormesh_spectrogram, denoise, archipelago_expander, regenerate_from_archipelagos
import copy
import numpy as np
from scipy.stats import multivariate_normal
import sys
from os import listdir
import math
import time


def threshold(image, min_val, title, display=True):
    """
    Converts a continuous-valued image to a binary image by
    setting all pixels below min_val to 0 and all pixels
    above min_val to the maximum value in the image
    :param display: If true, then displays the image before and after
    thresholding. Otherwise, does not display image at all
    :param image: a numpy array of floats representing a grayscale image
    :param min_val: the minimum value desired for cutoff
    :param title: the title of the images which get displayed
    :return: The thresholded numpy array
    """

    # Set all values in image less than min_val to zero
    thresholded = copy.deepcopy(image)
    dropped_indices = thresholded < min_val
    thresholded[dropped_indices] = 0
    # set all values in image less than min_val to max_val
    kept_indices = thresholded >= min_val
    thresholded[kept_indices] = 1

    if display:
        colormesh_spectrogram(thresholded, times, freqs, title)

    return thresholded


if __name__ == "__main__":
    wavs = listdir(sys.argv[1])
    for wav in wavs:
        # Generate spectrogram and get times and frequencies as well
        spectrogram, times, freqs = plot_spectrogram(sys.argv[1] + wav, "Unedited {}".format(wav))

        # Threshold the spectrogram at the mean
        thresholded_spectrogram = threshold(spectrogram, spectrogram.mean(), "Thresholded")

        # Create an array with a boundary of zeros for the Gibbs sampling
        ising = np.zeros((thresholded_spectrogram.shape[0] + 2, thresholded_spectrogram.shape[1] + 2), dtype=np.int8)

        # Populate Ising with values from thresholded spectrogram
        for row_idx, row in enumerate(thresholded_spectrogram):
            for pixel_idx, pixel in enumerate(row):
                if pixel == 0:
                    ising[row_idx + 1][pixel_idx + 1] = -1
                else:
                    ising[row_idx + 1][pixel_idx + 1] = 1

        # higher means that adjacent pixels will mores strongly want to be the same
        couple_strength = 4

        # Gibbs sampling (3 iterations)
        for gibbs_sample_pass in range(3):
            for ising_col in range(1, len(ising[0]) - 1):
                for ising_row in range(1, len(ising) - 1):
                    potentials = []
                    for alignment in [-1, 1]:
                        edge_pot = np.exp(couple_strength * ising[ising_row - 1, ising_col] * alignment)\
                                   * np.exp(couple_strength * ising[ising_row, ising_col - 1] * alignment)\
                                   * np.exp(couple_strength * ising[ising_row + 1, ising_col] * alignment)\
                                   * np.exp(couple_strength * ising[ising_row, ising_col + 1] * alignment)
                        potentials.append(edge_pot)
                    prob1 = multivariate_normal.pdf(thresholded_spectrogram[ising_row - 1, ising_col - 1], mean=1, cov=1) \
                            * potentials[1] / \
                            (multivariate_normal.pdf(thresholded_spectrogram[ising_row - 1, ising_col - 1], mean=1, cov=1)
                             * potentials[1]
                             + multivariate_normal.pdf(thresholded_spectrogram[ising_row - 1, ising_col - 1], mean=-1, cov=1)
                             * potentials[0])
                    if np.random.uniform() <= prob1:
                        ising[ising_row, ising_col] = 1
                    else:
                        ising[ising_row, ising_col] = -1

        # Convert back to binary image (0 and 2, not -1 and 1 like Ising)
        mcmc_denoised = ising[1:len(ising) - 1, 1:len(ising[0]) - 1] + 1

        colormesh_spectrogram(mcmc_denoised, times, freqs, "MCMC {}".format(wav), save=True)

        # Find archipelagos in denoised spectrogram
        archipelagos = []
        for row_idx, row in enumerate(mcmc_denoised):
            for point_idx, point in enumerate(row):
                if point:
                    arch = DenseArchipelago((point_idx, row_idx))
                    mcmc_denoised[row_idx][point_idx] = 0
                    # Allow for a gap between islands in archipelago of size at most 3
                    archipelago_expander(arch, (point_idx, row_idx), mcmc_denoised, 3)
                    # Only pick islands with more than 12 pixel
                    if arch.size() > 20:
                        archipelagos.append(arch)

        # Regenerate and display cleaned up spectrogram from denoised spectrogram
        audio_regen = regenerate_from_archipelagos(archipelagos, spectrogram, times, freqs)

        # Draw in bounding boxes on cleaned up and denoised spectrogram
        for archipelago in archipelagos:
            # Print the times that the bounding box is bounding
            print("Sample starting at {} ending at {}".format(
                time.strftime('%H:%M:%S', time.gmtime(math.floor(times[archipelago.left_bd]))),
                time.strftime('%H:%M:%S', time.gmtime(math.floor(times[archipelago.right_bd])))))

            # draw left and right lines
            for ising_col in range(archipelago.lower_bd, archipelago.upper_bd):
                spectrogram[ising_col][archipelago.left_bd] = spectrogram[ising_col][archipelago.right_bd] = spectrogram.max()

            # draw upper and lower lines
            for ising_col in range(archipelago.left_bd, archipelago.right_bd):
                spectrogram[archipelago.lower_bd][ising_col] = spectrogram[archipelago.upper_bd][ising_col] = spectrogram.max()

        colormesh_spectrogram(spectrogram, times, freqs, "Bounding Boxes {}".format(wav), save=True)

        input("Ready for the next spectrograms?")