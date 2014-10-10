import cv2
import logging
import glob
import numpy as np
import os
import re
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

import coil100vars


def get_objid_and_imgid(file_path):
    """input: ./full/path/to/coil-100/obj99__90.png
    output: (obj99, 90) tuple
    """
    file_name = os.path.basename(file_path)
    match = re.match('(obj\d+)__(\d+).png', file_name)
    if not match:
        logging.error('Unexpected file name: %s; ending execution')
        sys.exit(1)
    return (match.group(1), match.group(2))


def load_imgs(files_pattern):
    """Load ALL images into memory. Each image is saved as a dictionary:
    {
        'objid': 'obj1',
        'imgid': '0',
        'rgb': <3-dimension numpy array (128, 128, 3) representing obj1__0.png>,
    }
    """
    logging.info('Reading images to memory')
    coil100vars.imgs = []
    for file_path in glob.glob(files_pattern):
        objid, imgid = get_objid_and_imgid(file_path)
        coil100vars.imgs.append({
            'objid': objid,
            'imgid': imgid,
            'rgb': cv2.imread(file_path),
        })
    return


def calc_histograms(bins):
    """Given the original images and the number of bins to split the vector-space,
    compute the histograms in the following format:

    histogram = <1-dimension numpy array (3*COIL100_BINS)>
    E.g (assuming bins=4):
    histogram = [
        0.20, 0.40, 0.30, 0.10, # channel 1 frequencies
        0.10, 0.10, 0.75, 0.05, # channel 2 frequencies
        0.25, 0.25, 0.30, 0.20, # channel 3 frequencies
    ]
    """
    if not isinstance(coil100vars.imgs, list):
        raise ValueError('Called calc_histograms() before load_coil100_imgs()')

    logging.info('Calculating histograms')
    coil100vars.histograms_matrix = []
    for img in coil100vars.imgs:
        img['rgb_hist'] = load_color_histogram(img['rgb'], bins)
        coil100vars.histograms_matrix.append(img['rgb_hist'])
    coil100vars.histograms_matrix = np.asarray(coil100vars.histograms_matrix)
    return


def load_color_histogram(img, bins):
    """For a given image, calculates its normalized frequencies histograms.
    `img` is supposed to contain 3 channels
    Output is a 1-dimensional numpy array with shape=(3*bins)
    """
    total_pixels = img.shape[0] * img.shape[1]
    return np.concatenate((
        cv2.calcHist([img], [0], None, [bins], [0,256])[:,0] / total_pixels, # channel 1
        cv2.calcHist([img], [1], None, [bins], [0,256])[:,0] / total_pixels, # channel 2
        cv2.calcHist([img], [2], None, [bins], [0,256])[:,0] / total_pixels, # channel 3
    ))


def build_codebooks(coordinates_num=50, strategy='random'):
    """Build codebook from given images and corresponding histograms.
    `coordinates_num` represents the number of coordinates to return
    `strategy` is one of: "random" or "kmeans"
    Output: 2-dimensional numpy array with shape=(coordinates_num, 3*bins)
    """
    if not isinstance(coil100vars.histograms_matrix, np.ndarray):
        raise ValueError('Called build_codebook() before loading `coil100vars.imgs`')

    logging.info('Building codebook; strategy="%s"' % (strategy))

    if coordinates_num > coil100vars.histograms_matrix.shape[0]:
        raise ValueError('Codebook: unable to retrieve %s coordinates from %s images' % (coordinates_num, len(coil100vars.imgs)))

    if strategy == 'random':
        random_rows_indexes = np.random.choice(len(coil100vars.imgs), size=coordinates_num, replace=False)
        random_histograms = coil100vars.histograms_matrix[random_rows_indexes,:]
        coil100vars.codebook_histograms = random_histograms
    elif strategy == 'kmeans':
        estimator = KMeans(init='k-means++', n_clusters=coordinates_num)
        estimator = estimator.fit(coil100vars.histograms_matrix)
        coil100vars.codebook_histograms = estimator.cluster_centers_
    else:
        raise ValueError('Invalid codebook strategy: %s' % (strategy))

    # build codebook of colors based on codebook of histograms
    bins = coil100vars.histograms_matrix.shape[1] / 3
    # e.g (assuming bins=8): weights=[ 15  46  77 108 139 170 201 232 ]
    weights = np.arange(start=256/(2*bins), stop=255, step=256/bins)
    weights = np.concatenate((weights, weights, weights)) # 3 channels concatenated

    coil100vars.codebook_colors = []
    for hist in coil100vars.codebook_histograms:
       coil100vars.codebook_colors.append(np.sum((hist * weights).reshape((3, bins)), axis=1, dtype=int))
    coil100vars.codebook_colors = np.asarray(coil100vars.codebook_colors)
    return


def coding_and_pooling():
    """
    Strategies:
    1- (hist_hard) coil100vars.codebook_histograms + img_histogram + coding_hard
    2- (pixels_hard) coil100vars.codebook_colors + img_pixels + coding_hard + pooling_sum
    """
    logging.info('Performing coding and pooling')

    count = 0
    for img in coil100vars.imgs:
        # hist_hard
        img['feature_vector_hist_hard'] = np.zeros((coil100vars.codebook_histograms.shape[0],), dtype=int)
        nearest_histogram_codeword = np.argmin(euclidean_distances(coil100vars.codebook_histograms, img['rgb_hist']))
        img['feature_vector_hist_hard'][nearest_histogram_codeword] = 1

        # pixels_hard
        img['feature_vector_pixels_hard'] = np.zeros((coil100vars.codebook_colors.shape[0],), dtype=int)
        total_pixels = img['rgb'].shape[0] * img['rgb'].shape[1]
        flattened_pixels = img['rgb'].reshape((total_pixels, 3))
        random_pixels_indixes = np.sort(np.random.choice(total_pixels, size=int(0.1*total_pixels), replace=False))
        distances = euclidean_distances(flattened_pixels[random_pixels_indixes], coil100vars.codebook_colors)
        color_codewords_to_increment = np.argmin(distances, axis=1)

        # quick way to increment vectors based on
        # http://stackoverflow.com/questions/2004364/increment-numpy-array-with-repeated-indices
        increment_bins = np.bincount(color_codewords_to_increment)
        color_codewords_to_increment.sort()
        incr_values = increment_bins[np.nonzero(increment_bins)]
        incr_indixes = np.unique(color_codewords_to_increment)
        img['feature_vector_pixels_hard'][incr_indixes] += incr_values / len(random_pixels_indixes)

        count += 1
        if count % 100 == 0:
            logging.info('Processed: %s/%s' % (count, len(coil100vars.imgs)))

    return


def train(feature_vectors):
    # TODO
    return None


def execute_query(trained_model, query_feature_vector):
    # TODO
    return []# ['obj1__0', 'obj1__10', ...]
