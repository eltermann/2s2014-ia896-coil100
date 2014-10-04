import cv2
import logging
import glob
import numpy as np
import os
import re


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


def load_orig_imgs(files_pattern):
    """Load ALL images into memory as numpy arrays.
    Returned variable will be a dictionary of dictionaries, in the format:
    {
        'obj1': {
            '0': <3-dimension numpy array (128, 128, 3) representing obj1__0.png>,
            [etc.]
        },
        [etc.]
    }
    """
    logging.info('Reading images to memory')
    orig_imgs = {}
    for file_path in glob.glob(files_pattern):
        objid, imgid = get_objid_and_imgid(file_path)
        if not objid in orig_imgs:
            orig_imgs[objid] = {}
        orig_imgs[objid][imgid] = cv2.imread(file_path)
    return orig_imgs


def load_histograms(orig_imgs, bins):
    """Given the original images and the number of bins to split the vector-space,
    compute and return the histograms in the following format:
    {
        'obj1': {
            '0': <3-dimension numpy array (3, COIL100_BINS) representing frequency histograms of obj1__0.png>,
            [etc.]
        },
        [etc.]
    }
    """
    logging.info('Calculating histograms')
    histograms = {}
    for objid in orig_imgs.keys():
        if not objid in histograms:
            histograms[objid] = {}
        for imgid in orig_imgs[objid].keys():
            histograms[objid][imgid] = load_color_histogram(orig_imgs[objid][imgid], bins)
    return histograms


def flatten_from_dict(histograms):
    """Returns a 2-dimensional numpy array given 2-levels dictionary.
    E.g:
    histograms = {
        'obj1': {
            '0': [
                [1, 2], # channel 1
                [3, 4], # channel 2
                [5, 6], # channel 3
            ],
            ...
        },
    }
    flattened = [
        [1, 2, 3, 4, 5, 6], # all channels concatenated!
        ...
    ]
    """
    histograms_flattened = []
    for objid in histograms.keys():
        for imgid in histograms[objid].keys():
            histograms_flattened.append(histograms[objid][imgid].flatten())
    return np.asarray(histograms_flattened)


def load_color_histogram(img, bins):
    """For a given image, calculates its normalized frequencies histograms.
    `img` is supposed to contain 3 channels
    Output is a 2-dimensional numpy array with shape=(3,bins)
    E.g (assuming bins=4):
    histogram = [
        [0.20, 0.40, 0.30, 0.10], # channel 1 frequencies
        [0.10, 0.10, 0.75, 0.05], # channel 2 frequencies
        [0.25, 0.25, 0.30, 0.20], # channel 3 frequencies
    ]
    """
    total_pixels = img.shape[0] * img.shape[1]
    return np.asarray([
        cv2.calcHist([img], [0], None, [bins], [0,256])[:,0] / total_pixels,
        cv2.calcHist([img], [1], None, [bins], [0,256])[:,0] / total_pixels,
        cv2.calcHist([img], [2], None, [bins], [0,256])[:,0] / total_pixels,
    ])


def build_codebook(imgs, histograms, coordinates_num=50, strategy='random'):
    """Build codebook from given images and corresponding histograms.
    `coordinates_num` represents the number of coordinates to return
    `strategy` is one of: "random" or "knn"
    Output: 2-dimensional numpy array with shape=(coordinates_num, 3)
    E.g (assuming coordinates_num=2):
    codebook = [
        [128, 128, 128],
        [014, 197, 255],
    ]
    """
    logging.info('Building codebook; strategy="%s"' % (strategy))

    # flatten histograms into a matrix - we will lose (objid, imgid) reference
    histograms_flattened = flatten_from_dict(histograms)

    if coordinates_num > histograms_flattened.shape[0]:
        raise ValueError('Codebook: unable to retrieve %s coordinates from %s images' % (coordinates_num, histograms_aux.shape[0]))

    if strategy == 'random':
        random_rows_indexes = np.random.randint(histograms_flattened.shape[0], size=coordinates_num)
        random_histograms = histograms_flattened[random_rows_indexes, :]
        codebook = []
        bins = histograms_flattened.shape[1] / 3
        weights = np.arange(start=1, stop=255, step=255/bins)[:bins]
        # e.g (assuming bins = 16):
        # weights = [1, 16, 31, 46, 61, 76, 91, 106, 121, 136, 151, 166, 181, 196, 211, 226]
        for hist in random_histograms:
            codebook.append([
                np.sum(hist[:bins] * weights, dtype=int), # channel 1
                np.sum(hist[bins:2*bins] * weights, dtype=int), # channel 2
                np.sum(hist[2*bins:] * weights, dtype=int), # channel 3
            ])
        codebook = np.asarray(codebook)
    elif strategy == 'knn':
        # TODO
        raise NotImplementedError('KNN Codebook')
    else:
        raise ValueError('Invalid codebook strategy: %s' % (strategy))

    return codebook


def coding_and_pooling(orig_imgs, codebook_coordinates):
    # TODO
    return {}


def train(feature_vectors):
    # TODO
    return None


def execute_query(trained_model, query_feature_vector):
    # TODO
    return []# ['obj1__0', 'obj1__10', ...]
