import cv2
import logging
import glob
import numpy as np
import os
import re
from sklearn.cluster import KMeans


# globally used variables
imgs = None
codebook = None
histograms_matrix = None

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
    global imgs
    imgs = []
    for file_path in glob.glob(files_pattern):
        objid, imgid = get_objid_and_imgid(file_path)
        imgs.append({
            'objid': objid,
            'imgid': imgid,
            'rgb': cv2.imread(file_path),
        })
    return


def calc_histograms(bins):
    """Given the original images and the number of bins to split the vector-space,
    compute the histograms in the following format:

    histogram = <3-dimension numpy array (3, COIL100_BINS)>
    E.g (assuming bins=4):
    histogram = [
        [0.20, 0.40, 0.30, 0.10], # channel 1 frequencies
        [0.10, 0.10, 0.75, 0.05], # channel 2 frequencies
        [0.25, 0.25, 0.30, 0.20], # channel 3 frequencies
    ]
    """
    global imgs, histograms_matrix
    if not isinstance(imgs, list):
        raise ValueError('Called calc_histograms() before load_imgs()')

    logging.info('Calculating histograms')
    histograms_matrix = []
    for img in imgs:
        img['rgb_hist'] = load_color_histogram(img['rgb'], bins)
        histograms_matrix.append(img['rgb_hist'].flatten())
    histograms_matrix = np.asarray(histograms_matrix)
    return


def load_color_histogram(img, bins):
    """For a given image, calculates its normalized frequencies histograms.
    `img` is supposed to contain 3 channels
    Output is a 2-dimensional numpy array with shape=(3,bins)
    """
    total_pixels = img.shape[0] * img.shape[1]
    return np.asarray([
        cv2.calcHist([img], [0], None, [bins], [0,256])[:,0] / total_pixels,
        cv2.calcHist([img], [1], None, [bins], [0,256])[:,0] / total_pixels,
        cv2.calcHist([img], [2], None, [bins], [0,256])[:,0] / total_pixels,
    ])


def build_codebook(coordinates_num=50, strategy='random'):
    """Build codebook from given images and corresponding histograms.
    `coordinates_num` represents the number of coordinates to return
    `strategy` is one of: "random" or "kmeans"
    Output: 2-dimensional numpy array with shape=(coordinates_num, 3)
    E.g (assuming coordinates_num=2):
    codebook = [
        [128, 128, 128],
        [014, 197, 255],
    ]
    """
    global histograms_matrix, codebook
    if not isinstance(histograms_matrix, np.ndarray):
        raise ValueError('Called build_codebook() before loading `imgs`')

    logging.info('Building codebook; strategy="%s"' % (strategy))

    if coordinates_num > histograms_matrix.shape[0]:
        raise ValueError('Codebook: unable to retrieve %s coordinates from %s images' % (coordinates_num, histograms_aux.shape[0]))

    bins = histograms_matrix.shape[1] / 3
    weights = np.arange(start=256/(2*bins), stop=255, step=256/bins)
    # e.g (assuming bins = 8):
    # weights = [ 15  46  77 108 139 170 201 232]
    weights = np.concatenate((weights, weights, weights)) # 3 channels concatenated

    if strategy == 'random':
        random_rows_indexes = np.random.randint(histograms_matrix.shape[0], size=coordinates_num)
        random_histograms = histograms_matrix[random_rows_indexes,:]
        codebook = []
        for hist in random_histograms:
            codebook.append(np.sum((hist * weights).reshape((3, bins)), axis=1, dtype=int))
        codebook = np.asarray(codebook)

    elif strategy == 'kmeans':
        estimator = KMeans(init='k-means++', n_clusters=coordinates_num)
        estimator = estimator.fit(histograms_matrix)
        codebook = []
        for hist in estimator.cluster_centers_:
            codebook.append(np.sum((hist * weights).reshape((3, bins)), axis=1, dtype=int))
        codebook = np.asarray(codebook)
    else:
        raise ValueError('Invalid codebook strategy: %s' % (strategy))
    return


def coding_and_pooling(orig_imgs, codebook_coordinates):
    # TODO
    return {}


def train(feature_vectors):
    # TODO
    return None


def execute_query(trained_model, query_feature_vector):
    # TODO
    return []# ['obj1__0', 'obj1__10', ...]
