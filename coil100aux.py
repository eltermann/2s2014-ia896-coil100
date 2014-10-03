import cv2
import logging
import glob
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
    """
    logging.info('Reading images to memory')
    orig_imgs = {}
    for file_path in glob.glob(files_pattern):
        objid, imgid = get_objid_and_imgid(file_path)
        if not objid in orig_imgs:
            orig_imgs[objid] = {}
        orig_imgs[objid][imgid] = cv2.imread(file_path)
    return orig_imgs


def build_codebook(orig_imgs):
    # TODO
    return []


def coding_and_pooling(orig_imgs, codebook_coordinates):
    # TODO
    return {}


def train(feature_vectors):
    # TODO
    return None


def execute_query(trained_model, query_feature_vector):
    # TODO
    return []# ['obj1__0', 'obj1__10', ...]
