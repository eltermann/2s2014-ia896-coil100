import cv2
import logging
import os
import signal
import sys

import coil100aux


COIL100_PATH = './coil-100/*.png'
# configure "logging" module when and how to display log messages
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def signal_handler(signal, frame):
    # Ctrl+C pressed
    print('\n')
    logging.info('Execution finished')
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    orig_imgs = coil100aux.load_orig_imgs(COIL100_PATH)
    """
    e.g:
    orig_imgs = {
        'obj1': {
            '0': <numpy array representing obj1__0.png>,
            '10': <numpy array representing obj1__10.png>,
            [etc.]
        },
        [etc.]
    }
    """

    codebook_coordinates = coil100aux.build_codebook(orig_imgs)
    """
    e.g:
    codebook_coordinates = [nd.array([0, 0, 0]), nd.array([10, 20, 30]), ...]
    """

    feature_vectors = coil100aux.coding_and_pooling(orig_imgs, codebook_coordinates)
    """
    e.g:
    feature_vectors = {
        'obj1': {
            '0': <feature vector>,
            '10': <feature vector>,
            [etc.]
        },
        [etc.]
    }
    """

    trained_model = coil100aux.train(feature_vectors)
    # depends on training step

    while True:
        query_file = raw_input('\nFile name to query (e.g: ./coil-100/obj99__90.png): ')
        query_feature_vector = None # TODO - use aux methods from coil100aux
        results = coil100aux.execute_query(trained_model, query_feature_vector)
        # e.g: ['obj1__0', 'obj1__10', ...]

        query_objid, query_imgid = coil100aux.get_objid_and_imgid(query_file)
        target = '%s__' % (query_objid) # e.g: "obj1__"
        hits = 0
        # compute object hits on first 20 returned objects
        for result in results[:20]:
            if result.find(target) == 0:
                hits += 1
        print('Hits: %s/20' % (hits))
