import cv2
import logging
import os
import signal
import sys

import coil100aux


COIL100_PATH = './coil-100/*.png'
COIL100_BINS = 32
COIL100_CODEBOOK_SIZE = 50

# configure "logging" module when and how to display log messages
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def signal_handler(signal, frame):
    # Ctrl+C pressed
    print('\n')
    logging.info('Execution finished')
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    coil100aux.load_imgs(COIL100_PATH) # read images into coil100aux.imgs
    coil100aux.calc_histograms(COIL100_BINS) # append histograms
    coil100aux.build_codebook(COIL100_CODEBOOK_SIZE, 'kmeans') # build codebook into coil100aux.codebook
    coil100aux.coding_and_pooling()

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
