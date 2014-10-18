import cv2
import logging
import os
import signal
import sys
import re

import coil100aux
import coil100vars


COIL100_PATH = './coil-100/*.png'
COIL100_BINS = 32
COIL100_CODEBOOK_SIZE = 200


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
    coil100aux.build_codebooks(COIL100_CODEBOOK_SIZE, 'random')
    coil100aux.coding_and_pooling()

    while True:
        query_file = raw_input('\nFile name to query (e.g: ./coil-100/obj99__90.png): ')
        #query de teste: obj1__20.png
        
        #recupera o indice corresponde a query
        query_objid, query_imgid = coil100aux.get_objid_and_imgid(query_file)        
        query_objid = coil100aux.get_num(query_objid)
        query = (query_objid-1)*72 + ((int(query_imgid))/5)

        #query = 4
        #nao usar 3 com ed        
        coding_kind = int(raw_input('\nChoose the codebook you want to use:\n 1: RGB histogram. Its the frequence in what each of the 32 cluster occurs, 32 for color,\n 2: Codewords are colors. They are applied on a simple of pixels, using sum as pooling,\n 3: feature_vector_hist_hard codewords.\n'))
        proximity_by = raw_input('\nChoose the proximity function (Use "ed" for Euclidean distance or "md" for Manhattan distance): ')
       
        #results = vetor com os indices das imagens mais parecidas de acordo com a funcao de distancia e codebook escolhidos       
        results = coil100aux.search_query(coil100vars.imgs, coil100vars.imgs[query], proximity_by,coil100vars.rank_size,coding_kind)
        # e.g: ['obj1__0', 'obj1__10', ...]

#        query_objid, query_imgid = coil100aux.get_objid_and_imgid(query_file)
#        target = '%s__' % (query_objid) # e.g: "obj1__"
#        hits = 0
#        # compute object hits on first 20 returned objects
#        for result in results:
#            if result.find(target) == 0:
#                hits += 1
#        print('Hits: %s/20' % (hits))
