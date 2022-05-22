import genImg

import numpy


if __name__ == '__main__':
    
    '''
    init dl_engin
    - only init once at the begining
    '''
    dl_engine = genImg.Inference_Engine()
    dl_engine.init(task='text2image')

    ''' user upload data
    data -> dict:
    - 'im1': fake input image in (high pixel, width pixel, number of channels - e.g. RGB)
    - 'text': user input text
    '''
    data = {
            'im1': numpy.random.rand(256, 256, 3),
            'text': "An AI walking through Central Park with sakura trees in full bloom, digital art"
            }
    '''
    processing powered by AI
    - output -> dict:
    - 'out1': output image in (high pixel, width pixel, number of channels - e.g. RGB)
           - for example: numpy tensor in shape of (256, 256, 3)
    '''
    output = dl_engine.process(data)

