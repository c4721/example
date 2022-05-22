import numpy as np

import torch
import torch.nn as nn

class DL_Model(torch.nn.Model):
    def __init__(self):
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        generate a gray image
        '''
        x = torch.full((256, 256, 3), 0.5)
        x = self.relu(x)
        return x

class Text2Image():
    def __init__():
        self.model = DL_Model()

    def process(self, data):
        '''
        data is not used in exmaple case
        '''

        self.model()


def create_solver(name):
    if name == 'text2image':
        return Text2Image()
    else:
        raise NotImplementedError(f'method({name}) is not implemented')


class Inference_Engine():

    def __init__(self):
        return

    def init(self, gpu_id=None, task='text2image'):
        '''
        Resource allocation
        - sercerch if gpu exist
        - find a unused GPU or set by user defined gpu_id
        - raise IOError if all GPU are unavaliable
        '''
        self.resource_allocation()
        self.solver = create_solver(name=task)

    def self.resource_allocation(self):
        # Resource allocation
        pass
        
    def process(data):
        return self.solver(data)

