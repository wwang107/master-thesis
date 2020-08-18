from torch.utils.tensorboard import SummaryWriter
from os import path
import torch
import numpy as np
import cv2

class TensorBoardWriter:
    def __init__(self, log_directory=None):
        super().__init__()
        self.log_directory = log_directory
        if log_directory is not None:
            self.trainWriter = SummaryWriter(
                log_dir=path.join(log_directory, 'train'))
            self.validWriter = SummaryWriter(
                log_dir=path.join(log_directory, 'validation'))
    
    def add_scalar(self, mode, loss, step):
        if mode == 'train':
            self.trainWriter.add_scalar('loss', loss, step)
        elif mode == 'valid':
            self.validWriter.add_scalar('loss', loss, step)

    
    def add_images(self, mode, images, step):
        if mode == 'train':
            writer = self.trainWriter
        elif mode == 'valid':
            writer = self.validWriter

        for key in images:
            stack = np.stack(images[key], axis=0)/255
            writer.add_images(key, stack, step, dataformats='NHWC')

    
    
