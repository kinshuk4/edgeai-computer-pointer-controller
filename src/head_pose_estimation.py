'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from model import EdgeModel
import cv2
import numpy as np


class HeadPoseEstimationModel(EdgeModel):
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, threshold, device='CPU', extensions=None):
        '''
        DONE: Use this to set your instance variables.
        '''
        super().__init__(model_name, threshold, device, extensions)

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        return EdgeModel.load_model(self)

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_image = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.input_name: processed_image})
        return self.preprocess_output(outputs)

    def check_model(self):
        raise NotImplementedError

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        outs = [outputs['angle_y_fc'].tolist()[0][0],
                outputs['angle_p_fc'].tolist()[0][0],
                outputs['angle_r_fc'].tolist()[0][0]]
        return outs
