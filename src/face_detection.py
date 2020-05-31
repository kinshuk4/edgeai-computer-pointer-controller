'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from src.utils import *

import numpy as np
from openvino.inference_engine import IECore
from src.model import EdgeModel



class FaceDetectionModel(EdgeModel):
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, threshold, device='CPU', extensions=None):
        '''
        DONE: Use this to set your instance variables.
        '''
        super().__init__(model_name, device, extensions)
        self.threshold = threshold
        self.initial_frame_height = None
        self.initial_frame_width = None

    def load_model(self):
        '''
        DONE: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        return super().load_model()

    def predict(self, image):
        '''
        DONE: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_image = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.input_name: processed_image})

        coords = self.preprocess_output(outputs)
        if len(coords) == 0:
            return 0, 0

        coords, cropped_face = self._crop_face(coords, image)
        return cropped_face, coords

    def _crop_face(self, coords, image):
        coords = coords[0]  # select first detected face
        h = image.shape[0]
        w = image.shape[1]
        coords = coords * np.array([w, h, w, h])
        coords = coords.astype(np.int32)
        cropped_face = image[coords[1]:coords[3], coords[0]:coords[2]]
        return coords, cropped_face

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        return super().preprocess_input(image)

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        coords = []
        selected_outputs = outputs[self.output_names][0][0]
        for obj in selected_outputs:
            conf = obj[2]
            if conf >= self.threshold:
                xmin = obj[3]
                ymin = obj[4]
                xmax = obj[5]
                ymax = obj[6]
                coords.append((xmin, ymin, xmax, ymax))
        return coords

    def set_initial_frame_size(self, width, height):
        self.initial_frame_width = width
        self.initial_frame_height = height