'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from model import EdgeModel
import cv2
import numpy as np
import math


class GazeEstimationModel(EdgeModel):
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
        DONE: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        EdgeModel.load_model(self)
        self.input_name = [i for i in self.network.inputs.keys()]
        self.input_shape = self.network.inputs[self.input_name[1]].shape
        self.output_names = [i for i in self.network.outputs.keys()]

    def predict(self, left_eye_image, right_eye_image, head_pose_angles):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_left_image = self.preprocess_input(left_eye_image.copy())
        processed_right_image = self.preprocess_input(right_eye_image.copy())

        outputs = self.exec_net.infer({
            'head_pose_angles': head_pose_angles,
            'left_eye_image': processed_left_image,
            'right_eye_image': processed_right_image
        })
        new_mouse_coord, gaze_vector = self.preprocess_output(outputs, head_pose_angles)

        return new_mouse_coord, gaze_vector

    def check_model(self):
        raise NotImplementedError

    def preprocess_output(self, outputs, head_pose_angles):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        gaze_vector = outputs[self.output_names[0]].tolist()[0]

        roll_value = head_pose_angles[2]
        cos_theta = math.cos(roll_value * math.pi / 180.0)
        sin_theta = math.sin(roll_value * math.pi / 180.0)

        x_new = gaze_vector[0] * cos_theta + gaze_vector[1] * sin_theta
        y_new = -gaze_vector[0] * sin_theta + gaze_vector[1] * cos_theta
        return (x_new, y_new), gaze_vector
