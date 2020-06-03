'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from model import EdgeModel
import cv2
import numpy as np


class FacialLandmarksDetectionModel(EdgeModel):
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
        return EdgeModel.load_model(self)

    def predict(self, image):
        '''
        DONE: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        processed_image = self.preprocess_input(image.copy())
        outputs = self.exec_net.infer({self.input_name: processed_image})
        coords = self.preprocess_output(outputs)

        h = image.shape[0]
        w = image.shape[1]
        coords = coords * np.array([w, h, w, h])
        coords = coords.astype(np.int32)

        l_xmax, l_xmin, l_ymax, l_ymin = self._get_eye_coords(coords, 0, 1)

        r_xmax, r_xmin, r_ymax, r_ymin = self._get_eye_coords(coords, 2, 3)

        left_eye = image[l_ymin:l_ymax, l_xmin:l_xmax]
        right_eye = image[r_ymin:r_ymax, r_xmin:r_xmax]

        eye_coords = [[l_xmin, l_ymin, l_xmax, l_ymax], [r_xmin, r_ymin, r_xmax, r_ymax]]

        return left_eye, right_eye, eye_coords

    def _get_eye_coords(self, coords, i, j):
        xmin = coords[i] - 10
        ymin = coords[j] - 10
        xmax = coords[i] + 10
        ymax = coords[j] + 10
        return xmax, xmin, ymax, ymin

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_cvt, (self.input_shape[3], self.input_shape[2]))
        img_processed = np.transpose(np.expand_dims(image_resized, axis=0), (0, 3, 1, 2))
        return img_processed

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        outs = outputs[self.output_names][0]
        x_coord_left = outs[0].tolist()[0][0]
        y_coord_left = outs[1].tolist()[0][0]
        x_coord_right = outs[2].tolist()[0][0]
        y_coord_right = outs[3].tolist()[0][0]

        return x_coord_left, y_coord_left, x_coord_right, y_coord_right
