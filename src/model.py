'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

from openvino.inference_engine import IECore
import logging
import cv2
from abc import ABC, abstractmethod


class EdgeModel(ABC):
    '''
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        DONE: Use this to set your instance variables.
        '''
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.core = None
        self.network = None
        self.exec_net = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.output_shape = None

    def _get_layers(self):
        print(self.device)
        print(self.network)
        supported_layers = self.core.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]

        return supported_layers, unsupported_layers

    def _check_unsupported_layers(self):
        supported_layers, unsupported_layers = self._get_layers()
        if len(unsupported_layers) != 0 and self.device == 'CPU':
            logging.info("Adding cpu_extension")
            if self.extensions is not None:
                self.core.add_extension(self.extensions, self.device)

                supported_layers, unsupported_layers = self._get_layers()
                if len(unsupported_layers) != 0:
                    logging.error("Unsupported layers found even after adding extension: {}".format(self.extensions))
            else:
                logging.error("No extensions available for addition.")
        return unsupported_layers

    def load_model(self):
        '''
        DONE: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.core = IECore()
        print(self.core)
        try:
            self.network = self.core.read_network(self.model_structure, self.model_weights)
            print(self.network)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        unsupported_layers = self._check_unsupported_layers()

        if len(unsupported_layers) != 0:
            logging.error("Unsupported layers found: {}".format(unsupported_layers))
            exit(1)

        self.exec_net = self.core.load_network(network=self.network, device_name=self.device, num_requests=1)

        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_names = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_names].shape

    @abstractmethod
    def predict(self, image):
        '''
        DONE: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        pass

    @abstractmethod
    def check_model(self):
        pass

    def preprocess_input(self, image):

        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''

        processed_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))  # note width first
        processed_image = processed_image.transpose((2, 0, 1))
        processed_image = processed_image.reshape(1, *processed_image.shape)  # note height first
        return processed_image

    @abstractmethod
    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''

        pass
