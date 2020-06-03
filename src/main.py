import logging
from argparse import ArgumentParser
import os
import cv2

from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksDetectionModel
from gaze_estimation import GazeEstimationModel
from head_pose_estimation import HeadPoseEstimationModel
from input_feeder import InputFeeder
from mouse_controller import MouseController


def get_parser():
    """
    Parse command line arguments.
    :return: Command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fm", "--face-detection-model", required=True, type=str,
                        help="Path to folder where model exists")
    parser.add_argument("-hm", "--head-pose-model", required=True, type=str,
                        help="Path to folder where model exists")
    parser.add_argument("-lm", "--facial-landmarks-model", required=True, type=str,
                        help="Path to folder where model exists")
    parser.add_argument("-gm", "--gaze-estimation-model", required=True, type=str,
                        help="Path to folder where model exists")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to input video. Use 'cam' for capturing video stream from camera")
    parser.add_argument("-l", "--cpu_extension", type=str, default=None,
                        help="MKLDNN (CPU)-targeted custom layers. Absolute path to shared lib with the kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on; "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Looks"
                             "for a suitable plugin for device specified"
                             "(CPU by default)")
    parser.add_argument("-t", "--threshold", type=float, default=0.5,
                        help="Probability threshold for detections")
    parser.add_argument("-o", "--output_dir", type=str, default=None, help="Path to output directory")

    return parser


def init_feeder(args):
    input_feeder = None
    if args.input.lower() == "cam":
        input_feeder = InputFeeder("cam")
    else:
        if not os.path.isfile(args.input):
            logging.error("Unable to find specified video file")
            exit(1)
        input_feeder = InputFeeder("video", args.input)
    return input_feeder


def load_all_models(args):
    model_path_dict = {
        'FaceDetectionModel': args.face_detection_model,
        'FacialLandmarksDetectionModel': args.facial_landmarks_model,
        'GazeEstimationModel': args.gaze_estimation_model,
        'HeadPoseEstimationModel': args.head_pose_model
    }
    for fileNameKey in model_path_dict.keys():
        if not os.path.isfile(model_path_dict[fileNameKey] + ".xml"):
            logging.error("Unable to find specified " + fileNameKey + " xml file")
            exit(1)
    fdm = FaceDetectionModel(model_path_dict['FaceDetectionModel'], args.threshold, args.device, args.cpu_extension)
    fldm = FacialLandmarksDetectionModel(model_path_dict['FacialLandmarksDetectionModel'], args.threshold, args.device,
                                         args.cpu_extension)
    gem = GazeEstimationModel(model_path_dict['GazeEstimationModel'], args.threshold, args.device, args.cpu_extension)
    hpem = HeadPoseEstimationModel(model_path_dict['HeadPoseEstimationModel'], args.threshold, args.device,
                                   args.cpu_extension)

    fdm.load_model()
    fldm.load_model()
    gem.load_model()
    hpem.load_model()
    return fdm, fldm, gem, hpem


def main():
    args = get_parser().parse_args()
    input_feeder = init_feeder(args)
    fdm, fldm, gem, hpem = load_all_models(args)
    mc = MouseController('medium', 'fast')

    input_feeder.load_data()

    frame_count = 0
    for frame in input_feeder.next_batch():
        frame_count += 1
        if frame_count % 5 == 0:
            cv2.imshow('video', cv2.resize(frame, (500, 500)))

        key = cv2.waitKey(60)

        if key == 27:
            break

        cropped_face, face_coords = fdm.predict(frame.copy())
        logging.info(cropped_face)
        logging.info(face_coords)

    logging.error("VideoStream ended...")
    cv2.destroyAllWindows()
    input_feeder.close()


if __name__ == '__main__':
    main()
