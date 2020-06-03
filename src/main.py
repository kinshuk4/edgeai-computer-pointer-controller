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
import time

GREEN = (0, 255, 0)

MAGENTA = (255, 0, 255)


def get_parser():
    """
    Parse command line arguments.
    :return: Command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fm", "--face-detection-model", required=True, type=str,
                        help="Path to Face Detection model without extension")
    parser.add_argument("-hm", "--head-pose-model", required=True, type=str,
                        help="Path to Head Pose Estimation model without extension")
    parser.add_argument("-lm", "--facial-landmarks-model", required=True, type=str,
                        help="Path to Facial Landmarks Detection model without extension")
    parser.add_argument("-gm", "--gaze-estimation-model", required=True, type=str,
                        help="Path to Gaze Estimation model without extension")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to input video. Use 'cam' for capturing video stream from camera")
    parser.add_argument("-l", "--cpu_extension", type=str, default=None,
                        help="MKLDNN (CPU)-targeted custom layers. Absolute path to shared lib with the kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on; "
                             "Can be: CPU, GPU, FPGA or MYRIAD")
    parser.add_argument("-t", "--threshold", type=float, default=0.5,
                        help="Probability threshold for detections")
    parser.add_argument("-o", "--output-dir", type=str, default=None, help="Path to output directory")
    parser.add_argument("-v", "--show-intermediate-visualization", type=bool, default=True,
                        help="Shows intermediate step visualization")
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
    fd_model = FaceDetectionModel(model_path_dict['FaceDetectionModel'], args.threshold, args.device,
                                  args.cpu_extension)
    fld_model = FacialLandmarksDetectionModel(model_path_dict['FacialLandmarksDetectionModel'], args.threshold,
                                              args.device,
                                              args.cpu_extension)
    ge_model = GazeEstimationModel(model_path_dict['GazeEstimationModel'], args.threshold, args.device,
                                   args.cpu_extension)
    hpe_model = HeadPoseEstimationModel(model_path_dict['HeadPoseEstimationModel'], args.threshold, args.device,
                                        args.cpu_extension)
    start_time = time.time()

    fd_model.load_model()
    fld_model.load_model()
    ge_model.load_model()
    hpe_model.load_model()

    total_model_load_time = time.time() - start_time
    return fd_model, fld_model, ge_model, hpe_model, total_model_load_time


def visualize_intermediate_steps(cropped_face, head_pose_angles, eye_coords, left_eye, right_eye, gaze_vector):
    draw_rectangle(cropped_face, eye_coords, 0)  # left eye
    draw_rectangle(cropped_face, eye_coords, 1)  # right

    cv2.putText(cropped_face,
                "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(head_pose_angles[0], head_pose_angles[1],
                                                                              head_pose_angles[2]), (10, 20),
                cv2.FONT_HERSHEY_COMPLEX, 0.25, GREEN, 1)
    i, j, k = int(gaze_vector[0] * 12), int(gaze_vector[1] * 12), 160
    left_eye_line = draw_line(left_eye, i, j, k)
    right_eye_line = draw_line(right_eye, i, j, k)

    cropped_face[eye_coords[0][1]:eye_coords[0][3], eye_coords[0][0]:eye_coords[0][2]] = left_eye_line
    cropped_face[eye_coords[1][1]:eye_coords[1][3], eye_coords[1][0]:eye_coords[1][2]] = right_eye_line
    cv2.imshow("intermediate-visualization-step", cv2.resize(cropped_face, (500, 500)))


def draw_line(eye, i, j, k):
    eye_line = cv2.line(eye.copy(), (i - k, j - k), (i + k, j + k), MAGENTA, 2)
    cv2.line(eye_line, (i - k, j + k), (i + k, j - k), MAGENTA, 2)
    return eye_line


def draw_rectangle(cropped_face, eye_coords, i):
    cv2.rectangle(cropped_face, (eye_coords[i][0] - 10, eye_coords[i][1] - 10),
                  (eye_coords[i][2] + 10, eye_coords[i][3] + 10), GREEN, 3)


def run_workflow(fd_model, fld_model, ge_model, hpe_model, input_feeder, mc, should_visualize):
    input_feeder.load_data()
    fps = input_feeder.get_fps()

    frame_count = 0
    frame_threshold = 5

    start_inference_time = time.time()

    for frame in input_feeder.next_batch():
        if frame is None:
            break
        frame_count += 1
        if frame_count % frame_threshold == 0:
            cv2.imshow('video', cv2.resize(frame, (500, 500)))

        key = cv2.waitKey(60)
        if key == 27:
            break
        cropped_face, face_coords = fd_model.predict(frame.copy())
        if cropped_face is None:
            logging.error("Unable to detect the face.")
            continue

        head_pose_angles = hpe_model.predict(cropped_face.copy())

        left_eye, right_eye, eye_coords = fld_model.predict(cropped_face.copy())

        new_mouse_coord, gaze_vector = ge_model.predict(left_eye, right_eye, head_pose_angles)

        total_time = time.time() - start_inference_time
        total_inference_time = round(total_time, 1)
        effective_fps = frame_count / total_inference_time

        if should_visualize:
            visualize_intermediate_steps(cropped_face, head_pose_angles, eye_coords, left_eye,
                                         right_eye, gaze_vector)

        if frame_count % frame_threshold == 0:
            mc.move(new_mouse_coord[0], new_mouse_coord[1])

    logging.info("VideoStream ended...")
    cv2.destroyAllWindows()
    input_feeder.close()
    return fps, total_inference_time, effective_fps


def main():
    logging.info("Parsing the arguments.")
    args = get_parser().parse_args()

    logging.info("Arguments parsed successfully. Now initialing feedreader.")
    input_feeder = init_feeder(args)

    logging.info("FeedReader initialized, loading the models.")
    fd_model, fld_model, ge_model, hpe_model, total_model_load_time = load_all_models(args)
    mc = MouseController('medium', 'fast')

    logging.info("Starting the workflow")
    fps, total_inference_time, effective_fps = run_workflow(fd_model, fld_model, ge_model, hpe_model, input_feeder, mc,
                                                            args.show_intermediate_visualization)

    logging.debug("Writing the stats.")
    with open(os.path.join(args.output_dir, 'stats.txt'), 'w') as f:
        f.write(str(total_inference_time) + '\n')
        f.write(str(fps) + '\n')
        f.write(str(effective_fps) + '\n')
        f.write(str(total_model_load_time) + '\n')


if __name__ == '__main__':
    main()
