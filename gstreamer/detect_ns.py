
"""A demo that runs object detection on camera frames using OpenCV.
TEST_DATA=../all_models
Run face detection model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite
Run coco model:
python3 detect.py \
  --model ${TEST_DATA}/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels ${TEST_DATA}/coco_labels.txt
"""

import argparse
import cv2
import os
import time
import math

MIN_DEGREE_TO_MOVE = 5
CAMERA_ANGLE_VISION = 84
MID_CAMERA_ANGLE_VISION = CAMERA_ANGLE_VISION / 2
SEC_PANIC_TIME = 10

from motor import Motor
from common import avg_fps_counter
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

from periphery import Serial, PWM


def objects_analysis(inference_box, objs):
    """
    Realiza los calculos para determinar la posición del balon y 
    la necesidad de movimiento del motor para centrar la escena

    Args:
      inference_box: Tamaño de la imagen analizada
      objs: Lista de objetos detectados
      labels: Diccionario con los nombres de los objetos

    Returns:
      dict: Diccionario con los resultados de la detección
        
        time: Time of analysis
        x,y,w,h: Coordenadas y tamaño del objeto respecto a inference_box
        object_detected: True or False
        angle: Ángulo de giro del motor
    """
    box_x, box_y, box_w, box_h = inference_box
    x, y, w, h = 0, 0, 0, 0
    d, angle = 0, 0
    objects = False

    for obj in objs:
        objects = True
        bbox = obj.bbox
        if not bbox.valid:
            continue
        # Absolute coordinates, input tensor space.
        x, y = bbox.xmin, bbox.ymin
        w, h = bbox.width, bbox.height
        # Subtract boxing offset.
        x, y = x - box_x, y - box_y

        # Centro del balón - centro pantalla.
        d = (x+w/2) - box_w/2
        angle = 2*d / box_w * MID_CAMERA_ANGLE_VISION
    
    return {
        'time': time.monotonic(),
        'x': x, 'y': y, 'w': w, 'h': h,
        'object_detected': objects,
        'd': round(d,2), 'angle': round(angle)
    }


def main():
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    parser.add_argument('--videosrc', help='Which video source to use. ',
                        default='/dev/video0')
    parser.add_argument('--videofmt', help='Input video format.',
                        default='raw',
                        choices=['raw', 'h264', 'jpeg'])
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)
    detect(args.videosrc, args.threshold, args.top_k,
           interpreter, inference_size)


def detect(src, threshold, top_k, interpreter, inference_size):
    # Average fps over last 30 frames.
    fps_counter = avg_fps_counter(30)
    last_detection_time = time.monotonic()
    motor = Motor()

    cap = cv2.VideoCapture(src)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        #cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels)


        start_time = time.monotonic()
        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, threshold)[:top_k]
        end_time = time.monotonic()
        text_lines = [
            'Inference: {:.2f} ms'.format((end_time - start_time) * 1000),
            'FPS: {} fps'.format(round(next(fps_counter))),
        ]
        
        state = objects_analysis(inference_size, objs)
        if objs:
            last_detection_time = end_time
            if abs(state['angle']) > MIN_DEGREE_TO_MOVE:
                motor.rotate(state['angle'])
        else:
            if (start_time - last_detection_time) > SEC_PANIC_TIME:
                motor.scan()


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
