# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo which runs object detection on camera frames using GStreamer.

Run default object detection:
python3 detect.py

Choose different camera and input encoding
python3 detect.py --videosrc /dev/video1 --videofmt jpeg

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
import gstreamer
import os
import time

CAMERA_ANGLE_VISION = 84
MID_CAMERA_ANGLE_VISION = CAMERA_ANGLE_VISION / 2
SEC_PANIC_TIME = 10

from motor import Motor
from common import avg_fps_counter, SVG
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference


def generate_svg(src_size, inference_box, objs, labels, text_lines):
    svg = SVG(src_size)
    src_w, src_h = src_size
    box_x, box_y, box_w, box_h = inference_box
    scale_x, scale_y = src_w / box_w, src_h / box_h
    out_range = src_w /3
    out_left = out_range
    out_right = src_w - out_range

    for y, line in enumerate(text_lines, start=1):
        svg.add_text(10, y * 20, line, 20)
    cam_ok = False
    left_mov = False
    right_mov = False

    for obj in objs:
        cam_ok = True
        bbox = obj.bbox
        if not bbox.valid:
            continue
        # Absolute coordinates, input tensor space.
        x, y = bbox.xmin, bbox.ymin
        w, h = bbox.width, bbox.height
        # Subtract boxing offset.
        x, y = x - box_x, y - box_y
        # Scale to source coordinate space.
        x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
        left_mov = x < out_left
        right_mov = x+w > out_right
        #print (f"x={x}, y={y}, w={w}, h={h}, out_left={out_left}, out_right={out_right}, box_w={box_w}, left_mov={left_mov}, right_mov={right_mov}")
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
        svg.add_text(x, y - 5, label, 20)
        svg.add_rect(x, y, w, h, 'red', 2)


    svg.add_controls(left=left_mov, cam_ok=cam_ok, right=right_mov, bounds=(out_left, out_right))
    return svg.finish()

def objects_analysis(inference_box, objs, labels):
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

    for obj in objs:
        objects = True
        bbox = obj.bbox
        if not bbox.valid:
            continue
        # Abheadlesssolute coordinates, input tensor space.
        #print("bbox object",bbox)
        w, h = bbox.width, bbox.height
        x = round(bbox.xmin + w/2) 
        y = round(bbox.ymin + h/2)
        # Subtract boxing offset.
        # x, y = x - box_x, y - box_y

        # Centro del balón - inicio pantalla
        d = (x+w/2)
        #angle = 2*d / box_w * MID_CAMERA_ANGLE_VISION
        angle = round(d * CAMERA_ANGLE_VISION / box_w) - MID_CAMERA_ANGLE_VISION

    state = {'x': x, 'y': y, 'w': w, 'h': h,
                'd': round(d,2), 'angle': round(angle)}
    
    return state


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
    parser.add_argument('--headless', action='store_true', help='No screen output')
    args = parser.parse_args()

    if args.headless:
        print("Running in headless mode")

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    # Average fps over last 30 frames.
    fps_counter = avg_fps_counter(30)
    last_detection_time = time.monotonic()
    last_angle = 0
    motor = Motor(inverted=True, degree_to_move=5)
    _ = input("Pulse <intro> para iniciar el proceso")



    def user_callback(input_tensor, src_size, inference_box, headless=False):
        nonlocal fps_counter
        nonlocal last_detection_time
        nonlocal motor
        start_time = time.monotonic()
        run_inference(interpreter, input_tensor)
        # For larger input image sizes, use the edgetpu.classification.engine for better performance
        objs = get_objects(interpreter, args.threshold)[:args.top_k]
        end_time = time.monotonic()
        fps = round(next(fps_counter))
        text_lines = [
            'Inference: {:.2f} ms'.format((end_time - start_time) * 1000),
            'FPS: {} fps'.format(fps),
        ]
        
        #print(' '.join(text_lines))
        if objs:
            state = objects_analysis(inference_box, objs, labels)
            last_detection_time = end_time
            last_pos = motor.pos
            motor.rotate(state['angle']/4)
            if last_pos != motor.pos:
                print("FPS", fps, "state",state)
        else:
            if (start_time - last_detection_time) > SEC_PANIC_TIME:
                if not motor.scanning:
                    print(' '.join(text_lines))
                    motor.scan()

        if headless:
            return None
        else:
            return generate_svg(src_size, inference_box, objs, labels, text_lines)
        
    print("inference_size", inference_size)
    result = gstreamer.run_pipeline(user_callback,
                                    src_size=(640, 480),
                                    appsink_size=inference_size,
                                    videosrc=args.videosrc,
                                    videofmt=args.videofmt,
                                    headless=args.headless)

    print("fin del proceso")
    motor.close()

if __name__ == '__main__':
    main()
