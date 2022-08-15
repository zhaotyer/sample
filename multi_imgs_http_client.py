#!/usr/bin/env python3

from tritonclient.utils import *
import tritonclient.http as httpclient
import json
import numpy as np
import cv2
import argparse
import os

DEF_MODEL_NAME = 'predict'
DEF_URL = '127.0.0.1:8000'


def get_files(filepath, exts=None):
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    for parent, dirnames, filenames in os.walk(filepath):
        for filename in filenames:
            if exts is None:
                files.append(os.path.join(parent, filename))
            else:
                for ext in exts:
                    if filename.endswith(ext):
                        files.append(os.path.join(parent, filename))
                        break
    files = sorted(files)
    return files


def make_input_jpg_desc(index):
    params = {
        'id': str(index),
        'type': 'jpeg',
        'data': 'IMAGE_BINARY'
    }

    return params


def make_input_raw_desc(img_shape, index):
    params = {
        'id': str(index),
        'type': 'raw',
        'data': 'IMAGE_BINARY',
        'shape': img_shape
    }

    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--url', type=str, required=False, default=DEF_URL,
                        help='Inference server URL. Default is {}'.format(DEF_URL))
    parser.add_argument('-m', '--model', type=str, required=False, default=DEF_MODEL_NAME,
                        help='Model name. Default is {}'.format(DEF_MODEL_NAME))
    parser.add_argument('-f', '--format', type=str, required=False, default='jpg',
                        help='Image format in transmission')
    parser.add_argument('-v', '--verbose', action='store_true', required=False, default=False,
                        help='Output verbose info')
    parser.add_argument('image_filename', type=str, nargs='+', default=None,
                        help='Input image / Input folder.')
    FLAGS = parser.parse_args()

    with httpclient.InferenceServerClient(FLAGS.url, verbose=FLAGS.verbose) as client:
        # model_metadata: dict = client.get_model_metadata(model_name=FLAGS.model)
        # if FLAGS.verbose:
        #     print('model config:', model_metadata)

        if len(FLAGS.image_filename) == 1 and os.path.isdir(FLAGS.image_filename[0]):
            img_files = get_files(FLAGS.image_filename[0], ['jpg', 'jpeg', 'png', 'PNG'])
        else:
            img_files = FLAGS.image_filename

        image_bytes_list = []
        params_list = []
        for index, img_file in enumerate(img_files):
            with open(img_file, 'rb') as f:
                img_bytes = f.read()

                if FLAGS.format == 'jpg' or FLAGS.format == 'jpeg':
                    image_bytes_list.append(img_bytes)
                    params = make_input_jpg_desc(index)
                    params_list.append(params)
                else:
                    img_mat = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    image_bytes_list.append(img_mat.tobytes())
                    img_shape = list(img_mat.shape)
                    params = make_input_raw_desc(img_shape, index)
                    params_list.append(params)

        img_binary = np.array(image_bytes_list, dtype=np.object_)
        img_json = {
            'client_id': 'example',
            'params':params_list
        }
        img_json = np.array([json.dumps(img_json)], dtype=np.object_)
        if FLAGS.verbose:
            print('request binary size: {}'.format(img_binary.shape[0]))
            print('request json: {}'.format(img_json))

        # make inputs
        inputs = [
            httpclient.InferInput('IMAGE_BINARY', img_binary.shape, np_to_triton_dtype(img_binary.dtype)),
            httpclient.InferInput('IMAGE_DESC', img_json.shape, np_to_triton_dtype(img_json.dtype))
        ]
        inputs[0].set_data_from_numpy(img_binary)
        inputs[1].set_data_from_numpy(img_json)

        # make outputs
        outputs = [
            httpclient.InferRequestedOutput("RESULT")
        ]
        import time
        start = time.time()
        response = client.infer(FLAGS.model,
                                inputs,
                                request_id=str(1),
                                outputs=outputs)
        print("infer use time is:{0}".format(time.time()-start))

        out_json = response.as_numpy('RESULT')
        out_str = out_json[0].decode()
        json_res = json.loads(out_str)
        b_json = json.dumps(json_res, indent=4, ensure_ascii=False)
        print(b_json)
        print(len(out_str))
