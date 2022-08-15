import triton_python_backend_utils as pb_utils
import numpy as np
import json
import cv2
import asyncio

class TritonPythonModel:

    def initialize(self, args):
        model_name = args["model_name"]
        self.namespace = model_name[:model_name.rfind(".") + 1] if "." in model_name else ""
        self.model_config = json.loads(args['model_config'])
        self.threshold = '0.4'

        self.det_height, self.det_width = 300, 300
        self.rec_height, self.rec_width = 224, 224


    async def execute(self, requests):
        responses = []
        
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get input tensors
            req_data = pb_utils.get_input_tensor_by_name(request, 'IMAGE_BINARY')
            req_desc = pb_utils.get_input_tensor_by_name(request, "IMAGE_DESC")
            req_json = json.loads(req_desc.as_numpy()[0])
            
            cid = req_json.get('client_id', 'unknown id')
            
            # Get image matrix (NHWC)
            img_det_cv2_list, input_size_list = get_image_mat(req_data, req_desc, self.det_height, self.det_width)
            print("detect image 0 origin size: {}, shape :{}".format(input_size_list[0], img_det_cv2_list[0].shape))
            
            ################################################################################
            # Custom process logic for specific inference model
            ################################################################################

            # Reinterpret matrix, this is implemented as a workaround for (maybe) the bug in ensemble model
            out = {
                'client_id': cid,
                'results': []
            }
            
            inference_response_awaits = []
            for i in range(len(img_det_cv2_list)):
                inference_response_awaits.append(
                    self.generate_async_request(img_det_cv2_list, input_size_list, req_json, out, i))
        
            # Wait for all the inference requests to finish. The execution
            # of the Python script will be blocked until all the awaitables
            # are resolved.
            await asyncio.gather(
                *inference_response_awaits)


            print("result :{}".format(out))
            json_out = np.array([json.dumps(out)], dtype=np.object_)
            out_tensor_0 = pb_utils.Tensor("RESULT", json_out)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')

    async def generate_async_request(self, img_det_cv2_list, input_size_list, req_json, out, i):
        img = np.ascontiguousarray(np.expand_dims(img_det_cv2_list[i], axis=0))
        model_name_string = self.namespace + "spark_detection"

        in_0 = pb_utils.Tensor("image_tensor", img)
        infer_request = pb_utils.InferenceRequest(
            model_name=model_name_string,
            requested_output_names=["detection_boxes", "detection_scores", "detection_classes"],
            inputs=[in_0])
        
        infer_response = await infer_request.async_exec()
        id = req_json.get('params')[i].get('id', 'unknown id')
        threshold = float(req_json.get('params')[i].get('threshold', self.threshold))
        
        if infer_response.has_error():
            raise pb_utils.TritonModelException(infer_response.error().message())
        det_boxes = pb_utils.get_output_tensor_by_name(infer_response, "detection_boxes")
        det_scores = pb_utils.get_output_tensor_by_name(infer_response, "detection_scores")
        det_classes = pb_utils.get_output_tensor_by_name(infer_response, "detection_classes")

        detection_boxes = det_boxes.as_numpy()[0]
        detection_scores = det_scores.as_numpy()[0]
        detection_classes = det_classes.as_numpy()[0]
        
        print("spark_detection objects bbox: {}".format(detection_boxes))
        print("spark_detection objects class: {}".format(detection_classes))
        spark_objects = []
        
        max_index = np.argmax(detection_scores, axis=0)
        print("spark_detection objects maxscore [{}]: {}".format(max_index, detection_scores[max_index]))

        if detection_scores[max_index] >= threshold:
            box = detection_boxes[max_index]
            x1 = int(box[1] * input_size_list[i][0])
            x2 = int(box[3] * input_size_list[i][0])
            y1 = int(box[0] * input_size_list[i][1])
            y2 = int(box[2] * input_size_list[i][1])
            spark_objects.append({
                'attributes': [
                    {
                    'key': 'name',
                    'value': 'roi',
                    'desc': 'interest area'
                    },
                ],
                'position': [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2]                      
                ]
            })
        result = {
            'id': id,
            'objects': spark_objects
        }
        # build response json
        out['results'].append(result)

    def spark_detection(self, img):
        model_name_string = self.namespace + "spark_detection"

        in_0 = pb_utils.Tensor("image_tensor", img)
        infer_request = pb_utils.InferenceRequest(
            model_name=model_name_string,
            requested_output_names=["detection_boxes", "detection_scores", "detection_classes"],
            inputs=[in_0])

        # Perform synchronous blocking inference request
        infer_response = infer_request.exec()
        if infer_response.has_error():
            raise pb_utils.TritonModelException(infer_response.error().message())
        
        det_boxes = pb_utils.get_output_tensor_by_name(infer_response, "detection_boxes")
        det_scores = pb_utils.get_output_tensor_by_name(infer_response, "detection_scores")
        det_classes = pb_utils.get_output_tensor_by_name(infer_response, "detection_classes")

        return det_boxes, det_scores, det_classes


    def spark_recognition(self, img):
        print("spark_recognition")

        in_0 = pb_utils.Tensor("input_image", img)
        model_name_string = self.namespace + "spark_recognition"

        infer_request = pb_utils.InferenceRequest(
            model_name=model_name_string,
            requested_output_names=["output_classes", "output_scores"],
            inputs=[in_0])

        # Perform synchronous blocking inference request
        infer_response = infer_request.exec()
        if infer_response.has_error():
            raise pb_utils.TritonModelException(infer_response.error().message())
        
        output_classes = pb_utils.get_output_tensor_by_name(infer_response, "output_classes")
        output_scores = pb_utils.get_output_tensor_by_name(infer_response, "output_scores")

        return output_classes, output_scores

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def get_image_mat_jpg(img_data, height, width):
    img_mat = cv2.imdecode(img_data, cv2.IMREAD_COLOR) # decode image
    img_out = cv2.resize(img_mat, (width, height), interpolation = cv2.INTER_LINEAR) # resize
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB) # BGR -> RGB    
    return img_out, np.array([img_mat.shape[1], img_mat.shape[0]], dtype='int32')

def get_image_mat_raw(img_data, img_meta, height, width):
    shape = img_meta.get('shape')
    if len(shape) != 3:
        raise Exception('image shape not correctly set. is {}, should like [300, 300, 3]'.format(shape))
    img_mat = img_data.reshape(shape)
    img_out = cv2.resize(img_mat, (width, height), interpolation = cv2.INTER_LINEAR) # resize
    return img_out, np.array([img_mat.shape[1], img_mat.shape[0]], dtype='int32')

def get_image_mat(req_data, req_desc, height, width):
    req_json = json.loads(req_desc.as_numpy()[0])
    # img_data = np.frombuffer(req_data.as_numpy()[0], dtype=np.uint8)
    req_data = req_data.as_numpy()
    img_meta = req_json.get('params')[0]
    img_type = img_meta.get('type')

    # print(req_json)
    # print(len(img_data))
    img_mat_list = []
    img_size_list = []
    for i in range(len(req_data)):
        img_data = np.frombuffer(req_data[i], dtype=np.uint8)
        if img_type == 'jpg' or img_type == 'jpeg':
            img_mat, img_size = get_image_mat_jpg(img_data, height, width)
        if img_type == 'raw':
            img_mat, img_size =get_image_mat_raw(img_data, img_meta, height, width)
        img_mat_list.append(img_mat)
        img_size_list.append(img_size)
    return img_mat_list, img_size_list

    raise Exception('image type {} is not supported'.format(img_type))