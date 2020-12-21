import numpy as np
import requests
import json

# 1、 模型运行
# tensorflow_model_server --rest_api_port=8509  --model_name="abc" --model_base_path="/mnt/e/jupyter_tensor/tf2c2/abc_model"


# 2、 查看模型
# saved_model_cli show --dir abc_model/1 --all
# ---------------
# signature_def['serving_default']:
#   The given SavedModel SignatureDef contains the following input(s):
#     inputs['conv2d_input'] tensor_info:
#         dtype: DT_FLOAT
#         shape: (-1, 90, 90, 1)
#         name: serving_default_conv2d_input:0
#   The given SavedModel SignatureDef contains the following output(s):
#     outputs['dense_1'] tensor_info:
#         dtype: DT_FLOAT
#         shape: (-1, 10)
#         name: StatefulPartitionedCall:0
#   Method name is: tensorflow/serving/predict
# ---------------
# input： conv2d_input   method：predict


# 3、 RESTful
test_input = np.random.random((10, 32)).tolist()

data = json.dumps({"signature_name": "serving_default", "instances": test_input})
print('Data: {} ... {}'.format(data[:50], data[len(data) - 52:]))

headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8509/v1/models/abc:predict', data=data, headers=headers)
print(json_response.text)
predictions = json.loads(json_response.text)['predictions']
print(predictions)
