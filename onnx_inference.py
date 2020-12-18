import onnxruntime
import time
import torch
from torchvision.transforms import ToTensor
from config_file import *
from PIL import Image
import numpy as np
import cv2
import logging

from inference.classification_test import ClassifierEvaluation


def get_logger(path, name='Training_logger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s : %(message)s","%Y-%m-%d %H:%M:%S")

    handler = logging.FileHandler(path)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger

logger = ('logs/')

evaluation = ClassifierEvaluation('./','data_split/test/',logger,'test-b1')
evaluation.evaluate()                









'''
sess = onnxruntime.InferenceSession('test-b1.onnx')
img_in = Image.open('data_split/test/ACE/20191203T180028410-186-23.png').convert('RGB')

# img_in = cv2.imread('/home/ubuntu/sagun/Efficientnet_base_train/data_split/test/QUE/20191120T213513894-292-16.png')

img_c = cv2.imread('data_split/test/ALN/20191126T234704501-263-5.png')
img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB).astype(np.float32)
img_c = img_c/255.0
img_c -= np.array([0.485, 0.456, 0.406],dtype=np.float32)
img_c /=  np.array([0.229, 0.224, 0.225],dtype=np.float32)
img_c = np.transpose(img_c, (2, 0, 1)).astype(np.float32)
img_c = np.expand_dims(img_c, axis=0)

img_in = Image.open('data_split/test/ALN/20191126T234704501-263-5.png').convert('RGB')
img_in = transform_test(img_in)
img_in = np.array(img_in)
# img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
# print(img_in == img_c)
# exit()
# img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
# img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
img_in = np.expand_dims(img_in, axis=0)

print(img_in.shape)



input_name = sess.get_inputs()[0].name
print("input name", input_name)
input_shape = sess.get_inputs()[0].shape
print("input shape", input_shape)
input_type = sess.get_inputs()[0].type
print("input type", input_type)

input_name = sess.get_inputs()[0].name

a = time.time()
for i in range(1):
    # print('onnx',i)
    outputs = sess.run(None, {input_name: img_c})

    outputs = torch.tensor(outputs).view(1,-1)
print(time.time() - a)
# print(outputs)
# exit()

preds = torch.topk(outputs, k=5).indices.squeeze(0).tolist()
for idx in preds:
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    print('{:<75} ({:.2f}%)'.format(idx, prob*100))
'''