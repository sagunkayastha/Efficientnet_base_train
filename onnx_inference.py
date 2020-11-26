import onnxruntime
import time
import torch
from torchvision.transforms import ToTensor
from config_file import *
from PIL import Image
import numpy as np
import cv2

sess = onnxruntime.InferenceSession('test-b1.onnx')
img_in = Image.open('/home/ubuntu/sagun/Efficientnet_base_train/data_split/test/PIN/20191205T170714172-29-17.png').convert('RGB')

# img_in = cv2.imread('/home/ubuntu/sagun/Efficientnet_base_train/data_split/test/QUE/20191120T213513894-292-16.png')


tfms = transforms.Compose([transforms.Resize(image_shape), 
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
img = tfms(img_in)

img_in = np.array(img)
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
    outputs = sess.run(None, {input_name: img_in})

    outputs = torch.tensor(outputs).view(1,-1)
print(time.time() - a)
# print(outputs)
# exit()

preds = torch.topk(outputs, k=5).indices.squeeze(0).tolist()
for idx in preds:
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    print('{:<75} ({:.2f}%)'.format(idx, prob*100))