import os
import time
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import ToTensor
from config_file import *
from efficientnet_pytorch import EfficientNet
from PIL import Image
import numpy as np
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Pollen(nn.Module):
    def __init__(self, num_classes, num_features):
        super(Pollen, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.features = EfficientNet()
        self.features._fc = nn.Linear(self.num_features,self.num_classes)
        # self.features._fc = nn.Sequential(nn.Linear(self.num_features, 128),
        #                          nn.BatchNorm1d(128),
        #                          nn.ReLU(),
        #                          nn.Dropout(p=0.3),
        #                          nn.Linear(128, 56),
        #                          nn.BatchNorm1d(56),
        #                          nn.ReLU(),
        #                          nn.Dropout(p=0.3),
        #                          nn.Linear(56, self.num_classes))


    def forward(self, img_data):
        output = self.features(img_data)

        return output

# model = Pollen(20,1280)
img = Image.open('/home/ubuntu/sagun/Efficientnet_base_train/data_split/test/QUE/20191120T213513894-292-16.png').convert('RGB')
# img = np.array(img)
# img = np.expand_dims(img,0)
# img = Image.fromarray(img)
# checkpoint = torch.load('checkpoint.pth')
# model.load_state_dict(checkpoint['state_dict'])
model = torch.load('efficientnet-b2.pth')


tfms = transforms.Compose([transforms.Resize(image_shape), 
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
img = tfms(img)
img = img.unsqueeze(0)
model.eval()

with torch.no_grad():
    for i in range(200):
        logits = model(img.to(device))
preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()
for idx in preds:
    prob = torch.softmax(logits, dim=1)[0, idx].item()
    print('{:<75} ({:.2f}%)'.format(idx, prob*100))