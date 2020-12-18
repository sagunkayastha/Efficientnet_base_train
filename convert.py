import torch

import torch.nn as nn
from efficientnet_pytorch import EfficientNet

# class Pollen(nn.Module):
#     def __init__(self, num_classes, num_features):
#         super(Pollen, self).__init__()
#         self.num_classes = num_classes
#         self.num_features = num_features
#         self.features = EfficientNet.from_pretrained('efficientnet-b1')
#         self.features._fc = nn.Linear(self.num_features,self.num_classes)
#         # self.features._fc = nn.Sequential(nn.Linear(self.num_features, 128),
#         #                          nn.BatchNorm1d(128),
#         #                          nn.ReLU(),
#         #                          nn.Dropout(p=0.3),
#         #                          nn.Linear(128, 56),
#         #                          nn.BatchNorm1d(56),
#         #                          nn.ReLU(),
#         #                          nn.Dropout(p=0.3),
#         #                          nn.Linear(56, self.num_classes))


#     def forward(self, img_data):
#         output = self.features(img_data)

# model = Pollen(20, 1280)
# model = EfficientNet.from_pretrained('efficient-b2')
model = torch.load('efficientnet-b1.pth').cpu()
model.eval()
# # model = EfficientNet.from_pretrained('full.pth')
# checkpoint = torch.load('checkpoint_b1.pth')
# model.load_state_dict(checkpoint['state_dict'])
dummy_input = torch.randn(1, 3, 256, 256).cpu()

model.set_swish(memory_efficient=False)
torch.onnx.export(model, dummy_input, "test-b1.onnx", verbose=True)
