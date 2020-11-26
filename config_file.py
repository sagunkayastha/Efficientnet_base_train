from torchvision import transforms
num_classes = 20
image_shape = 256
train_batch_size = 64
test_val_batch_size = 32
epochs=15
model_name='efficientnet-b1'
learning_rate = 0.00001
print_freq = 10
save_filename_f = model_name+'.pth'
transform_train = transforms.Compose([
    transforms.Resize((image_shape,image_shape)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

transform_test = transforms.Compose([
    transforms.Resize((image_shape,image_shape)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

'''
1 ║   32 ║   32 ║   40 ║   48 ║   48 ║   56 ║   64 ║
║     2 ║   16 ║   16 ║   24 ║   24 ║   24 ║   32 ║   32 ║
║     3 ║   24 ║   24 ║   32 ║   32 ║   40 ║   40 ║   48 ║
║     4 ║   40 ║   48 ║   48 ║   56 ║   64 ║   72 ║   80 ║
║     5 ║   80 ║   88 ║   96 ║  112 ║  128 ║  144 ║  160 ║
║     6 ║  112 ║  120 ║  136 ║  160 ║  176 ║  200 ║  224 ║
║     7 ║  192 ║  208 ║  232 ║  272 ║  304 ║  344 ║  384 ║
║     8 ║  320 ║  352 ║  384 ║  448 ║  512 ║  576 ║  640 ║
║     9 ║ 1280 ║ 1408 ║ 1536 ║ 1792 ║ 2048 ║ 2304 ║ 2560 
'''
features_ = {'efficientnet-b1':1280,
            'efficientnet-b2':1408,
            'efficientnet-b3':1536
}