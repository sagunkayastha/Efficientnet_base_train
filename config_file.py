from torchvision import transforms

image_shape = 256
train_batch_size = 4
test_val_batch_size = 2
epochs=10
model_name='efficientnet-b1'
learning_rate = 0.0001
print_freq = 5

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

