# config.py
pkl_dir_path = 'pickles'
train_val_df_pkl_path = 'train_val_df.pickle'
test_df_pkl_path = 'test_df.pickle'
disease_classes_pkl_path = 'disease_classes.pickle'
models_dir = 'models'

from torchvision import transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Augmentation pipeline
augmentation_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(15),
    transforms.RandomRotation(10),
    transforms.RandomRotation(5),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.ToTensor(),
    normalize
])

# Validation and Test pipeline
transform = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.Resize(224),
    transforms.ToTensor(),
    normalize
])

