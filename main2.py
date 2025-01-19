import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import XRaysTrainDataset, XRaysTestDataset
from trainer import fit
import config
import torch.nn.functional as F
from torchinfo import summary
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np


from sklearn.metrics import roc_auc_score
from scipy.special import expit  # For sigmoid conversion if needed

def test_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_preds = []

    with torch.no_grad():  # Disable gradient computation
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Get model predictions
            outputs = model(images)  # Raw logits
            probs = expit(outputs.cpu().numpy())  # Apply sigmoid to get probabilities

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(probs)

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Calculate ROC AUC scores
    try:
        avg_auc = roc_auc_score(all_labels, all_preds, average="macro", multi_class="ovr")
        auc_per_class = roc_auc_score(all_labels, all_preds, average=None)
        
        print(f"Average ROC AUC Score (macro): {avg_auc:.4f}")
        print(f"ROC AUC Score Per Class: {auc_per_class}")
    except ValueError as e:
        print(f"Error during ROC AUC calculation: {e}")
        avg_auc, auc_per_class = None, None

    return avg_auc, auc_per_class


# ConvNeXt Block
class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, dim, drop_path=0.0):
        super(ConvNeXtBlock, self).__init__()
        # Handle channel mismatch with pointwise convolution
        self.input_proj = nn.Conv2d(in_channels, dim, kernel_size=1) if in_channels != dim else nn.Identity()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, stride=1, groups=dim)  # Depthwise convolution
        self.norm = nn.GroupNorm(1, dim)  # GroupNorm over the channel dimension
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1, stride=1)  # Pointwise convolution
        self.act = nn.GELU()  # Activation
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1, stride=1)  # Pointwise convolution
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0.0 else nn.Identity()  # Stochastic depth

    def forward(self, x):
        # Adjust input channels if needed
        x = self.input_proj(x)
        shortcut = x

        # Depthwise convolution
        x = self.dwconv(x)

        # Apply GroupNorm over the channel dimension
        x = self.norm(x)

        # Pointwise convolution and activation
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Drop path (if any) and residual connection
        x = self.drop_path(x)
        return shortcut + x


# ResNet Block
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


# ResNet Hybrid Model
class ResNetHybrid(nn.Module):
    def __init__(self, layers, num_classes=15):
        super(ResNetHybrid, self).__init__()
        self.inplanes = 64  # Starting number of channels

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks for the first two layers
        self.layer1 = self._make_layer(ResNetBlock, 64, layers[0])
        self.layer2 = self._make_layer(ResNetBlock, 128, layers[1], stride=2)

        # ConvNeXt blocks for the last two layers (updated channels)
        self.layer3 = self._make_layer(ConvNeXtBlock, 256, layers[2], stride=2, is_convnext=True)
        self.layer4 = self._make_layer(ConvNeXtBlock, 512, layers[3], stride=2, is_convnext=True)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, is_convnext=False):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes) if not is_convnext else nn.Identity()
            )

        layers = []
        # Handle first block with downsample if needed
        if is_convnext:
            layers.append(block(self.inplanes, planes))
        else:
            layers.append(block(self.inplanes, planes, stride, downsample))

        # Update inplanes to the new number of channels
        self.inplanes = planes
        # Add subsequent blocks
        for _ in range(1, blocks):
            if is_convnext:
                layers.append(block(planes, planes))
            else:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolution and max pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Pass through each layer
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling and fully connected layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# Layer configuration for each stage of training
def configure_layers_for_training(model, stage):
    print(f"Configuring layers for Stage {stage}...")
    for name, param in model.named_parameters():
        if stage == 1:
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        elif stage == 2:
            if 'layer3' in name or 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        elif stage == 3:
            if 'layer2' in name or 'layer3' in name or 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        elif stage == 4:
            param.requires_grad = True  # Unfreeze all layers


# Count the parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Argument parser setup
parser = argparse.ArgumentParser(description='Arguments for the script')
parser.add_argument('--data_path', type=str, default='NIH Chest X-rays', help='Path of the training data')
parser.add_argument('--bs', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate for the optimizer')
parser.add_argument('--stage', type=int, default=1, help='Stage to decide which layers to train')
parser.add_argument('--loss_func', type=str, default='FocalLoss', choices={'BCE', 'FocalLoss'}, help='Loss function')
parser.add_argument('-r', '--resume', action='store_true', help='Resume training')
parser.add_argument('--ckpt', type=str, help='Path to the checkpoint to load')
parser.add_argument('-t', '--test', action='store_true', help='Test the model')
args = parser.parse_args()

def q(text=''):  # Easy way to exit the script
    print('> ', text)
    sys.exit()

if args.resume and args.test:  # Prevent both resume and test flags being active simultaneously
    q('The flow of this code has been designed either to train the model or to test it.\nPlease choose either --resume or --test')

# Automatically set stage to 1 if starting fresh
stage = args.stage
if not args.resume:
    print(f'\nOverwriting stage to 1, as the model training is being done from scratch')
    stage = 1

if args.test:
    print('TESTING THE MODEL')
else:
    if args.resume:
        print('RESUMING THE MODEL TRAINING')
    else:
        print('TRAINING THE MODEL FROM SCRATCH')

import os, time

script_start_time = time.time()

# Data loading
data_dir = os.path.join('data', args.data_path)
XRayTrain_dataset = XRaysTrainDataset(data_dir, transform=config.augmentation_transforms)
train_percentage = 0.8
train_dataset, val_dataset = torch.utils.data.random_split(
    XRayTrain_dataset,
    [int(len(XRayTrain_dataset) * train_percentage), len(XRayTrain_dataset) - int(len(XRayTrain_dataset) * train_percentage)]
)
XRayTest_dataset = XRaysTestDataset(data_dir, transform=config.transform)

print('\n-----Initial Dataset Information-----')
print('num images in train_dataset   : {}'.format(len(train_dataset)))
print('num images in val_dataset     : {}'.format(len(val_dataset)))
print('num images in XRayTest_dataset: {}'.format(len(XRayTest_dataset)))
print('-------------------------------------')

batch_size = args.bs
# Resampling the dataset and creating dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(XRayTest_dataset, batch_size=batch_size, shuffle=False)

print('\n-----Initial Batchloaders Information -----')
print('num batches in train_loader: {}'.format(len(train_loader)))
print('num batches in val_loader  : {}'.format(len(val_loader)))
print('num batches in test_loader : {}'.format(len(test_loader)))
print('-------------------------------------------')

# Sanity check
if len(XRayTrain_dataset.all_classes) != 15:  # 15 is the unique number of diseases in this dataset
    q('\nnumber of classes not equal to 15 !')

# Initialize models directory for saving
if not os.path.exists(config.models_dir):
    os.mkdir(config.models_dir)

# Loss function setup
if args.loss_func == 'FocalLoss':
    from losses import FocalLoss
    loss_fn = FocalLoss(device=device, gamma=2.0).to(device)
else:
    loss_fn = nn.BCEWithLogitsLoss().to(device)

lr = args.lr

# Initialize the hybrid model
if not args.test:
    if not args.resume:
        print('\nInitializing ResNet Hybrid (ResNet + ConvNeXt) model...')
        model = ResNetHybrid(layers=[3, 3, 6, 3], num_classes=len(XRayTrain_dataset.all_classes))
        model.to(device)
        num_params = count_parameters(model)
        print(f'Total Trainable Parameters: {num_params:.2f} million')

        # Configure layers for the initial training stage
        configure_layers_for_training(model, stage)

        # Initialize optimizer
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        epochs_till_now = 0
        losses_dict = {'epoch_train_loss': [], 'epoch_val_loss': [], 'total_train_loss_list': [], 'total_val_loss_list': []}
    else:
        if not args.ckpt:
            sys.exit('Checkpoint required to resume training')
        print(f'Loading checkpoint: {args.ckpt}')
        ckpt = torch.load(os.path.join(config.models_dir, args.ckpt))
        model = ckpt['model']
        model.to(device)
        epochs_till_now = ckpt['epochs']
        losses_dict = ckpt['losses_dict']

    # Training for the specific stage
    fit(device=device,
    train_loader=train_loader,
    val_loader=val_loader,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    num_epochs=10,
    save_interval=1,
    all_classes=XRayTrain_dataset.all_classes)


   # Test the model
print("--- Testing the Model ---")
avg_auc, auc_per_class = test_model(model, test_loader, device)
if avg_auc is not None:
    print(f"Test Average ROC AUC Score: {avg_auc:.4f}")
else:
    print("ROC AUC Score could not be calculated.")

