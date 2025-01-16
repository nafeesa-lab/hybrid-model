import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os, time, pickle
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
from sklearn.metrics import roc_auc_score
import config


def get_roc_auc_score(y_true, y_probs):
    with open(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path), 'rb') as handle:
        all_classes = pickle.load(handle)

    class_roc_auc_list = [roc_auc_score(y_true[:, i], y_probs[:, i]) for i in range(y_true.shape[1])]
    return np.mean(class_roc_auc_list)

def make_plot(epoch_train_loss, epoch_val_loss, total_train_loss_list, total_val_loss_list, save_name):
    fig = plt.figure(figsize=(16, 16))
    fig.suptitle('Loss Trends', fontsize=20)

    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.plot(epoch_train_loss)
    ax1.set_title('Epoch Train Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')

    ax2.plot(epoch_val_loss)
    ax2.set_title('Epoch Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')

    ax3.plot(total_train_loss_list)
    ax3.set_title('Batch Train Loss')
    ax3.set_xlabel('Batches')
    ax3.set_ylabel('Loss')

    ax4.plot(total_val_loss_list)
    ax4.set_title('Batch Validation Loss')
    ax4.set_xlabel('Batches')
    ax4.set_ylabel('Loss')

    plt.savefig(os.path.join(config.models_dir, f'losses_{save_name}.png'))

def train_epoch(device, train_loader, model, loss_fn, optimizer):
    model.train()
    running_loss = 0

    for batch_idx, (img, target) in enumerate(train_loader):
        img, target = img.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img.size(0)

    return running_loss / len(train_loader.dataset)

def validate_epoch(device, val_loader, model, loss_fn):
    model.eval()
    running_loss = 0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for img, target in val_loader:
            img, target = img.to(device), target.to(device)
            output = model(img)

            loss = loss_fn(output, target)
            running_loss += loss.item() * img.size(0)

            all_probs.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    roc_auc = get_roc_auc_score(all_targets, all_probs)

    return running_loss / len(val_loader.dataset), roc_auc

def apply_augmentation_to_classes(dataset, selected_classes, augmentation_fn, all_classes=None):
    # If `all_classes` is not provided, try to get it from the original dataset
    if all_classes is None:
        # Assuming `dataset.dataset` gives the original dataset
        all_classes = dataset.dataset.all_classes  # Access all_classes from the original dataset
    
    augmented_data = []
    for img, target in dataset:
        # Only apply augmentation if the target contains any of the selected classes
        if any(target[all_classes.index(cls)] > 0 for cls in selected_classes):
            img = augmentation_fn(img)  # Apply augmentation to image
        augmented_data.append((img, target))
    return augmented_data


def fit(device, train_loader, val_loader, model, loss_fn, optimizer, num_epochs, save_interval, all_classes, log_interval):
    selected_classes = ["Mass", "Nodule", "Hernia", "Edema", "Cardiomegaly", "Pleural_Thickening", 
                        "Fibrosis", "Consolidation", "Emphysema", "Pneumothorax", "Pneumonia"]

    # Make sure all_classes is passed correctly
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Apply augmentation only to the selected classes for the training set
        augmented_train_loader = apply_augmentation_to_classes(
    train_loader.dataset,
    selected_classes,
    config.augmentation_transforms,  # Augmentation function from config
    all_classes=None  # Pass None to use the default behavior of fetching `all_classes`
)
        # Wrap the augmented data in a DataLoader
        augmented_train_loader = torch.utils.data.DataLoader(augmented_train_loader, 
                                                              batch_size=train_loader.batch_size, 
                                                              shuffle=True)

        # Train for one epoch
        train_loss = train_epoch(device, augmented_train_loader, model, loss_fn, optimizer)
        
        # Validate after each epoch
        val_loss, val_roc_auc = validate_epoch(device, val_loader, model, loss_fn)

        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation ROC AUC: {val_roc_auc:.4f}")

        # Save the model periodically
        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(config.models_dir, f'model_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
