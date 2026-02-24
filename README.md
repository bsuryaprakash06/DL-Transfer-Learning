# DL - Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Include the problem statement and Dataset


## Neural Network Model
<img width="811" height="542" alt="image" src="https://github.com/user-attachments/assets/3d200f54-72c8-48cb-855c-6810da69fe8d" />

## DESIGN STEPS
## Step 1: Data Loading and Preprocessing
1. Define image transformations including resizing and tensor conversion.
2. Load the training and testing datasets using `ImageFolder` from a structured dataset directory.
3. Create `DataLoader` objects for batch processing.
4. Optionally, display sample images to verify the dataset.

## Step 2: Model Setup and Modification
1. Load a pre-trained VGG19 model from `torchvision.models`.
2. Modify the final fully connected layer to match the number of classes (1 for binary classification).
3. Freeze the feature extractor layers to perform transfer learning.
4. Move the model to GPU if available.

## Step 3: Model Training
1. Define the loss function (`BCEWithLogitsLoss`) and optimizer (`Adam`).
2. Iterate over multiple epochs to train the model:
   - Forward pass, compute loss, backward pass, and update weights.
   - Track training and validation loss for each epoch.
3. After each epoch, evaluate the model on the validation set.
4. Plot the training and validation losses to monitor performance.

## Step 4: Model Evaluation and Prediction
1. Evaluate the model on the test dataset:
   - Compute predictions, test accuracy, confusion matrix, and classification report.
2. Visualize the confusion matrix using `seaborn`.
3. Make predictions on single images:
   - Apply sigmoid to output logits and threshold at 0.5.
   - Display the image with actual and predicted labels.

## PROGRAM

### Name: Surya Prakash B

### Register Number: 212224230281

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
## Step 1: Load and Preprocess Data
# Define transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for pre-trained model input
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization for pre-trained models
])
# Load dataset from a folder (structured as: dataset/class_name/images)
dataset_path = "./data/dataset/"
train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)
# Display some input images
def show_sample_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(5, 5))
    for i in range(num_images):
        image, label = dataset[i]
        image = image.permute(1, 2, 0)  # Convert tensor format (C, H, W) to (H, W, C)
        axes[i].imshow(image)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")
    plt.show()
# Show sample images from the training dataset
show_sample_images(train_dataset)
# Get the total number of samples in the training dataset
print(f"Total number of training samples: {len(train_dataset)}")

# Get the shape of the first image in the dataset
first_image, label = train_dataset[0]
print(f"Shape of the first image: {first_image.shape}")

# Get the total number of samples in the testing dataset
print(f"Total number of test samples: {len(test_dataset)}")

# Get the shape of the first image in the dataset
first_image1, label = test_dataset[0]
print(f"Shape of the first image: {first_image1.shape}")

# Create DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# Load Pretrained Model and Modify for Transfer Learning
model = models.vgg19(weights = models.VGG19_Weights.DEFAULT)
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
from torchsummary import summary
# Print model summary
summary(model, input_size=(3, 224, 224))

# Modify the final fully connected layer to match the dataset classes
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
summary(model, input_size=(3, 224, 224))
# Freeze all layers except the final layer
for param in model.features.parameters():
    param.requires_grad = False  # Freeze feature extractor layers

# Include the Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

## Step 3: Train the Model
def train_model(model, train_loader, test_loader, num_epochs=100):
  train_losses = []
  val_losses = []
  model.train()
  for epoch in range(num_epochs):
      train_loss = 0.0
      for images, labels in train_loader:
          images, labels = images.to(device), labels.to(device)
          optimizer.zero_grad()
          outputs = model(images)
          loss = criterion(outputs, labels.unsqueeze(1).float())
          loss.backward()
          optimizer.step()
          train_loss += loss.item()
      
      # Calculate average training loss for the epoch
      avg_train_loss = train_loss / len(train_loader)
      train_losses.append(avg_train_loss)

      # Validation loop
      model.eval() # Set model to evaluation mode
      val_loss = 0.0
      with torch.no_grad():
          for images, labels in test_loader:
              images, labels = images.to(device), labels.to(device)
              outputs = model(images)
              loss = criterion(outputs, labels.unsqueeze(1).float())
              val_loss += loss.item()
      
      # Calculate average validation loss for the epoch
      avg_val_loss = val_loss / len(test_loader)
      val_losses.append(avg_val_loss)
      
      model.train() # Set model back to training mode

      print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
  
  return train_losses, val_losses # Return the losses

# Define num_epochs for the training run
num_epochs_to_run = 10 

# Call the train_model function and get the losses
trained_train_losses, trained_val_losses = train_model(model, train_loader, test_loader, num_epochs=num_epochs_to_run)

# Plot training and validation loss
print("Name: Surya Prakash B")
print("Register Number: 212224230281")
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs_to_run + 1), trained_train_losses, label='Train Loss', marker='o')
plt.plot(range(1, num_epochs_to_run + 1), trained_val_losses, label='Validation Loss', marker='s')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
## Step 4: Test the Model and Compute Confusion Matrix & Classification Report
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # For binary classification with BCEWithLogitsLoss, apply sigmoid and threshold
            predicted = (torch.sigmoid(outputs) > 0.5).squeeze().long()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Name: Surya Prakash B")
    print("Register Number: 212224230281")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Print classification report
    print("Name: Surya Prakash B")
    print("Register Number: 212224230281")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

# Call the test_model function to run evaluation
test_model(model, test_loader)
## Step 5: Predict on a Single Image and Display It
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    with torch.no_grad():
        image_tensor = image.unsqueeze(0).to(device)
        output = model(image_tensor)

        # Apply sigmoid to get probability, threshold at 0.5
        prob = torch.sigmoid(output)
        predicted = (prob > 0.5).int().item()


    class_names = dataset.classes
    # Display the image
    image_to_display = transforms.ToPILImage()(image)
    plt.figure(figsize=(4, 4))
    plt.imshow(image_to_display)
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted]}')
    plt.axis("off")
    plt.show()

    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted]}')
# Example Prediction
predict_image(model, image_index=55, dataset=test_dataset)



```

## OUTPUT

## Sample images from training dataset
<img width="407" height="109" alt="image" src="https://github.com/user-attachments/assets/7f35263f-93ff-421f-a3f9-53cf6a9c785b" />

## Training Loss, Validation Loss Vs Iteration Plot
<img width="701" height="211" alt="image" src="https://github.com/user-attachments/assets/ad07237e-4281-40f4-806d-4f512241b7bf" />
<img width="700" height="547" alt="image" src="https://github.com/user-attachments/assets/faa974e3-6f0e-4397-b1e8-b54eaa46eab6" />

## Confusion Matrix
<img width="640" height="50" alt="image" src="https://github.com/user-attachments/assets/8ddf4ef3-bbde-41c3-a029-b5485a0e52b1" />
<img width="640" height="547" alt="image" src="https://github.com/user-attachments/assets/8d4818ce-9e3d-49ec-b110-e17ec261cfd7" />

## Classification Report
<img width="642" height="196" alt="image" src="https://github.com/user-attachments/assets/4bf96ea0-3c93-45da-85e5-a07c2dc09bcc" />

## New Sample Data Prediction
<img width="328" height="371" alt="image" src="https://github.com/user-attachments/assets/73a4ca97-b818-4e2d-88a3-499e6258eead" />

## RESULT
Test Accuracy: <b> 0.95 </b> <br>
Binary classification achieved with VGG19; confusion matrix and classification report indicate strong performance on both classes.
