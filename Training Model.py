!pip install torch torchvision timm matplotlib


import os
import pandas as pd
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt


num_classes = 7
num_epochs = 15
learning_rate = 0.001
device = torch.device('cpu')


class FER2013Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        self.image_paths = []

        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            class_files = os.listdir(class_path)
            self.image_paths.extend([os.path.join(class_path, file) for file in class_files])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("RGB")

        class_name = os.path.basename(os.path.dirname(img_path))
        label = self.class_to_idx[class_name]

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


train_dataset = FER2013Dataset(root_dir=r'training dataset directory', transform=transform)
test_dataset = FER2013Dataset(root_dir=r'testing dataset directory', transform=transform)


train_dataloader = DataLoader(train_dataset, batch_size=200,shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=10,shuffle=True)


len(train_dataloader)


len(test_dataloader)


class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(EmotionRecognitionModel, self).__init__()
        self.backbone = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.backbone.num_features, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


model = EmotionRecognitionModel(num_classes).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

losses = []
accuracies = []


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = CustomCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    model.train()

    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_dataloader)
    epoch_accuracy = 100 * correct_predictions / total_samples
    losses.append(epoch_loss)
    accuracies.append(epoch_accuracy)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")


torch.save(model.state_dict(), 'model save path')


model = CustomCNN(num_classes).to(device)
saved_model_state = torch.load('model save path')
new_state_dict = model.state_dict()
for name, param in saved_model_state.items():
    if name in new_state_dict:
        new_state_dict[name].copy_(param)
model.load_state_dict(new_state_dict)
model.eval()


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')


plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy')
plt.show()


model.eval()
true_labels = [0]
predicted_labels = [0]
true_labels = [int(label) for label in true_labels]
predicted_labels = [int(label) for label in predicted_labels]

for images, labels in test_dataloader:
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)

    _, preds = torch.max(outputs, 1)

    true_labels.extend(labels.cpu().numpy())
    predicted_labels.extend(preds.cpu().numpy())


labels


true_labels


predicted_labels


from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(true_labels, predicted_labels)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

class_names = train_dataset.classes
classification_rep = classification_report(true_labels, predicted_labels, target_names=class_names)
print(classification_rep)


true_labels


predicted_labels


accuracy = accuracy_score(true_labels, predicted_labels)
print(f'Test Accuracy: {accuracy * 100:.2f}%')


image_path = r'input image path'
external_image = Image.open(image_path).convert("RGB")
input_image = transform(external_image).unsqueeze(0)


model.eval()


with torch.no_grad():
    output = model(input_image)


predicted_class = torch.argmax(output, dim=1).item()


predicted_class


print(f"Predicted class: {class_names[predicted_class]}")
plt.imshow(external_image)
plt.axis('off')
plt.show()