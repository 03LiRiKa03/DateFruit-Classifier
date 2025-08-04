import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

labels_df = pd.read_csv('train_labels.csv')

label_map = {label: idx for idx, label in enumerate(labels_df['label'].unique())}
inv_label_map = {idx: label for label, idx in label_map.items()}

class DateDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, train=True):
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform
        self.train = train
        if self.train:
            self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.train:
            label = self.label_map[self.df.iloc[idx, 1]]
            return image, label
        else:
            return image, img_name

BATCH_SIZE = 33
NUM_EPOCHS = 7
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = DateDataset(dataframe=labels_df, image_dir='train', transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, len(label_map))
model = model.to(DEVICE)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for inputs, labels in loop:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loop.set_postfix(loss=running_loss/total, accuracy=100.*correct/total)

torch.save(model.state_dict(), 'date_classifier.pth')


test_files = os.listdir('test')
test_df = pd.DataFrame(test_files, columns=['filename'])
test_dataset = DateDataset(test_df, 'test', transform=transform_train, train=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model.eval()
predictions = []
with torch.no_grad():
    for inputs, fnames in tqdm(test_loader, desc='Predicting'):
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()
        for fname, pred in zip(fnames, preds):
            predictions.append({'filename': fname, 'label': inv_label_map[pred]})


submission_df = pd.DataFrame(predictions)
submission_df.to_csv('submission.csv', index=False)

