# train.py
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import os
from collections import OrderedDict
from tqdm import tqdm
import time

def load_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    }

    return dataloaders, image_datasets

def build_model(arch, hidden_units):
    if arch == "vgg19":
        model = models.vgg19(weights='DEFAULT')
    else:
        model = models.vgg19(weights='DEFAULT')

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4096)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.5)),
        ('fc2', nn.Linear(4096, hidden_units)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(0.5)),
        ('fc3', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    return model

def train_model(data_dir, arch, learning_rate, hidden_units, epochs, gpu):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    
    dataloaders, image_datasets = load_data(data_dir)
    model = build_model(arch, hidden_units)
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Early stopping parameters
    patience = 3
    best_loss = float('inf')
    no_improve = 0

    print('Training started')
    start_training_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        # Progress bar for training
        with tqdm(dataloaders['train'], unit="batch") as tepoch:
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{epochs}")
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                log_probabilities = model(inputs)
                loss = criterion(log_probabilities, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                tepoch.set_postfix(loss=train_loss/len(tepoch))

        avg_train_loss = train_loss / len(dataloaders['train'])
        print(f'\nEpoch: {epoch + 1}/{epochs}')
        print(f'Training Loss: {avg_train_loss:.4f}')

        model.eval()
        valid_loss = 0
        valid_accuracy = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                log_probabilities = model(inputs)
                loss = criterion(log_probabilities, labels)
                valid_loss += loss.item()

                probabilities = torch.exp(log_probabilities)
                top_probability, top_class = probabilities.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        avg_valid_loss = valid_loss / len(dataloaders['valid'])
        avg_valid_accuracy = valid_accuracy / len(dataloaders['valid'])
        print(f"Validation Loss: {avg_valid_loss:.4f}")
        print(f"Validation Accuracy: {avg_valid_accuracy:.4f}")

        # Early stopping
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            no_improve = 0
            # Save the best model
            checkpoint = {
                'arch': arch,
                'hidden_units': hidden_units,
                'state_dict': model.state_dict(),
                'class_to_idx': image_datasets['train'].class_to_idx
            }
            torch.save(checkpoint, 'best_checkpoint.pth')
        else:
            no_improve += 1
            if no_improve == patience:
                print("Early stopping triggered")
                break

    end_training_time = time.time()
    print('Training ended')
    training_time = end_training_time - start_training_time
    print(f'\nTraining time: {training_time//60:.0f}m {training_time%60:.0f}s')

    return model

def save_checkpoint(model, train_data, save_dir, arch):
    model.class_to_idx = train_data.class_to_idx
    
    checkpoint = {
        'arch': arch,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    
    torch.save(checkpoint, f"{save_dir}/checkpoint.pth")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Path to the flower dataset')
    parser.add_argument('--arch', type=str, default='vgg19', help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    args = parser.parse_args()
    
    model = train_model(args.data_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)
    save_checkpoint(model, model.class_to_idx, '.', args.arch)
