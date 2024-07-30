# predict.py
import torch
from torchvision import models, transforms
from PIL import Image
import json

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg19(weights='DEFAULT')
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(image)
    return transform(img)

def predict_image(image_path, checkpoint, top_k, category_names, gpu):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    
    model = load_checkpoint(checkpoint)
    model.to(device)
    model.eval()
    
    image = process_image(image_path)
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.exp(output)
        top_probs, top_indices = probabilities.topk(top_k)
    
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]
    
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_names = [cat_to_name[cls] for cls in top_classes]
    else:
        top_names = top_classes
    
    for prob, name in zip(top_probs, top_names):
        print(f"{name}: {prob:.3f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Path to image')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()
    
    predict_image(args.image_path, args.checkpoint, args.top_k, args.category_names, args.gpu)