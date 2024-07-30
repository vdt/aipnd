# main.py
import argparse
import os
from train import train_model
from predict import predict_image

def main():
    parser = argparse.ArgumentParser(description="Flower Classification CLI")
    subparsers = parser.add_subparsers(dest="command", help="train or predict")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new network")
    train_parser.add_argument("data_dir", help="Path to the flower dataset")
    train_parser.add_argument("--arch", default="vgg19", help="Model architecture")
    train_parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units")
    train_parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    train_parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict flower name from image")
    predict_parser.add_argument("image_path", help="Path to image")
    predict_parser.add_argument("checkpoint", help="Path to checkpoint")
    predict_parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    predict_parser.add_argument("--category_names", help="Path to JSON file mapping categories to real names")
    predict_parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()

    if args.command == "train":
        train_model(args.data_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)
    elif args.command == "predict":
        predict_image(args.image_path, args.checkpoint, args.top_k, args.category_names, args.gpu)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()