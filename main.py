import argparse
import sys

from src.train import train_model
from src.generate import generate_text
from src.visualize import generate_demo_charts


def main():
    parser = argparse.ArgumentParser(description="Char-level LSTM Memorizer PoC")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    train_parser = subparsers.add_parser(
        "train", help="Train the LSTM model on a text file"
    )
    train_parser.add_argument("input", help="Path to input text file to memorize")
    train_parser.add_argument(
        "--epochs", type=int, default=2000, help="Number of training epochs"
    )
    train_parser.add_argument(
        "--hidden", type=int, default=512, help="Hidden size of LSTM"
    )
    train_parser.add_argument(
        "--layers", type=int, default=2, help="Number of LSTM layers"
    )
    train_parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for optimizer"
    )
    train_parser.add_argument(
        "--weights",
        default="model.safetensors",
        help="Path to save trained model weights",
    )

    # Generate subcommand
    gen_parser = subparsers.add_parser(
        "generate", help="Generate text from a trained model"
    )
    gen_parser.add_argument(
        "input", help="Path to original text file (for sequence length)"
    )
    gen_parser.add_argument(
        "--weights", default="model.safetensors", help="Path to trained model weights"
    )
    gen_parser.add_argument(
        "--output", default="generated.txt", help="Path to write generated text"
    )
    gen_parser.add_argument(
        "--hidden", type=int, default=512, help="Hidden size of LSTM"
    )
    gen_parser.add_argument(
        "--layers", type=int, default=2, help="Number of LSTM layers"
    )

    # Visualize subcommand
    vis_parser = subparsers.add_parser(
        "visualize", help="Generate demonstration charts"
    )
    vis_parser.add_argument(
        "input", help="Path to original text file used for training"
    )
    vis_parser.add_argument(
        "--weights",
        default="./output/model.safetensors",
        help="Path to trained model weights",
    )
    vis_parser.add_argument(
        "--hidden", type=int, default=32, help="Hidden size of LSTM"
    )
    vis_parser.add_argument(
        "--layers", type=int, default=2, help="Number of LSTM layers"
    )
    vis_parser.add_argument(
        "--output-dir", default="charts", help="Directory to save charts"
    )

    args = parser.parse_args()

    if args.command == "train":
        train_model(
            input_path=args.input,
            epochs=args.epochs,
            lr=args.lr,
            hidden=args.hidden,
            layers=args.layers,
            weights_out=args.weights,
        )
    elif args.command == "generate":
        generate_text(
            weights=args.weights,
            input_path=args.input,
            output_path=args.output,
            hidden=args.hidden,
            layers=args.layers,
        )
    elif args.command == "visualize":
        generate_demo_charts(
            weights_path=args.weights,
            input_path=args.input,
            hidden_size=args.hidden,
            num_layers=args.layers,
            output_dir=args.output_dir,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
