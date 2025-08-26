import argparse

from trafo_model.trafo_autoencoder import TransAEManager

from base_model.base_autoencoder import BaseAEManager
from vit_model.ViT import ViTManager


def parse_arguments():
    """
    Parse command-line arguments for the anomaly detection script.
    This function sets up an argument parser to handle various command-line options needed for running the script:
        --config (str): Path to the configuration file. Default is "config.yaml".
        --product_class (str): Name of the product class to process (or 'all' for all classes). Default is "hazelnut".
        --model_name (str): Identifier for the model to be used. Default is "base".
        --train_path (str): Filesystem path to the training outputs. Default is None.
        --test_path (str): Filesystem path to save or retrieve testing results. Default is "/home/jaspinder/Github/Anomaly_Detection".
        --mode (str): Operational mode of the script, either "train" for training or "test" for evaluation. Default is "train".
    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments as attributes.
    """
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Anomaly Detection Script")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--product_class", type=str, default="hazelnut", help="class name or 'all'"
    )
    parser.add_argument(
        "--model_name", type=str, default="vit", help="Name of the model to use"
    )
    parser.add_argument(
        "--train_path", 
        type=str,
        default="trains",
        help="Path to the training output"
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="tests",
        help="Path to the testing results",
    )
    parser.add_argument("--mode", type=str, default="train", help="'train' or 'test'")

    return parser.parse_args()


def main():
    """Main function for anomaly detection."""
    args = parse_arguments()
    if args.mode == "train" and not args.train_path:
        raise ValueError("Training mode requires --train_path to be specified.")
    elif args.mode == "test" and not args.test_path and not args.train_path:
        raise ValueError(
            "Testing mode requires --test_path and --train_path to be specified."
        )
    elif args.mode not in ["train", "test"]:
        raise ValueError("Invalid mode. Please specify 'train' or 'test'.")

    if args.model_name == "base":
        model = BaseAEManager(
            args.product_class, args.config, args.train_path, args.test_path
        )
        if args.mode == "train":
            model.train()
            mean_error, std_error, threshold = model.compute_thresh()
            model.save_model(args, mean_error, std_error, threshold)
        else:
            model.test()

    elif args.model_name == "trafo":
        model = TransAEManager(
            args.product_class, args.config, args.train_path, args.test_path
        )
        if args.mode == "train":
            model.train()
            mean_error, std_error, threshold = model.compute_thresh()
            model.save_model(args, mean_error, std_error, threshold)
        else:
            model.test()
    elif args.model_name == "vit":
        model = ViTManager(
            args.product_class, args.config, args.train_path, args.test_path
        )
        if args.mode == "train":
            model.train()
            # mean_error, std_error, threshold = model.compute_thresh()
            thres = model.thresholding()
            model.save_model(args, thres)
        else:
            model.test()

    ### TODO can add more models here with elif
    else:
        raise ValueError(f"Model name '{args.model_name}' not defined.")


if __name__ == "__main__":
    main()
