import argparse
import yaml

from base_autoencoder import BaseAutoencoder

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Anomaly Detection Script")
    parser.add_argument('--config',     type=str, default="config.yaml",    help="Path to the configuration file")
    parser.add_argument('--product_class',      type=str, default="hazelnut",       help="class name or 'all")
    parser.add_argument('--model_name', type=str, default="base",           help="Name of the model to use")
    parser.add_argument('--train_path', type=str, default=None,             help="Path to the training output")
    parser.add_argument('--test_path',  type=str, default="/home/jaspinder/Github/Anomaly_Detection",             help="Path to the testing results")
    parser.add_argument('--mode',       type=str, default="train",          help="'train' or 'test'")
    
    return parser.parse_args()


def main():
    """Main function for anomaly detection."""
    args = parse_arguments()
    if args.mode == "train" and not args.train_path:
        raise ValueError("Training mode requires --train_path to be specified.")
    elif args.mode == "test" and not args.test_path and not args.train_path:
        raise ValueError("Testing mode requires --test_path and --train_path to be specified.")
    elif args.mode not in ["train", "test"]:
        raise ValueError("Invalid mode. Please specify 'train' or 'test'.")

    if args.model_name == "base":
        model = BaseAutoencoder(args.product_class, args.config, args.train_path, args.test_path)
        if args.mode == "train":
            model.train()
            model.save_model(args)
        else:
            model.test()

    ### TODO can add more models here with elif
    else:
        raise ValueError(f"Model name '{args.model_name}' not defined.")
if __name__ == "__main__":
    main()