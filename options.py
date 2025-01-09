import argparse
import os
import datetime

def setup_args():
    parser = argparse.ArgumentParser()
    # Data and model configuration
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--train_batch_size", type=int, default=24)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=1024,
                        help="max length for input document")
    parser.add_argument("--knowledge_length", type=int, default=128,
                        help="max length for knowledge")
    parser.add_argument("--max_decode_step", type=int, default=128,
                        help="maximum decode step")
    parser.add_argument("--num_train_epochs", type=int, default=50,
                        help="Number of epochs to train")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--entity_embed_size", type=int, default=512)

    # Training configuration
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")

    # Path and retrieval configuration
    parser.add_argument("--num_paths", type=int, default=100)
    parser.add_argument("--penalty", type=float, default=1.5)
    
    # Output and logging
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./save/")
    parser.add_argument("--eval_frequency", type=int, default=1)
    
    args = parser.parse_args()

    # Set up output directory
    basedir = os.path.join(os.getcwd(), "save")  # Using relative path instead of hardcoded
    today = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(basedir, today)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Print important configurations
    print(f"Output directory: {args.output_dir}")
    print(f"Data directory: {args.data_dir}")
    print(f"Knowledge length: {args.knowledge_length}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Seed: {args.seed}")

    return args 