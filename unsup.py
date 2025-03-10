import logging
import os

import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import WEIGHTS_NAME, AutoTokenizer, BertConfig

from trainer import Trainer
from options import setup_args
from utils import Dialprocessor

# Configure logging
logger = logging.getLogger(__name__)

# Global constants
WEIGHTS_NAME = "pytorch_model.bin"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class DataModule:
    """Handles all data loading and processing operations"""
    
    def __init__(self, args):
        self.args = args
        self.processor = Dialprocessor(args, stage="unsupervised")
        
    def load_examples(self):
        """Load and process examples for given fold"""
        features = self.processor.get_unsupervised_examples(self.args.data_dir)
        dataloader = self._create_dataloader(features)
        return dataloader
    
    def _create_dataloader(self, features):
        """Create appropriate dataloader based on fold"""
        sampler = RandomSampler(features)
        batch_size = self.args.train_batch_size
            
        return DataLoader(
            features,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=self._collate_fn,
            num_workers=4
        )
    
    def _collate_fn(self, batch):
        """Collate batch of examples into model inputs"""
        def create_padded_sequence(input, output):
            """Create padded sequence from target"""
            max_len =0
            for i, o in zip(input, output):
                max_len = max(max(i.size(1), o.size(1)), max_len)
            
            new_input = []
            new_output = []
            for i, o in zip(input, output):
                i = torch.cat([i, torch.zeros(i.size(0), max_len - i.size(1), dtype=torch.long)], dim=1)
                o = torch.cat([o, torch.zeros(o.size(0), max_len - o.size(1), dtype=torch.long)], dim=1)
                new_input.append(i)
                new_output.append(o)
            new_input = torch.cat(new_input, dim=0)
            new_output = torch.cat(new_output, dim=0)
            
            return new_input, new_output
        
        paths_inputs_ids, paths_outputs_ids = zip(*batch)
        paths_inputs_ids, paths_outputs_ids = create_padded_sequence(paths_inputs_ids, paths_outputs_ids)
        
        return {
            "input_ids": paths_inputs_ids,
            "labels": paths_outputs_ids,
        }

class ModelManager:
    """Handles model initialization and training"""
    
    def __init__(self, args):
        self.args = args
        self.best_dev_score = 0.0
        
    def initialize_model(self):
        """Initialize model and tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.args.tokenizer = tokenizer
        
        
        from models.modeling import KnowledgePretrainer
        model = KnowledgePretrainer(self.args)
        
        return model, tokenizer


def run(args):
    """Main training loop"""
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # Initialize model and data
    data_module = DataModule(args)
    model_manager = ModelManager(args)
    model, tokenizer = model_manager.initialize_model()
    model.to(args.device)

    train_dataloader = data_module.load_examples()
    # Calculate training steps
    num_train_steps_per_epoch = len(train_dataloader)
    num_train_steps = int(num_train_steps_per_epoch * args.num_train_epochs)

    # Save training args
    torch.save(args, os.path.join(args.output_dir, "unsup/training_args.bin"))
    
    best_weights = [None]

    def step_callback(model: torch.nn.Module, global_step: int) -> None:
        """Callback function called after each training step."""
        if global_step % (num_train_steps_per_epoch // args.eval_frequency) == 0:
            # Save model checkpoint
            if hasattr(model, "module"):
                best_weights[0] = {k: v.to("cpu").clone() for k, v in model.module.state_dict().items()}
            else:
                best_weights[0] = {k: v.to("cpu").clone() for k, v in model.state_dict().items()}
                
            logger.info("Saving model checkpoint to %s", args.output_dir)
            torch.save(best_weights[0], os.path.join(args.output_dir, WEIGHTS_NAME))                
        model.train()

    # Initialize and run trainer
    trainer = Trainer(
        args,
        model=model,
        dataloader=train_dataloader,
        num_train_steps=num_train_steps,
        step_callback=step_callback,
    )
    trainer.train()

    # Save final model
    logger.info("Saving final model checkpoint to %s", args.output_dir)
    torch.save(best_weights[0], os.path.join(args.output_dir, WEIGHTS_NAME))


if __name__ == "__main__":
    args = setup_args()
    run(args)