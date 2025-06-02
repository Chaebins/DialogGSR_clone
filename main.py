import json
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import WEIGHTS_NAME, AutoTokenizer

from torch.nn.utils.rnn import pad_sequence
from trainer import Trainer
from options import setup_args
from utils.utils import (
    Dialprocessor,
    load_raw_dataset,
    Profiler
)
from utils.metrics import sequence_loss, bleu_metric, f1_metric
from rouge import Rouge
from models.modeling import (
    T5ForKnowledgeAugmentedGeneration
)

# Constants
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
WEIGHTS_NAME = "pytorch_model.bin"

class DataModule:
    """Handles all data loading and processing operations"""
    
    def __init__(self, args):
        self.args = args
        self.processor = Dialprocessor(args)
        
    def load_examples(self, fold):
        """Load and process examples for given fold"""
        if fold == "train":
            features = self.processor.get_train_examples(self.args.data_dir)
        elif fold == "dev":
            features = self.processor.get_dev_examples(self.args.data_dir)
        else:
            features = self.processor.get_test_examples(self.args.data_dir)
            
        dataloader = self._create_dataloader(features, fold)
        return dataloader

    def _create_dataloader(self, features, fold):
        """Create appropriate dataloader based on fold"""
        if fold == "train":
            sampler = RandomSampler(features)
            batch_size = self.args.train_batch_size
        else:
            sampler = None
            batch_size = self.args.eval_batch_size
            
        return DataLoader(
            features,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=self._collate_fn,
            num_workers=4
        )

    def _collate_fn(self, batch):
        def create_padded_sequence(target, padding_value):
            """Create padded sequence from target"""
            if isinstance(target, str):
                tensors = [torch.tensor(getattr(o[1], target), dtype=torch.long) for o in batch]
            elif isinstance(target, tuple):
                tensors = target
            else:
                tensors = [torch.tensor(o, dtype=torch.long) for o in target]
            return pad_sequence(tensors, batch_first=True, padding_value=padding_value)

        """Collate batch of examples into model inputs"""
        user_ids, response_ids = zip(*batch)
        user_ids = create_padded_sequence(user_ids, 0)
        response_ids = create_padded_sequence(response_ids, 0)
        
        src_response_ids = response_ids[:, :-1]
        trg_response_ids = response_ids[:, 1:]
        trg_response_ids = trg_response_ids.masked_fill(trg_response_ids == 0, 0)
        
        enc_mask = torch.sign(user_ids)
        dec_mask = torch.sign(src_response_ids)
        dec_mask[:, 0] = 1

        return {
            "input_ids": user_ids,
            "attention_mask": enc_mask,
            "decoder_input_ids": src_response_ids,
            "decoder_attention_mask": dec_mask,
            "labels": trg_response_ids,
        }


class Evaluator:
    """Handles model evaluation"""
    
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.rouge = Rouge()
        self.profiler = Profiler(args)
        
    def _compute_metrics(self, test_hyp, test_ref):
        """Compute evaluation metrics"""
        # Clean empty responses
        for i in range(len(test_hyp)):
            if len(test_hyp[i].replace(".", "").strip()) == 0:
                test_hyp[i] = "dialogue:"
        
        # Calculate metrics
        f1 = f1_metric(test_hyp, test_ref)
        b1, b2, b3, b4 = bleu_metric(test_hyp, test_ref)
        rouge_score = self.rouge.get_scores(hyps=test_hyp, refs=test_ref, avg=True)
        
        # Combine results
        results = {
            'bleu-1': b1, 
            'bleu-2': b2, 
            'bleu-3': b3, 
            'bleu-4': b4,
            'f1': f1,
        }
        results.update(rouge_score)
        
        return results
    
    def evaluate(self, model, dataloader, fold="dev", global_step=-1):
        """Evaluate model on given dataloader"""
        dataset = load_raw_dataset(self.args, fold)
        
        # Setup output files
        pred_file, ref_file, profile_file = self._setup_output_files(fold, global_step)
        
        pred_fw = open(pred_file, "w")
        ref_fw = open(ref_file, "w")
        profile_fw = open(profile_file, "w")

        # Initialize metrics
        test_hyp, test_ref = [], []
        dataset_ptr = 0
        profiler = Profiler(self.args)
        
        model.eval()
        
        for batch in tqdm(dataloader, desc="Eval"):
            gen_inputs = {k: v.to(self.args.device) for k, v in batch.items() \
                if k in ['input_ids','attention_mask']}
            recon_inputs = {k: v.to(self.args.device) for k, v in batch.items() \
                if k in ['input_ids','attention_mask','decoder_input_ids','decoder_attention_mask']}
            labels = batch['labels'].to(self.args.device)
            # import pdb; pdb.set_trace()
            gen_inputs["max_length"] = 128
            gen_inputs["num_beams"] = 5
            gen_inputs["length_penalty"] = self.args.penalty
            gen_inputs["repetition_penalty"] = 1
            gen_inputs["early_stopping"] = True
            gen_inputs["use_cache"] = True
            gen_inputs["do_sample"] = False
            gen_inputs["top_p"] = 0.95
            gen_inputs["top_k"] = 50
            gen_inputs["return_dict_in_generate"] = True
                
            input_ids = gen_inputs["input_ids"]
            batch_size = gen_inputs["input_ids"].size(0)

            with torch.no_grad():
                if hasattr(model, "module"):
                    outputs = model.module.response_generator.generate(**gen_inputs)
                else:
                    outputs = model.response_generator.generate(**gen_inputs)
                
                for i in range(batch_size):
                    pred_response = outputs.sequences[i].cpu()
                    pred_response_token = self.tokenizer.decode(pred_response,
                                                skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)

                    # Avoid -50
                    labels[i][labels[i] == 0] = 0
                    label_token = self.tokenizer.decode(labels[i].cpu(),
                                                skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
                    test_hyp.append(pred_response_token)
                    test_ref.append(label_token)
                        
                    pred_fw.write(pred_response_token.strip() + "\n")
                    pred_fw.flush()
                    ref_fw.write(label_token.strip() + "\n")
                    ref_fw.flush()
                    profiler.write_profile(profile_fw, 
                                        dataset[dataset_ptr], 
                                        input_ids[i], 
                                        pred_response_token,
                                        None,
                                        i # number of batch
                                        )
                    dataset_ptr += 1
                # break
        pred_fw.close()
        ref_fw.close()
        profile_fw.close()
                
        return self._compute_metrics(test_hyp, test_ref)


    def _setup_output_files(self, fold, global_step):
        """Setup output files for evaluation"""
        os.makedirs(os.path.join(self.args.output_dir, "candidates"), exist_ok=True)
        os.makedirs(os.path.join(self.args.output_dir, "profiles"), exist_ok=True)

        if global_step > 0:
            pred_file = os.path.join(self.args.output_dir, "candidates", 
                                   f"{fold}_candidate_step{global_step}.txt")
            profile_file = os.path.join(self.args.output_dir, "profiles",
                                      f"{fold}_profile_step{global_step}.txt")
        else:
            pred_file = os.path.join(self.args.output_dir, f"{fold}_candidate.txt")
            profile_file = os.path.join(self.args.output_dir, "profiles",
                                      f"{fold}_profile.txt")

        ref_file = os.path.join(self.args.output_dir, f"{fold}_reference.txt")
        
        return pred_file, ref_file, profile_file

class ModelManager:
    """Handles model initialization and training"""
    
    def __init__(self, args):
        self.args = args
        self.best_dev_score = 0.0
        
    def initialize_model(self, entity_embeddings=None):
        """Initialize model and tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.args.tokenizer = tokenizer

        model = T5ForKnowledgeAugmentedGeneration(self.args)
        # self._load_pretrained_weights(model)
        
        return model, tokenizer

def load_entity_embeddings_memory(args):
    """ Below are used if we use the pre-computed entity embeddings """
    memory_path = os.path.join(args.data_dir, "entity_codebook.pkl")
    label_path = os.path.join(args.data_dir, "relation_codebook.pkl")

    with open(memory_path, 'rb') as f:
        entity_embeddings_memory = pickle.load(f)

    with open(label_path, 'rb') as f:
        label_memory = pickle.load(f)
    label_map = dict()
    for idx, (key, value) in enumerate(label_memory.items()):
        label_map[value] = idx
    args.label_map = label_map

    wikidata_to_memory_map = dict()
    for idx, (key, value) in enumerate(entity_embeddings_memory.items()):
        wikidata_to_memory_map[value] = idx + 1

    args.wikidata_to_memory_map = wikidata_to_memory_map
    entity_embeddings = torch.zeros(len(wikidata_to_memory_map) + 1, args.entity_embed_size)
    args.initialize_embedding = True
    print(f"The number of entities: {entity_embeddings.shape[0]}")

    return entity_embeddings

def run(args):
    """Main training loop"""
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # Initialize components
    data_module = DataModule(args)
    model_manager = ModelManager(args)
    
    entity_embeddings = load_entity_embeddings_memory(args)
    model, tokenizer = model_manager.initialize_model(entity_embeddings)
    model.to(args.device)
    
    evaluator = Evaluator(args, tokenizer)
    
    # Load data
    train_dataloader = data_module.load_examples("train")
    num_train_steps_per_epoch = len(train_dataloader)
    num_train_steps = int(num_train_steps_per_epoch * args.num_train_epochs)
    
    def step_callback(model, global_step):
        if global_step % (num_train_steps_per_epoch * args.eval_frequency) == 0 and getattr(args,"local_rank",-1) in [0, -1] and model_manager.flag:
            epoch = int(global_step / num_train_steps_per_epoch - 1)
            dev_dataloader = data_module.load_examples("dev")
            dev_results = evaluator.evaluate(model, dev_dataloader, "dev", global_step)
            
            tqdm.write("dev: " + str(dev_results))
            
            # Save best model
            if dev_results["bleu-1"] > model_manager.best_dev_score:
                model_manager.best_dev_score = dev_results["bleu-1"]
                best_weights = {k: v.to("cpu").clone() for k, v in model.state_dict().items()}
                torch.save(best_weights, os.path.join(args.output_dir, WEIGHTS_NAME))
                
            model.train()
    
    # Train model
    trainer = Trainer(
        args,
        model=model,
        dataloader=train_dataloader,
        num_train_steps=num_train_steps,
        step_callback=step_callback,
    )
    trainer.train()
    
    # Final evaluation
    model, tokenizer = model_manager.initialize_model(entity_embeddings)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, WEIGHTS_NAME), map_location="cpu"))
    model.to(args.device)
    
    test_dataloader = data_module.load_examples("test")
    results = evaluator.evaluate(model, test_dataloader, "test")
        
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f)
        
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    print(results)
        
    return results

if __name__ == "__main__":
    args = setup_args()
    run(args)