import json
import logging
import os
import pickle

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

from trainer import Trainer
from options import setup_args
from utils import Dialprocessor
from transformers import WEIGHTS_NAME, AutoTokenizer

from models.modeling import GraphConstraintLogitsProcessor, KnowledgeGenerator
from transformers import LogitsProcessorList

import re
from trie import Trie


logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
WEIGHTS_NAME = "pytorch_model.bin"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class DataModule:
    """Handles all data loading and processing operations"""
    
    def __init__(self, args):
        self.args = args
        self.processor = Dialprocessor(args, stage="retrieval")
        
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
        """Collate batch of examples into model inputs"""
        def create_padded_sequence(target, padding_value):
            """Create padded sequence from target"""
            if isinstance(target, str):
                tensors = [torch.tensor(getattr(o[1], target), dtype=torch.long) for o in batch]
            elif isinstance(target, tuple):
                tensors = target
            else:
                tensors = [torch.tensor(o, dtype=torch.long) for o in target]
            return pad_sequence(tensors, batch_first=True, padding_value=padding_value)
        
        dialog_history_ids, output_ids, episode_id, turn_id, entities = zip(*batch)
        filtered_dialog_history_ids = []
        filtered_output_ids = []
        filtered_entities = []
        filtered_episode_id = []
        filtered_turn_id = []
        for dhi, oi, ei, eid, tid in zip(dialog_history_ids, output_ids, entities, episode_id, turn_id):
            if oi is not None and (len(ei) > 0):
                if ei[0] == "":
                    continue
                filtered_dialog_history_ids.append(dhi)
                filtered_output_ids.append(oi)
                filtered_entities.append(ei)
                filtered_episode_id.append(eid)
                filtered_turn_id.append(tid)
                
        dialog_history_ids = create_padded_sequence(filtered_dialog_history_ids, 0)
        output_ids = create_padded_sequence(filtered_output_ids, 0)

        src_knowledge_ids = output_ids[:, :-1]
        trg_knowledge_ids = output_ids[:, 1:]
        trg_knowledge_ids = trg_knowledge_ids.masked_fill(trg_knowledge_ids == 0, 0)
        
        enc_mask = torch.sign(dialog_history_ids)
        dec_mask = torch.sign(src_knowledge_ids)
        dec_mask[:, 0] = 1
        
        return {
            "input_ids": dialog_history_ids,
            "attention_mask": enc_mask,
            "decoder_input_ids": src_knowledge_ids,
            "decoder_attention_mask": dec_mask,
            "labels": trg_knowledge_ids,
            "episode_id": filtered_episode_id,
            "turn_id": filtered_turn_id,
            "entities": filtered_entities,
        }
        

class KnowledgeGen:
    """Handles model evaluation"""
    
    def __init__(self, args, tokenizer, train_trie, valid_trie, test_trie):
        self.args = args
        self.tokenizer = tokenizer
        self.train_trie = train_trie
        self.valid_trie = valid_trie
        self.test_trie = test_trie
        
    def generate(self, model, dataloader, fold="dev"):
        def create_padded_sequence(target, padding_value):
            if isinstance(target, str):
                tensors = [torch.tensor(getattr(o[1], target), dtype=torch.long) for o in batch]
            elif isinstance(target, tuple):
                tensors = target
            else:
                tensors = [torch.tensor(o, dtype=torch.long) for o in target]
            return pad_sequence(tensors, batch_first=True, padding_value=padding_value)
        def extract_triplets(sentence):
            def generate_patterns(num_hops=1):
                basic_pattern_list = []
                basic_pattern = r'\[HEAD\]\s*(.*?)\[Int\d\_\d\]\[Int\d\_\d\]\s*(.*?)\[Int\d\_\d\]\[Int\d\_\d\]\s*(.*?)'
                basic_pattern_list.append(basic_pattern)
                basic_pattern = r'\[HEAD\]\s*(.*?)\[Rev\d\_\d\]\[Rev\d\_\d\]\s*(.*?)\[Rev\d\_\d\]\[Rev\d\_\d\]\s*(.*?)'
                basic_pattern_list.append(basic_pattern)
                for _ in range(num_hops-1):
                    new_basic_pattern_list = []
                    for bp in basic_pattern_list:
                        new_basic_pattern_list.append(bp + r'\[Int\d\_\d\]\[Int\d\_\d\]\s*(.*?)\[Int\d\_\d\]\[Int\d\_\d\]\s*(.*?)')
                        new_basic_pattern_list.append(bp + r'\[Rev\d\_\d\]\[Rev\d\_\d\]\s*(.*?)\[Rev\d\_\d\]\[Rev\d\_\d\]\s*(.*?)')
                    basic_pattern_list = new_basic_pattern_list

                basic_pattern_list = [bp + r'\[TAIL\]' for bp in basic_pattern_list]
                return basic_pattern_list
            
            pattern1 = generate_patterns(1)
            pattern2 = generate_patterns(2)
            
            for idx, p in enumerate(pattern2):
                hop2_triplet = re.findall(p, sentence)
                if len(hop2_triplet) != 0:
                    curr_triplet = hop2_triplet[0]
                    h, r1, e1, r2, e2 = curr_triplet
                    if idx == 0:
                        hop2_triplet = [(h,r1,e1), (e1,r2,e2)]
                    elif idx == 1:
                        hop2_triplet = [(h,r1,e1), (e2,r2,e1)]
                    elif idx == 2:
                        hop2_triplet = [(e1,r1,h), (e1,r2,e2)]
                    elif idx == 3:
                        hop2_triplet = [(e1,r1,h), (e2,r2,e1)]
                    return hop2_triplet
            
            for idx, p in enumerate(pattern1):
                hop1_triplet = re.findall(p, sentence)
                if len(hop1_triplet) != 0:
                    if idx == 1:
                        hop1_triplet = [(hop1_triplet[0][2], hop1_triplet[0][1], hop1_triplet[0][0])]
                    return hop1_triplet

            return []
            
        test_hyp, test_ref = [], []
        dataset_ptr = 0
        
        model.eval()
        
        for batch in tqdm(dataloader, desc="Eval"):
            gen_inputs = {k: v.to(self.args.device) for k, v in batch.items() \
                if k in ['input_ids','attention_mask']}
            gen_inputs["max_new_tokens"] = 128
            gen_inputs["num_beams"] = 5
            gen_inputs["early_stopping"] = True
            gen_inputs["use_cache"] = True
            gen_inputs["do_sample"] = False
            gen_inputs["top_p"] = 0.9
            gen_inputs["return_dict_in_generate"] = True
                
            
            if fold == "train":
                trie = self.train_trie
            elif fold == "dev":
                trie = self.valid_trie
            elif fold == "test":
                trie = self.test_trie
            else:
                raise ValueError(f"Invalid fold: {fold}")
            
            trie_list = []
            for ei, ti in zip(batch['episode_id'], batch['turn_id']):
                trie_list.append(trie[ei][ti])
                if trie[ei][ti] is None:
                    import pdb; pdb.set_trace()
                
            def load_const(batch_id, sent):
                if trie_list[batch_id] is None:
                    return None
                else:
                    return trie_list[batch_id].get(sent)
                
            logit_processor = LogitsProcessorList([GraphConstraintLogitsProcessor(prefix_allowed_tokens_fn=lambda batch_id, sent: load_const(batch_id ,sent.tolist()), num_beams=gen_inputs["num_beams"], args=self.args)])
            gen_inputs["logits_processor"] = logit_processor

            with torch.no_grad():
                if hasattr(model, "module"):
                    outputs = model.module.knowledge_generator.generate(**gen_inputs, output_scores=True, num_return_sequences=5)
                else:
                    outputs = model.knowledge_generator.generate(**gen_inputs, output_scores=True, num_return_sequences=5)
                
            seq = outputs.sequences
            seq_nlp = self.tokenizer.batch_decode(seq)
            seq_paths = [extract_triplets(sent) for sent in seq_nlp]
                
            path_list = []
            for sp in seq_paths:
                path_list.append(sp)
            path_list = [list(set(p)) for p in path_list]

        return

class ModelManager:
    """Handles model initialization and training"""
    
    def __init__(self, args):
        self.args = args
        self.best_dev_score = 0.0
        
    def initialize_model(self):
        """Initialize model and tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.args.tokenizer = tokenizer
        
        
        model = KnowledgeGenerator(self.args)
        unsup_checkpoint = torch.load(os.path.join(self.args.output_dir, "unsup/training_args.bin"), map_location="cpu")

        model.knowledge_generator.new_embed.weight.data = unsup_checkpoint['encoder.new_embed.weight']
        model.knowledge_generator.encoder.new_embed.weight.data = unsup_checkpoint['encoder.new_embed.weight']
        model.knowledge_generator.decoder.new_embed.weight.data = unsup_checkpoint['encoder.new_embed.weight']
        
        return model, tokenizer

def load_trie():
    with open("/hub_data1/jinyoungp/mhkp_public/data/trie_test.pkl", 'rb') as f:
        test_trie = pickle.load(f)
    with open("/hub_data1/jinyoungp/mhkp_public/data/trie_valid.pkl", 'rb') as f:
        valid_trie = pickle.load(f)
    with open("/hub_data1/jinyoungp/mhkp_public/data/trie_train.pkl", 'rb') as f:
        train_trie = pickle.load(f)
    
    return train_trie, valid_trie, test_trie

def run(args):
    """Main training loop"""
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # Initialize components
    data_module = DataModule(args)
    model_manager = ModelManager(args)
    
    model, tokenizer = model_manager.initialize_model()
    model.to(args.device)
    
    train_dataloader = data_module.load_examples("train")
    num_train_steps_per_epoch = len(train_dataloader)
    num_train_steps = int(num_train_steps_per_epoch * args.num_train_epochs)
    num_train_steps = 1
    
    # Train model
    trainer = Trainer(
        args,
        model=model,
        dataloader=train_dataloader,
        num_train_steps=num_train_steps,
    )
    trainer.train()
    
    # Final evaluation
    train_trie, valid_trie, test_trie = load_trie()
    knoweldge_generator = KnowledgeGen(args, tokenizer, train_trie, valid_trie, test_trie)

    model.to(args.device)
    train_dataloader = data_module.load_examples("train")
    knoweldge_generator.generate(model, train_dataloader, "train")
    
    valid_dataloader = data_module.load_examples("dev")
    knoweldge_generator.generate(model, valid_dataloader, "dev")
    
    test_dataloader = data_module.load_examples("test")
    knoweldge_generator.generate(model, test_dataloader, "test")

if __name__ == "__main__":
    args = setup_args()
    run(args)