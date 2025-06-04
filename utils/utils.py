import json
import linecache
import os
import subprocess
import pickle

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, BatchEncoding

from time import time

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def construct_paths(remaining_triplets, entities, curr_path_nlp_list=[], num_hops=2):
    if len(remaining_triplets) == 0:
        if len(curr_path_nlp_list) == 0:
            return ""
        if "[TAIL]" not in curr_path_nlp_list[-1]:
            curr_path_nlp_list.append("[TAIL]")
        return "".join(curr_path_nlp_list)

    if len(curr_path_nlp_list) == 0:
        # Start new path
        if remaining_triplets[0][0] not in entities:
            return construct_paths(remaining_triplets[1:], entities, ["[HEAD]"+remaining_triplets[0][2]+"[Rev1_1][Rev1_2]"+remaining_triplets[0][1].replace("_", " ").replace("-", " ")+"[Rev2_1][Rev2_2]"+remaining_triplets[0][0]], num_hops)
        else:
            return construct_paths(remaining_triplets[1:], entities, ["[HEAD]"+remaining_triplets[0][0]+"[Int1_1][Int1_2]"+remaining_triplets[0][1].replace("_", " ").replace("-", " ")+"[Int2_1][Int2_2]"+remaining_triplets[0][2]], num_hops)
    
    last_segment = curr_path_nlp_list[-1]
    if "[Int" + str(num_hops*2) + "_1]" in last_segment or "[Rev" + str(num_hops*2) + "_1]" in last_segment:
        return construct_paths(remaining_triplets[1:], entities, curr_path_nlp_list + ["[HEAD]"+remaining_triplets[0][0]+"[Int1_1][Int1_2]"+remaining_triplets[0][1].replace("_", " ").replace("-", " ")+"[Int2_1][Int2_2]"+remaining_triplets[0][2]], num_hops)
    else:
        for nh in range(num_hops, 0, -1):
            if "[Int"+str(nh*2)+"_2]" in last_segment:
                curr_entity = last_segment.split("[Int"+str(nh*2)+"_2]")[1].strip()
                break
            elif "[Rev"+str(nh*2)+"_2]" in last_segment:
                curr_entity = last_segment.split("[Rev"+str(nh*2)+"_2]")[1].strip()
                break
        
        if remaining_triplets[0][0] == curr_entity:
            curr_hop = nh
            if curr_hop == num_hops:
                return construct_paths(remaining_triplets[1:], entities, curr_path_nlp_list + ["[Int"+str(curr_hop*2+1)+"_1][Int"+str(curr_hop*2+1)+"_2]"+remaining_triplets[0][1].replace("_", " ").replace("-", " ")+"[Int"+str(curr_hop*2+2)+"_1][Int"+str(curr_hop*2+2)+"_2]"+remaining_triplets[0][2]+"[TAIL]"], num_hops)
            else:
                return construct_paths(remaining_triplets[1:], entities, curr_path_nlp_list + ["[Int"+str(curr_hop*2+1)+"_1][Int"+str(curr_hop*2+1)+"_2]"+remaining_triplets[0][1].replace("_", " ").replace("-", " ")+"[Int"+str(curr_hop*2+2)+"_1][Int"+str(curr_hop*2+2)+"_2]"+remaining_triplets[0][2]], num_hops)
        elif remaining_triplets[0][2] == curr_entity:
            curr_hop = nh
            if curr_hop == num_hops:
                return construct_paths(remaining_triplets[1:], entities, curr_path_nlp_list + ["[Rev"+str(curr_hop*2+1)+"_1][Rev"+str(curr_hop*2+1)+"_2]"+remaining_triplets[0][1].replace("_", " ").replace("-", " ")+"[Rev"+str(curr_hop*2+2)+"_1][Rev"+str(curr_hop*2+2)+"_2]"+remaining_triplets[0][0]+"[TAIL]"], num_hops)
            else:
                return construct_paths(remaining_triplets[1:], entities, curr_path_nlp_list + ["[Rev"+str(curr_hop*2+1)+"_1][Rev"+str(curr_hop*2+1)+"_2]"+remaining_triplets[0][1].replace("_", " ").replace("-", " ")+"[Rev"+str(curr_hop*2+2)+"_1][Rev"+str(curr_hop*2+2)+"_2]"+remaining_triplets[0][0]], num_hops)
        else:
            if remaining_triplets[0][0] not in entities:
                return construct_paths(remaining_triplets[1:], entities, curr_path_nlp_list + ["[TAIL]"]+["[HEAD]"+remaining_triplets[0][2]+"[Rev1_1][Rev1_2]"+remaining_triplets[0][1].replace("_", " ").replace("-", " ")+"[Rev2_1][Rev2_2]"+remaining_triplets[0][0]], num_hops)
            else:
                return construct_paths(remaining_triplets[1:], entities, curr_path_nlp_list + ["[TAIL]"]+["[HEAD]"+remaining_triplets[0][0]+"[Int1_1][Int1_2]"+remaining_triplets[0][1].replace("_", " ").replace("-", " ")+"[Int2_1][Int2_2]"+remaining_triplets[0][2]], num_hops)
class T5Dataset(Dataset):
    def __init__(self, jsonl_file, args, stage="response"):
        self.args = args
        self.is_train = 'train' in jsonl_file

        self.max_length = args.max_length
        self.max_decode_step = args.max_decode_step
        self.tokenizer = args.tokenizer
        self.hist_turn = 100
        self.file_name = jsonl_file
        self.total_size = int(subprocess.check_output(
            "wc -l " + jsonl_file, shell=True).split()[0])

        special_tokens = ['[HEAD]']
        for i in range(1, 3):
            special_tokens.extend([
                f'[Int{i*2-1}_1]', f'[Int{i*2-1}_2]',
                f'[Rev{i*2-1}_1]', f'[Rev{i*2-1}_2]',
                f'[Int{i*2}_1]', f'[Int{i*2}_2]',
                f'[Rev{i*2}_1]', f'[Rev{i*2}_2]'
            ])
        special_tokens.append('[TAIL]')
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.path_lim = self.args.num_paths
        self.stage = stage
            
        self.apprentice_prefix = "apprentice: "
        self.wizard_prefix = "wizard: "
        self.knowledge_prefix = "knowledge: "
        self.prefix = "dialogue: "
        self.topic_prefix = "topic: "

        with open(os.path.join(args.data_dir, "entity_codebook.pkl"), 'rb') as f:
            self.entity_codebook = pickle.load(f)
        self.reverse_entity_codebook = {v:k for k, v in self.entity_codebook.items()}

        with open(os.path.join(args.data_dir, "relation_codebook.pkl"), 'rb') as f:
            self.relation_codebook = pickle.load(f)
        self.reverse_relation_codebook = {v:k for k, v in self.relation_codebook.items()}

    def check_reverse(self, path_element):
        if "reverse_" in path_element:
            return True
        else:
            return False

    def _get_added_tokens(self, reverse_flag, position):
        """Get the appropriate added tokens based on path element and position."""
        if reverse_flag:
            return f"[Rev{position}_1][Rev{position}_2]"
        else:
            return f"[Int{position}_1][Int{position}_2]"

    def _process_path_element(self, sent, label_sent, element, is_masked, idx, added_tokens="", is_tail=False):
        """Process a single path element and return updated sent, label_sent, and idx."""
        if is_masked:
            sent = sent + f'<extra_id_{idx}>' + ("[TAIL]" if is_tail else added_tokens)
            clean_element = element.replace("reverse_", "") if not is_tail else element
            label_sent = label_sent + clean_element + f'<extra_id_{idx+1}>'
            idx += 1
        else:
            clean_element = element.replace("reverse_", "") if not is_tail else element
            sent = sent + clean_element + ("[TAIL]" if is_tail else added_tokens)
            
        return sent, label_sent, idx

    def _process_path(self, path, masked_indices):
        """Process a single path with masked indices."""
        sent = "[HEAD]"
        label_sent = '<extra_id_0>'
        idx = 0
        
        for i in range(2*2 + 1):
            if i % 2 == 0 and i != 0:
                if len(path)-1 == i:
                    sent, label_sent, idx = self._process_path_element(sent, label_sent, path[i], masked_indices[i], idx, is_tail=True)
                    break
                else:
                    reverse_flag = self.check_reverse(path[(i//2)*2 +1])
                    added_tokens = self._get_added_tokens(reverse_flag, i+1)
                    sent, label_sent, idx = self._process_path_element(sent, label_sent, path[i], masked_indices[i], idx, added_tokens)
            else:
                reverse_flag = self.check_reverse(path[(i//2)*2 +1])
                added_tokens = self._get_added_tokens(reverse_flag, i+1)
                sent, label_sent, idx = self._process_path_element(sent, label_sent, path[i], masked_indices[i], idx, added_tokens)

            idx = 0

        return sent, label_sent

    def unsupervised(self, index):
        """Generate masked path data for unsupervised learning."""
        line = linecache.getline(self.file_name, index + 1)
        json_dict = json.loads(line)
        
        bos_id = torch.tensor([self.tokenizer.pad_token_id], dtype=torch.long)

        paths = json_dict["paths"]
        paths_inputs, paths_inputs_label = list(), list()
        
        # Generate masking matrix
        prob_matrix = torch.full((len(paths), 5), self.args.masking_ratio)
        masked_indices = torch.bernoulli(prob_matrix).bool()

        # Process each path
        for path, masks in zip(paths, masked_indices):
            sent, label_sent = self._process_path(path, masks)
            paths_inputs.append(sent)
            paths_inputs_label.append(label_sent)
            
        if len(paths_inputs) > 0:
            paths_inputs = paths_inputs[-self.path_lim:]
            paths_inputs_label = paths_inputs_label[-self.path_lim:]
            paths_ids = self.tokenizer.batch_encode_plus(paths_inputs, return_tensors="pt", padding=True).input_ids
            paths_ids_label = self.tokenizer.batch_encode_plus(paths_inputs_label, return_tensors="pt", padding=True).input_ids
        else:
            paths_ids, paths_ids_label = None, None
        
        return_data = (paths_ids, paths_ids_label)
        return return_data

    def knowledge_retrieval(self, index):
        line = linecache.getline(self.file_name, index + 1)
        json_dict = json.loads(line)
        
        episode_id = json_dict["episode_id"]
        turn_id = json_dict["turn_id"]
        entities = json_dict["entities"]
        
        bos_id = torch.tensor([self.tokenizer.pad_token_id], dtype=torch.long)
        eos_id = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
        
        dialog_history = json_dict["history"]
        prefixed_dialog_history = self.prefix + '\n '.join(dialog_history[-self.hist_turn:])
        
        gold_triplets = json_dict["gold_triplets"]
        if len(gold_triplets) >= 1:
            gold_knowledge = construct_paths(gold_triplets, json_dict['entities'])
            output_ids = self.tokenizer.encode(
                gold_knowledge,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            ).squeeze(0)
            output_ids = torch.cat([bos_id, output_ids], dim=0)
        else:
            output_ids = None
            
        assert len(prefixed_dialog_history) > 0
        dialog_history_ids = self.tokenizer.encode(
            prefixed_dialog_history,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length).squeeze(0)
        
        return_data = (dialog_history_ids, output_ids, episode_id, turn_id, entities)
        return return_data


    
    def with_inference(self, index):
        line = linecache.getline(self.file_name, index + 1)
        json_dict = json.loads(line)
        
        bos_id = torch.tensor([self.tokenizer.pad_token_id], dtype=torch.long)
        eos_id = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)

        dialog_history = json_dict["history"]
        prefixed_dialog_history = self.prefix + '\n '.join(dialog_history[-self.hist_turn:])

        rel_paths = json_dict["gold_triplets"]
        rel_knowledge = self.knowledge_prefix

        for idx, rel_triplets in enumerate(reversed(rel_paths)):
            if idx < 2:
                continue
            curr_rel_paths = construct_paths(rel_triplets, json_dict['entities'])
            
            if len(self.tokenizer.encode(rel_knowledge+curr_rel_paths)) > self.args.knowledge_length:
                break
            else:
                rel_knowledge += curr_rel_paths


        prefixed_dialog_history =  rel_knowledge + "</s>" +  prefixed_dialog_history

        assert len(prefixed_dialog_history) > 0
        
        dialog_history_ids = self.tokenizer.encode(
            prefixed_dialog_history,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length).squeeze(0)

        response = json_dict["label"]
        assert len(response) > 0
        
        response_ids = self.tokenizer.encode(
            response,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_decode_step).squeeze(0)
        response_ids = torch.cat([bos_id, response_ids], dim=0)

        return_data = (dialog_history_ids, response_ids)
        return return_data

    def with_train(self, index):
        line = linecache.getline(self.file_name, index + 1)
        json_dict = json.loads(line)
        
        bos_id = torch.tensor([self.tokenizer.pad_token_id], dtype=torch.long)
        eos_id = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)

        # prefixed_dialog_history = []
        dialog_history = json_dict["history"]
        # prefixed_dialog_history = self.prefix + ' '.join(dialog_history)
        prefixed_dialog_history = self.prefix + '\n '.join(dialog_history[-self.hist_turn:])
        # checked_knowledge = self.knowledge_prefix + json_dict['checked_knowledge']

        tot_knowledge = self.knowledge_prefix

        rel_paths = json_dict["gold_triplets"]

        for idx, rel_triplets in enumerate(reversed(rel_paths)):
            if idx < 2:
                continue
            curr_rel_paths = construct_paths(rel_triplets, json_dict['entities'])
            
            if len(self.tokenizer.encode(tot_knowledge+curr_rel_paths)) > self.args.knowledge_length:
                break
            else:
                tot_knowledge += curr_rel_paths


        prefixed_dialog_history =  tot_knowledge + "</s>" +  prefixed_dialog_history

        assert len(prefixed_dialog_history) > 0
        dialog_history_ids = self.tokenizer.encode(
            prefixed_dialog_history,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length).squeeze(0)

        response = json_dict["label"]
        assert len(response) > 0
        
        # Tokenize response
        response_ids = self.tokenizer.encode(
            response,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_decode_step).squeeze(0)
        response_ids = torch.cat([bos_id, response_ids], dim=0)
        

        return_data = (dialog_history_ids, response_ids)
        return return_data

    def __getitem__(self, index):
        if self.stage == "retrieval":
            return self.knowledge_retrieval(index)
        elif self.stage == "unsupervised":
            return self.unsupervised(index)
        elif self.stage == "response":
            if self.is_train:
                return self.with_train(index)
            else:
                return self.with_inference(index)
        else:
            raise ValueError(f"Invalid stage: {self.stage}")

    def __len__(self):
        return self.total_size

class Dialprocessor(object):
    def __init__(self, args, stage="response"):
        self.train_file = "train.jsonl"
        self.dev_file = "valid.jsonl"
        self.test_file = "test.jsonl"
        self.unsupervised_file = "unsup_path.jsonl"
        self.args = args
        self.stage = stage
        args.dev_file = self.dev_file
        args.test_file = self.test_file

    def get_train_examples(self, data_dir):
        print(f"DataProcessor: {self.train_file}")
        return T5Dataset(os.path.join(data_dir, self.train_file), args=self.args, stage=self.stage)

    def get_dev_examples(self, data_dir):
        print(f"DataProcessor: {self.dev_file}")
        return T5Dataset(os.path.join(data_dir, self.dev_file), args=self.args, stage=self.stage)

    def get_test_examples(self, data_dir):
        print(f"DataProcessor: {self.test_file}")
        return T5Dataset(os.path.join(data_dir, self.test_file), args=self.args, stage=self.stage)

    def get_unsupervised_examples(self, data_dir):
        print(f"DataProcessor: {self.unsupervised_file}")
        return T5Dataset(os.path.join(data_dir, self.unsupervised_file), args=self.args, stage=self.stage)

def load_raw_dataset(args, fold):
    if fold == "train":
        filename = "train.jsonl"
    elif fold == "dev":
        filename = "valid.jsonl"
    else:
        filename = "test.jsonl"

    datafile = os.path.join(args.data_dir, filename)
    with open(datafile, 'r') as f:
        dataset = [json.loads(data) for data in f.readlines()]
    return dataset

class Profiler(object):
    def __init__(self, args):
        with open(os.path.join(args.data_dir, "entity_codebook.pkl"), 'rb') as f:
            self.entity_codebook = pickle.load(f)
        self.reverse_entity_codebook = {v:k for k, v in self.entity_codebook.items()}

        with open(os.path.join(args.data_dir, "relation_codebook.pkl"), 'rb') as f:
            self.relation_codebook = pickle.load(f)
        self.reverse_relation_codebook = {v:k for k, v in self.relation_codebook.items()}
        self.tokenizer = args.tokenizer
        self.reverse_label_map = {v:k for k, v in args.label_map.items()}

    def write_profile(self,
                      profile_fw,
                      data,
                      new_input_ids,
                      pred_response_token,
                      path_ids,
                      batch_idx):
        headline = f"Episode {data['episode_id']}, Turn {data['turn_id']}"
        history = "HISTORY ==================\n" + '\n'.join(data['history'])
        response = "GT RESPONSE ================\n" + data["label"]
        preds = "PREDICTIONS =================\n" + pred_response_token.strip()
        # entities = data["history_entities"]
        # knowledges = "ENTITIES ====================\n" + ', '.join(entities)
        # knowledges += f"# Knowledge entities: {len(entities)}\n"
        gold_knowledges = ""
        for gt in data["gold_triplets"]:
            gold_knowledges += " ".join(gt)
            gold_knowledges += "\n"
            
        gold_knowledges = "GOLD_knowledges ====================\n" + gold_knowledges
        new_history = self.tokenizer.decode(new_input_ids.cpu(),
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
        new_history = ("Selected FACT + HISTORY ============\n" + new_history).strip()

        profile_fw.write(headline + '\n')
        profile_fw.write(history + '\n')
        profile_fw.write(response + '\n')
        profile_fw.write(new_history + '\n')
        profile_fw.write(gold_knowledges)
        profile_fw.write(preds + '\n\n\n')

            
        profile_fw.flush()
            

