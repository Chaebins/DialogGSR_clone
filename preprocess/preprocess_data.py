import csv
import json
import random
import pickle
import os
import sys
import argparse
from copy import deepcopy
from typing import Dict, List, Set, Tuple, Any
import spacy
import en_core_web_md
from tqdm import tqdm
import networkx as nx

class Preprocessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.maximum_triplets = 100
        self.maximum_rels = 10

        self.nlp = en_core_web_md.load()
        self._load_codebooks()

        # Load and process databases
        self._load_triplets()
        self.graph = self._build_and_filter_database()

    def _load_codebooks(self):
        """Load entity and relation codebooks"""
        with open(f"{self.data_dir}/entity_codebook.pkl", 'rb') as f:
            self.entity_codebook = pickle.load(f)
        self.reverse_entity_codebook = {v:k for k, v in self.entity_codebook.items()}
        
        with open(f"{self.data_dir}/relation_codebook.pkl", 'rb') as f:
            self.relation_codebook = pickle.load(f)
        self.reverse_relation_codebook = {v:k for k, v in self.relation_codebook.items()}

    def _load_triplets(self):
        """Load all triplets from file"""
        with open(f"{self.data_dir}/opendialkg_triples.txt", 'r') as f:
            self.entire_triplets = f.readlines()
        print(f"# Entire Triples: {len(self.entire_triplets)}")

    def _generate_graph(self, database):
        G = nx.DiGraph()
        for key_id, triplet in database.items():
            for t in triplet:
                sub, rel, obj = t
                sub = self.reverse_entity_codebook[sub]
                obj = self.reverse_entity_codebook[obj]
                rel = self.reverse_relation_codebook[rel]
                if "~" in rel:
                    G.add_edge(obj, sub, relation=rel[1:].replace("_", " ").replace("-", " "))
                    G.add_edge(sub, obj, relation="~"+rel[1:].replace("_", " ").replace("-", " "))
                else:
                    G.add_edge(sub, obj, relation=rel.replace("_", " ").replace("-", " "))
                    G.add_edge(obj, sub, relation="~"+rel.replace("_", " ").replace("-", " "))
        return G
        
    def _build_and_filter_database(self) -> Dict:
        """Build and filter database for either head or tail entities"""
        database = self._build_database()
        filtered_database = self._filter_database(database)
        G = self._generate_graph(filtered_database)
        return G

    def _build_database(self) -> Dict:
        """Build initial database"""
        database = dict()
        for triplet in self.entire_triplets:
            _triplet = triplet.strip().split('\t')
            if len(_triplet) < 3:
                continue
                
            head, relation, tail = _triplet
            head_id = self.entity_codebook[head.lower()]
            relation_id = self.relation_codebook[relation.lower()]
            tail_id = self.entity_codebook[tail.lower()]

            _id = {
                "head": head_id,
                "tail": tail_id,
                "relation": relation_id
            }["head"]
            
            if _id not in database:
                database[_id] = set()
            database[_id].add((head_id, relation_id, tail_id))
            
        return database

    def _filter_database(self, database: Dict) -> Dict:
        """Filter database to limit number of relations"""
        new_database = {}
        for key, triplet in database.items():
            rel_dict = {}
            new_triplet = set()
            
            # Group by relation
            for t in triplet:
                _id = t[1]
                if _id not in rel_dict:
                    rel_dict[_id] = set()
                rel_dict[_id].add(t)

            # Sample if needed
            for tr_rel in rel_dict.values():
                if len(tr_rel) > self.maximum_rels:
                    new_triplet.update(random.sample(tr_rel, self.maximum_rels))
                else:
                    new_triplet.update(tr_rel)
                    
            new_database[key] = new_triplet

        return new_database


    def map_entity(self, entity: str) -> int:
        """Map entity string to code"""
        try:
            return self.entity_codebook[entity.lower()]
        except:
            return None

    def map_code(self, code: int) -> str:
        """Map code to entity string"""
        try:
            return self.reverse_entity_codebook[code]
        except:
            return None
        
    def find_entity(self, message, entities):
        found_entities = set()
        for entity in entities:
            if entity.lower() in message.lower():
                entity_flag = self.map_entity(entity)
                if entity_flag is not None:
                    found_entities.add(entity_flag)

        doc = self.nlp(message)
        for ent in doc.ents:
            entity_flag = self.map_entity(ent.text)
            if entity_flag is not None:
                found_entities.add(entity_flag)
        
        return found_entities
    
    
    def preprocess(self, dataset, fold):
        new_dataset = []
        episode_id = 0
        
        for rows, unique_id in tqdm(dataset, desc="Preprocessing..."):
            turn_id = 0
            history = []
            candidate_entities = set()
            history_entities = set()

            pp_rows = json.loads(rows[0])

            for row in pp_rows:
                if 'metadata' in row.keys():
                    if 'path' in row['metadata'].keys():
                        row_gold_triplets = row['metadata']['path'][1]
                        for triplet in row_gold_triplets:
                            candidate_entities.add(triplet[0])
                            candidate_entities.add(triplet[-1])

            gold_triplets = []
            has_gold = False
            for row in pp_rows:
                if 'metadata' in row.keys() and 'path' in row['metadata'].keys():
                    gold_triplets = row['metadata']['path'][1]
                    has_gold = True
                    
                elif 'message' in row.keys():
                    message = row['message']
                    
                    if row['sender'] == 'assistant':
                        # Create dialogue turn data
                        dialog_text = '\n'.join(history)
                        triplets_data = self._get_triplets_data(history_entities, gold_triplets, has_gold, fold)
                        data = {
                            "episode_id": episode_id,
                            "turn_id": turn_id,
                            "history": deepcopy(history),
                            "label": message,
                            "entities": [self.reverse_entity_codebook[entity] for entity in history_entities],
                            "unique_id": unique_id,
                            **triplets_data
                        }
                        new_dataset.append(data)
                        turn_id += 1
                    # Update history
                    found_entities = self.find_entity(message, list(candidate_entities))
                    history_entities.update(found_entities)
                    history.append(message)
            
            episode_id += 1
        
        print(f"{fold} Size: {len(new_dataset)}")
        self._save_dataset(new_dataset, fold)

    def _get_triplets_data(self, history_entities, gold_triplets, has_gold, fold):
        triplets = self._preprocess_triplets(history_entities)
        
        if has_gold:
            gold_triplets = self._map_triplet(gold_triplets)
            has_gold = False
        else:
            gold_triplets = []

        if fold == "train":
            for _triplet in gold_triplets:
                if _triplet not in triplets:
                    triplets.append(_triplet)
        
        return {"triplets": triplets, "gold_triplets": gold_triplets}

    def _preprocess_triplets(self, history_entities):
        entire_entities = list(history_entities)
        _triplets = set()
        for entity_id in entire_entities:
            entity = self.reverse_entity_codebook[entity_id]
            if entity not in self.graph.nodes():
                print(entity)
                continue
            one_hop_neighbors = list(self.graph.successors(entity))
                
            one_hop_facts = []
            for neighbor in one_hop_neighbors:
                rel = self.graph[entity][neighbor]['relation']
                one_hop_facts.append((entity, rel, neighbor))
            if len(one_hop_facts) > self.maximum_triplets:
                one_hop_facts = random.sample(one_hop_facts, self.maximum_triplets)
            _triplets.update(one_hop_facts)

                    
        new_triplets = []
        for triplet in _triplets:
            if triplet[1][0] == '~':
                new_triplets.append((triplet[2], triplet[1][1:], triplet[0]))
            else:
                new_triplets.append((triplet[0], triplet[1], triplet[2]))

        return new_triplets
    
    def _map_triplet(self, triplets):
        return_list = []
        for triplet in triplets:
            if triplet[1][0] != '~':
                return_list.append((triplet[0].lower(), triplet[1].lower().replace("_", " ").replace("-", " "), triplet[2].lower()))
            else:
                return_list.append((triplet[2].lower(), triplet[1][1:].lower().replace("_", " ").replace("-", " "), triplet[0].lower()))

        return return_list

    def _save_dataset(self, dataset: List[Dict], fold: str):
        """Save processed dataset to file"""
        filename = f"{fold}.jsonl"
        with open(os.path.join(self.data_dir, filename), 'w') as f:
            for data in dataset:
                f.write(json.dumps(data) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/data')
    args = parser.parse_args()
    
    dataset_dict = {}
    dataset_idx = []
    data_dir = args.data_dir
    with open(f"{data_dir}/opendialkg.csv") as csvfile:
        reader = csv.reader(csvfile)

        unique_id = 1000000
        for i, rows in enumerate(reader):
            if i == 0: continue
            dataset_dict[unique_id] = rows
            dataset_idx.append(unique_id)        
            unique_id += 1

    random.seed(42)
    random.shuffle(dataset_idx)

    train_ratio = 0.7
    valid_ratio = 0.15

    train_idx = dataset_idx[:int(len(dataset_idx) * train_ratio)]
    valid_idx = dataset_idx[int(len(dataset_idx) * train_ratio):int(len(dataset_idx) * (train_ratio + valid_ratio))]
    test_idx = dataset_idx[int(len(dataset_idx) * (train_ratio + valid_ratio)):]
    
    train_dataset, valid_dataset, test_dataset = [], [], []
    for unique_id, rows in dataset_dict.items():
        if unique_id in train_idx:
            train_dataset.append((rows, unique_id))
        elif unique_id in valid_idx:
            valid_dataset.append((rows, unique_id))
        elif unique_id in test_idx:
            test_dataset.append((rows, unique_id))
        else:
            import pdb; pdb.set_trace()
    # train_dataset, valid_dataset, test_dataset = split_dataset(dataset)
    
    #Initialize preprocessor and process each split
    preprocessor = Preprocessor(args.data_dir)
    preprocessor.preprocess(train_dataset, fold='train')
    preprocessor.preprocess(valid_dataset, fold='valid')
    preprocessor.preprocess(test_dataset, fold='test')

if __name__ == "__main__":
    main()