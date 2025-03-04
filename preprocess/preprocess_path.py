import json
import random
from tqdm import tqdm
import networkx as nx
import argparse
import os
from typing import Dict, List, Tuple, Optional, Any
import pickle
import torch
from transformers import AutoTokenizer

class Trie(object):
    def __init__(self, sequences: List[List[int]] = [], values: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences and not values:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1
        if sequences and values:
            for sequence, value in zip(sequences, values):
                Trie._add_to_trie_values(sequence, value, self.trie_dict)
                self.len += 1


        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _add_to_trie_values(sequence: List[int], value: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {-1: value[0]}
            Trie._add_to_trie_values(sequence[1:], value[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        value_list = []
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            output_list = []
            for o in output:
                if o == -1:
                    continue
                next_value = trie_dict[o][-1]
                value_list.append(next_value)
                output_list.append(o)
                
            if append_trie and bos_token_id in output_list:
                output_list.remove(bos_token_id)
                output_list += list(append_trie.trie_dict.keys())
            return output_list, value_list
        
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)
    
    
class GraphPathFinder:
    def __init__(self, graph: nx.DiGraph, entities: List[str], katz_dict: Dict[str, float], max_paths: int = 100):
        self.graph = graph
        self.entities = entities
        self.katz_dict = katz_dict
        self.max_paths = max_paths
        
    def find_paths(self):
        all_paths = []
        
        for node in self.entities:
            if node == "" or node not in self.graph.nodes:
                continue
            one_hop_paths = self._get_one_hop_paths(node)
            two_hop_paths = self._get_two_hop_paths(node)
            if len(two_hop_paths) >= self.max_paths:
                two_hop_paths = random.sample(two_hop_paths, self.max_paths)
            
            all_paths.extend(one_hop_paths)
            all_paths.extend(two_hop_paths)
        
        
        return all_paths
    
    def _get_one_hop_paths(self, node: str) -> List[Tuple[List[Tuple[str, str, str]], Tuple[float, ...]]]:
        neighbors = self._get_filtered_neighbors(node)
        path_score_list = []
        for neighbor in neighbors:
            if "reverse_" in self.graph[node][neighbor]['relation']:
                path_score_list.append(([(neighbor, self.graph[node][neighbor]['relation'].replace("reverse_", ""), node)],
                (self.katz_dict[node], self.katz_dict[neighbor])))
            else:
                path_score_list.append(([(node, self.graph[node][neighbor]['relation'], neighbor)],
                (self.katz_dict[node], self.katz_dict[neighbor])))
        return path_score_list
        
    def _get_two_hop_paths(self, node: str) -> List[Tuple[List[Tuple[str, str, str]], Tuple[float, ...]]]:
        two_hop_paths = []
        try:
            neighbors = self._get_filtered_neighbors(node)
            for neighbor in neighbors:
                two_hop_neighbors = self._get_filtered_two_hop_neighbors(node, neighbor)
                for two_hop_neighbor in two_hop_neighbors:
                    path = self._construct_two_hop_path(node, neighbor, two_hop_neighbor)
                    if path:
                        two_hop_paths.append(path)
        except:
            import pdb; pdb.set_trace()
            pass
        return two_hop_paths
        
    def _get_filtered_neighbors(self, node: str) -> List[str]:
        try:
            neighbors = list(self.graph.successors(node))
            rel_dict = {}
            for neighbor in neighbors:
                rel = self.graph[node][neighbor]['relation']
                rel_dict.setdefault(rel, []).append(neighbor)
            
            filtered_neighbors = []
            for neighbors in rel_dict.values():
                filtered_neighbors.extend(neighbors)
            return filtered_neighbors
        except:
            import pdb; pdb.set_trace()
            return []
        
    def _get_filtered_two_hop_neighbors(self, root_node:str, node: str) -> List[str]:
        neighbors = list(self.graph.successors(node))
        rel_dict = {}
        for neighbor in neighbors:
            rel = self.graph[node][neighbor]['relation']
            rel_dict.setdefault(rel, []).append(neighbor)
        
        filtered_neighbors = []
        for neighbors in rel_dict.values():
            if root_node in neighbors:
                neighbors.remove(root_node)
            if len(neighbors) >= 10:
                neighbors = random.sample(neighbors, 10)
            filtered_neighbors.extend(neighbors)
        return filtered_neighbors

    def _construct_two_hop_path(self, node: str, neighbor: str, two_hop_neighbor: str) -> Optional[Tuple[List[Tuple[str, str, str]], Tuple[float, ...]]]:
        score = (self.katz_dict[node], self.katz_dict[neighbor], self.katz_dict[two_hop_neighbor])
        rel1 = self.graph[node][neighbor]['relation']
        rel2 = self.graph[neighbor][two_hop_neighbor]['relation']
        
        if "reverse" in rel1:
            if neighbor in self.entities:
                return None
            rel1 = rel1.replace("reverse_", "")
            path1 = (neighbor, rel1, node)
        else:
            path1 = (node, rel1, neighbor)

        if "reverse" in rel2:
            rel2 = rel2.replace("reverse_", "")
            path2 = (two_hop_neighbor, rel2, neighbor)
        else:
            path2 = (neighbor, rel2, two_hop_neighbor)
        
        return([path1, path2], score)

def process_data_entry(data, trie_dict, tokenizer):
    episode_id = data['episode_id']
    turn_id = data['turn_id']
    entities = data['entities']
    
    if episode_id not in trie_dict:
        trie_dict[episode_id] = {}
        
    G = nx.DiGraph()
    for sub, rel, obj in data['triplets']:
        if sub == "" or obj == "":
            continue
        G.add_edge(sub, obj, relation=rel)
        G.add_edge(obj, sub, relation=f"reverse_{rel}")
    
    katz_dict = nx.katz_centrality(G, alpha=0.005, beta=1.0, max_iter=2, tol=1e+6)
    
    
    path_finder = GraphPathFinder(G, entities, katz_dict)
    paths_scores = path_finder.find_paths()
    
    paths_nlp = []
    scores = []
    for path, score in paths_scores:
        nlp_path = construct_paths(path, entities)
        paths_nlp.append(nlp_path)
        scores.append(score)
        
    if len(paths_nlp) > 0:
        input_ids = tokenizer.batch_encode_plus(
            paths_nlp,
            return_tensors="pt",
            padding=True,
            max_length=128
        )
        trie = make_constraints(input_ids, scores)
    else:
        input_ids = None
        trie = None
        
    trie_dict[episode_id][turn_id] = trie
    
    
        
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

def make_constraints(input_ids: Dict[str, torch.Tensor], scores: List[Tuple[float, ...]]) -> Trie:
    input_ids_tensor = input_ids['input_ids']
    input_ids_list = input_ids_tensor.tolist()
    
    score_list = []
    for i, tr in enumerate(input_ids_list):
        curr_score_list = [0]
        score = scores[i][0]
        for token in tr:
            if token >= 32101:
                score = scores[i][1]
            elif token >= 32109:
                score = scores[i][2]
            curr_score_list.append(score)
        score_list.append(curr_score_list)
                    
    return Trie([[0]+tokens for tokens in input_ids_list], score_list)

def process_fold(data_dir, fold, tokenizer):
    trie_dict = {}
    with open(os.path.join(data_dir, f"{fold}.jsonl"), "r") as f:
        for line in tqdm(f):
            data = json.loads(line)
            process_data_entry(data, trie_dict, tokenizer)
    with open(os.path.join(data_dir, f"trie_{fold}.pkl"), "wb") as f:
        pickle.dump(trie_dict, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='../data')
    args = parser.parse_args()

    data_dir = args.data_dir
    
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    special_tokens = ['[HEAD]']
    for i in range(1, 3):
        special_tokens.extend([
            f'[Int{i*2-1}_1]', f'[Int{i*2-1}_2]',
            f'[Rev{i*2-1}_1]', f'[Rev{i*2-1}_2]',
            f'[Int{i*2}_1]', f'[Int{i*2}_2]',
            f'[Rev{i*2}_1]', f'[Rev{i*2}_2]'
        ])
    special_tokens.append('[TAIL]')
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    
    process_fold(data_dir, "train", tokenizer)
    process_fold(data_dir, "valid", tokenizer)
    process_fold(data_dir, "test", tokenizer)
    
if __name__ == "__main__":
    main()