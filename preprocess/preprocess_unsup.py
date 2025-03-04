import json
import random
from tqdm import tqdm
import networkx as nx
import argparse
import os
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
    
class GraphPathFinder:
    def __init__(self, max_paths: int = 100):
        self.graph = nx.DiGraph()
        self.entities = set()
        self.max_paths = max_paths
        
    def process_fold(self, data_dir, fold):
        with open(os.path.join(data_dir, f"{fold}_public.jsonl"), "r") as f:
            for line in tqdm(f):
                data = json.loads(line)
                self._add_triplets(data['triplets'], data['entities'])

        
    def _add_triplets(self, triplets:List[Tuple[str, str, str]], entities:List[str]):
        for sub, rel, obj in triplets:
            if sub == "" or obj == "":
                continue
            self.graph.add_edge(sub, obj, relation=rel)
            self.graph.add_edge(obj, sub, relation=f"reverse_{rel}")
        self.entities.update(entities)

    def find_paths(self):
        all_paths = []
        for node in tqdm(self.entities):
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
        return [
            (node, self.graph[node][neighbor]['relation'], neighbor)
            for neighbor in neighbors
        ]
        
    def _get_two_hop_paths(self, node: str) -> List[Tuple[List[Tuple[str, str, str]], Tuple[float, ...]]]:
        two_hop_paths = []
        neighbors = self._get_filtered_neighbors(node)
        for neighbor in neighbors:
            two_hop_neighbors = self._get_filtered_two_hop_neighbors(node, neighbor)
            for two_hop_neighbor in two_hop_neighbors:
                path = self._construct_two_hop_path(node, neighbor, two_hop_neighbor)
                if path:
                    two_hop_paths.append(path)
        return two_hop_paths
        
    def _get_filtered_neighbors(self, node: str) -> List[str]:
        neighbors = list(self.graph.successors(node))
        rel_dict = {}
        for neighbor in neighbors:
            rel = self.graph[node][neighbor]['relation']
            rel_dict.setdefault(rel, []).append(neighbor)
        
        filtered_neighbors = []
        for neighbors in rel_dict.values():
            filtered_neighbors.extend(neighbors)
        return filtered_neighbors
        
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
        rel1 = self.graph[node][neighbor]['relation']
        rel2 = self.graph[neighbor][two_hop_neighbor]['relation']
        
        return (node, rel1, neighbor, rel2, two_hop_neighbor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='../data')
    args = parser.parse_args()

    data_dir = args.data_dir
    
    path_finder = GraphPathFinder()
    path_finder.process_fold(data_dir, "train")
    path_finder.process_fold(data_dir, "valid")
    path_finder.process_fold(data_dir, "test")
    path_list = path_finder.find_paths()
    
    path_data = []
    for i in range(len(path_list)//64 + 1):
        path_data.append({"triplets":path_list[i*64:(i+1)*64]})
    
    with open(os.path.join(data_dir, f"unsup_path.jsonl"), 'w') as f:
        for item in path_data:
            f.write(json.dumps(item) + "\n")
        f.close()
    

    
if __name__ == "__main__":
    main()