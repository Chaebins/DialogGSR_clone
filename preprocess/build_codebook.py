import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='/data')
args = parser.parse_args()

dataset = []
data_dir = args.data_dir
def reverse(relation):
    if '~' in relation:
        new_relation = relation[1:]
    else:
        new_relation = '~' + relation
    return new_relation

with open(f"{data_dir}/opendialkg_entities.txt", 'r') as f:
    entities = f.readlines()

with open(f"{data_dir}/opendialkg_relations.txt", 'r') as f:
    relations = f.readlines()

entity_codebook = {}
for i, entity in enumerate(entities):
    entity = entity.strip().lower()
    entity_codebook[entity] = i

relation_codebook = {}
relation_idx = 0
for relation in relations:
    relation = relation.strip().lower()
    relation_codebook[relation] = relation_idx
    relation_idx += 1
    reverse_relation = reverse(relation).lower()
    if reverse_relation not in relation_codebook.keys():
        relation_codebook[reverse_relation] = relation_idx
        relation_idx += 1

with open(f"{data_dir}/entity_codebook.pkl", 'wb+') as f:
    pickle.dump(entity_codebook, f)

with open(f"{data_dir}/relation_codebook.pkl", 'wb+') as f:
    pickle.dump(relation_codebook, f)