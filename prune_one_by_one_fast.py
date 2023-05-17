import os,json
import copy
from tqdm import tqdm
from collections import defaultdict
path = "./prune_lang"

pruned_set = set()
for file_name in os.listdir(path):
    full_path = os.path.join(path, file_name)
    if full_path.endswith("json"):
        pruned_lines = json.load(open(full_path))
        for key in pruned_lines:
            pruned_set.add(int(key))
pruned_set = list(pruned_set)

tokenizer = json.load(open("/mnt/home/tokenizer/tokenizer.json"))
###### processing vocab -> token2id; id2token ######
vocab_token2id = tokenizer["model"]["vocab"] 
vocab_id2token = {}
for key in vocab_token2id:
    vocab_id2token[vocab_token2id[key]] = key

###### processing merges -> fulltoken2id; token2id 
merges = tokenizer["model"]["merges"] #list
fulltoken2id = {}
token2id = defaultdict(list) ##'çľģå§Ķ': (239862, 0)
for i in range(len(merges)):
    two_tokens = merges[i].split()
    fulltoken2id["".join(two_tokens)] = i
    token2id[two_tokens[0]].append((i,0))
    token2id[two_tokens[1]].append((i,1))

####### get mapping from vocab to merges, and merges to vocab #######################
vocab_id2merge_id = {}
merge_id2vocab_id = {}
for i in range(len(merges)):
    two_tokens = merges[i].split()
    full_token = "".join(two_tokens)
    assert full_token in vocab_token2id
    merge_id2vocab_id[i] = vocab_token2id[full_token]
    vocab_id2merge_id[vocab_token2id[full_token]] = i 
##############################################

def delete_prune_token(prune_token):
    print(prune_token)
    if prune_token not in vocab_token2id: return ##cannot compose other tokens any more
    del vocab_token2id[prune_token]
    del fulltoken2id[prune_token]
    token_pos_list = token2id[prune_token]
    for merge_id, pos in token_pos_list:
        next_prune_token = "".join(merges[merge_id].split())
        delete_prune_token(next_prune_token)
for prune_id in tqdm(pruned_set):
    delete_prune_token(vocab_id2token[prune_id])

new_vocab = {}
cnt = 0
for key in vocab_token2id:
    new_vocab[key] = cnt
    cnt += 1
new_merges = []
for key in fulltoken2id:
    new_merges.append(merges[fulltoken2id[key]])

tokenizer["model"]["vocab"] = new_vocab
tokenizer["model"]["merges"] = new_merges
with open('./my_tokenizer_test/tokenizer.json', 'w',encoding="utf-8") as f:
    json.dump(tokenizer, f, indent=4,ensure_ascii=False)
