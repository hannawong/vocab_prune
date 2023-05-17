from transformers import AutoTokenizer
import json
tokenizer = AutoTokenizer.from_pretrained("my_tokenizer_test") ##vocab:250680
cnt = 0
dic = {}
text = '''BLOOM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bigscience/bigscience-small-testing",
]z

def _make_causal_mask(
    input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int
) -> torch.BoolTensor:
    """
    Make causal mask used for self-attention.
    """
    batch_size, target_length = input_ids_shape
    return expanded_mask
'''
output = tokenizer.encode(text)
print(output)

ans = ""
for i in output:
    decoded = tokenizer.decode([i])
    print(decoded)
    ans += decoded + "<sep>"
print(repr(ans))

out = open("/mnt/home/vocab_prune/my_tokenizer_test/realvocab.txt","w")
for i in range(160130):
    decoded = tokenizer.decode([i])
    out.write(str(i)+"\t"+decoded+"\n")