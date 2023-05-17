from tokenizers import ByteLevelBPETokenizer
import os

ByteLevelBPE_tokenizer_pt = ByteLevelBPETokenizer()

# Get list of paths to corpus files
paths = ["MTData/cn_wiki_sample.txt"]

# Customize training with <|endoftext|> special GPT2 token
ByteLevelBPE_tokenizer_pt.train(files=paths, 
                                vocab_size=50257, 
                                min_frequency=2, 
                                special_tokens=["<|endoftext|>"])

# Get sequence length max of 1024
ByteLevelBPE_tokenizer_pt.enable_truncation(max_length=1024)

ByteLevelBPE_tokenizer_pt_rep = 'ByteLevelBPE_tokenizer_pt'
path_to_ByteLevelBPE_tokenizer_pt_rep = "MTData_byte/"+ByteLevelBPE_tokenizer_pt_rep
os.makedirs(path_to_ByteLevelBPE_tokenizer_pt_rep,exist_ok = True)
ByteLevelBPE_tokenizer_pt.save_model(str(path_to_ByteLevelBPE_tokenizer_pt_rep))


ByteLevelBPE_tokenizer_pt_vocab = ByteLevelBPE_tokenizer_pt.get_vocab() 
ByteLevelBPE_tokenizer_pt_vocab_ls = [k for k, v in sorted(ByteLevelBPE_tokenizer_pt_vocab.items(), key=lambda item: item[1])]
print(len(ByteLevelBPE_tokenizer_pt_vocab_ls),ByteLevelBPE_tokenizer_pt_vocab_ls[:5])


ByteLevelBPE_tokenizer_pt = ByteLevelBPETokenizer(
    "MTData_byte/ByteLevelBPE_tokenizer_pt/vocab.json",
    "MTData_byte/ByteLevelBPE_tokenizer_pt/merges.txt",
    add_prefix_space=True,
)

text = "1948年第一屆國民大會召開，國民政府改組為中華民國政府，蔣中正當選為行憲後的第一任中華民國總統。"
output = ByteLevelBPE_tokenizer_pt.encode(text)

print(output.ids,output.tokens,output.offsets)
back_to_text = ByteLevelBPE_tokenizer_pt.decode(output.ids)
for i in range(100,1000):
    print(i,ByteLevelBPE_tokenizer_pt.decode([i]))

print('input text:', text)
print('tokens ids:', output.ids)
print('back to text:', back_to_text)
