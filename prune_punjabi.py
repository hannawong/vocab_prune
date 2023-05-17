def is_contains_punjabi(strs): ##detect if a string is composed of Arabic characters
    if "�" in strs: return False
    for _char in strs:
        if '\u0a00' <= _char <= '\u0a7f':
            return True
    return False




from transformers import AutoTokenizer
import json
tokenizer = AutoTokenizer.from_pretrained("../tokenizer") ##vocab:250680
cnt = 0
dic = {}
for i in range(0,300000):
    decoded = tokenizer.decode([i])
    if is_contains_punjabi(decoded):
        dic[i] = decoded
        cnt += 1
print(cnt)

import json
json_str = json.dumps(dic, ensure_ascii=False).encode('utf-8')
with open("punjabi.json", "wb") as f:
    f.write(json_str)