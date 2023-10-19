from rouge import Rouge
import jieba as jb  # you can use any other word cutting library
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

# path_qwen = "/workspace/Qwen-7B-Chat"
# tokenizer = AutoTokenizer.from_pretrained(path_qwen, trust_remote_code=True)

hypothesis = "你好啊明天"
hypothesis = " ".join(jb.cut(hypothesis))

# a = [str(i, encoding="utf-8") for i in tokenizer.tokenize(hypothesis)]
# print(a)
# hypothesis = ' '.join(a)
print(hypothesis)

reference = "你好啊明天"
reference = " ".join(jb.cut(reference))
print(reference)
# reference = ' '.join([str(i, encoding="utf-8") for i in tokenizer.tokenize(reference)])

hypothesis = [hypothesis]
reference = [reference]

rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference)
print(type(scores[0]["rouge-l"]["f"]))
