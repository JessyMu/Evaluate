# 执行
 python metric.py
# 输入
predictions = ["我今天很开心", "今天气很好", "今天晚上吃什么", "你在哪个位置", "你好啊明天"]<br/>
references = ["我今天也很开心", "今天天气还行", "今晚吃什么", "你在哪里", "你好啊明天"]<br/>

## Metric
metric = Metric(method=0, predictions=predictions, references=references)<br/>
  method:<br/>
    0:bleu<br/>
    1:rougeL<br/>
    2:Yule<br/>
    3:Embedding search(cosine)<br/>
# 输出
print(metric.score())
