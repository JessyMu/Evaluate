import evaluate

class Metric():
    def __init__(self,method,predictions,references) -> None: 
        self.method=method
        self.predictions=predictions
        self.references=references
    def initModel(self,method):
        if method==0:#Bleu
            path='metrics/bleu.py'
            tokenizer=''
            model=Bleu(path,tokenizer)
        elif method==1:#Rouge
            path=''
            model=RougeL(path,tokenizer)
        elif method==2:#Yule
            path=''
            model=Yule(path)
        elif method==3:#Embedding
            path=''
            model=Cosine(path)
        else:
            model=None
        return model
    def score(self):
        self.model=self.initModel(self.method)
        if self.model:
            scores=self.model.evaluate(self.predictions,self.references)
        return scores
class Bleu():
    def __init__(self,path,tokenizer) -> None:
        self.bleu = evaluate.load(path)
        self.tokenizer=tokenizer
        super().__init__()
    def evaluate(self,predictions,references):
        #TODO: tokenizer???
        assert len(references)==len(predictions),"amount of sentence in predictions and references must be equal"
        length=len(predictions)
        res_list=[]
        for i in range(0,length):
            reference_list=[[]]
            predict_list=[]
            predict_list.append(predictions[i])
            reference_list[0].append(references[i])
            result=self.bleu.compute(predictions=predict_list,references=reference_list)
            res=round(result["score"], 1)/100
            res_list.append(res)
        return res_list

class RougeL():
    def __init__(self,path,tokenizer) -> None:
        from rouge import Rouge
        self.rougel = Rouge()
        self.tokenizer=tokenizer
        super().__init__()
    def evaluate(self,predictions,references,tokenizer):
        #TODO: tokenizer???
        assert len(references)==len(predictions),"amount of sentence in predictions and references must be equal"
        length=len(predictions)
        res_list=[]
        for i in range(0,length):
            reference_list=[[]]
            predict_list=[]
            predict_list.append(predictions[i])
            reference_list[0].append(references[i])
            result=self.bleu.compute(predictions=predict_list,references=reference_list)
            res=round(result["score"], 1)/100
            res_list.append(res)
        return res_list

class Yule():
    def __init__(self,path) -> None:
        pass

class Cosine():
    def __init__(self,path) -> None:
        pass


predictions = ["我 今 天 很 开 心",'今 天 天 气 很 好','今 天 晚 上 吃 什 么','你 在 哪 个 位 置']
references = ["我 今 天 也 很 开 心",'今 天 天 气 还 行','今 晚 吃 什 么','你 在 哪 里']
# predictions=['你好']
# references=['你好']
# print(predictions[1])
metric=Metric(0,predictions=predictions,references=references)
print(metric.score())
