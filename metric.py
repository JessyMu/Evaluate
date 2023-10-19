import evaluate
import torch
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings


class Metric:
    def __init__(self, method, predictions, references) -> None:
        self.method = method
        self.predictions = predictions
        self.references = references

    def initModel(self, method):
        if method == 0:  # Bleu
            import jieba as jb

            path = "config/bleu.py"
            tokenizer = jb
            model = Bleu(path, tokenizer)
        elif method == 1:  # Rouge
            import jieba as jb

            tokenizer = jb
            model = RougeL(tokenizer)
        elif method == 2:  # Yule
            path = ""
            model = Yule(path)
        elif method == 3:  # Embedding
            path = "/media/jesse/data/project/bge_qwen_docker/project/bge-large-zh"
            model = Cosine(path)
        else:
            model = None
        return model

    def score(self):
        self.model = self.initModel(self.method)
        if self.model:
            scores = self.model.evaluate(self.predictions, self.references)
        return scores


class Bleu:
    def __init__(self, path, tokenizer) -> None:
        self.bleu = evaluate.load(path)
        self.tokenizer = tokenizer
        super().__init__()

    def evaluate(self, predictions, references):
        # TODO: tokenizer???
        assert len(references) == len(
            predictions
        ), "amount of sentence in predictions and references must be equal"
        length = len(predictions)
        res_list = []
        for i in range(0, length):
            reference_list = [[]]
            predict_list = []
            prediction = " ".join(self.tokenizer.cut(predictions[i]))
            predict_list.append(prediction)
            reference = " ".join(self.tokenizer.cut(references[i]))
            reference_list[0].append(reference)
            print(prediction, "---", reference)
            result = self.bleu.compute(
                predictions=predict_list, references=reference_list
            )
            res = round(result["score"], 1) / 100
            res_list.append(res)
        return res_list


class RougeL:
    def __init__(self, tokenizer) -> None:
        from rouge import Rouge

        self.rouge = Rouge()
        self.tokenizer = tokenizer
        super().__init__()

    def evaluate(self, predictions, references):
        # TODO: tokenizer???
        assert len(references) == len(
            predictions
        ), "amount of sentence in predictions and references must be equal"
        length = len(predictions)
        res_list = []
        for i in range(0, length):
            reference_list = []
            predict_list = []
            prediction = " ".join(self.tokenizer.cut(predictions[i]))
            predict_list.append(prediction)
            reference = " ".join(self.tokenizer.cut(references[i]))
            reference_list.append(reference)
            print(prediction, "---", reference)
            result = self.rouge.get_scores(predict_list, reference_list)
            res = result[0]["rouge-l"]["f"]
            res_list.append(str(res))
        return res_list


class Yule:
    def __init__(self, path) -> None:
        pass


class Cosine:
    def __init__(self, path) -> None:
        EMBEDDING_DEVICE = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=path, model_kwargs={"device": EMBEDDING_DEVICE}
        )
        pass

    def evaluate(self, predictions, references):
        assert len(references) == len(
            predictions
        ), "amount of sentence in predictions and references must be equal"
        length = len(predictions)
        res_list = []
        for i in range(0, length):
            vector1 = np.array(self.embeddings.embed_query(predictions[i]))
            vector2 = np.array(self.embeddings.embed_query(references[i]))

            # 计算向量的内积
            dot_product = np.dot(vector1, vector2)

            # 计算向量的范数（模）
            norm_vector1 = np.linalg.norm(vector1)
            norm_vector2 = np.linalg.norm(vector2)

            # 计算余弦相似度
            cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
            res_list.append(str(cosine_similarity))
        return res_list


predictions = ["我今天很开心", "今天气很好", "今天晚上吃什么", "你在哪个位置", "你好啊明天"]
references = ["我今天也很开心", "今天天气还行", "今晚吃什么", "你在哪里", "你好啊明天"]
metric = Metric(0, predictions=predictions, references=references)
print(metric.score())
