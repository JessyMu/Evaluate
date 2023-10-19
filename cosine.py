import torch
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings


EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(model_name= "/media/jesse/data/project/bge_qwen_docker/project/bge-large-zh",
                                model_kwargs={'device': EMBEDDING_DEVICE}
                                )
def cosine_similarity(embeddings, sentence1, sentence2):
    
    vector1 = np.array(embeddings.embed_query(sentence1))
    vector2 = np.array(embeddings.embed_query(sentence2))

    # 计算向量的内积
    dot_product = np.dot(vector1, vector2)

    # 计算向量的范数（模）
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    # 计算余弦相似度
    cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
    return cosine_similarity

print(cosine_similarity(embeddings,'今天是个好日子','今天'))
