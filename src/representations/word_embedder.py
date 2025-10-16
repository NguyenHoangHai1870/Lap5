
import gensim.downloader as api
import numpy as np

class WordEmbedder:
    def __init__(self, model_name: str = "glove-wiki-gigaword-50"):
        print(f"Loading model: {model_name} ...")
        self.model = api.load(model_name)
        print("Model loaded successfully!")

    def get_vector(self, word: str):
        if word in self.model:
            return self.model[word]
        else:
            print(f"'{word}' not in vocabulary (OOV).")
            return np.zeros(self.model.vector_size)

    def get_similarity(self, word1: str, word2: str):
        if word1 in self.model and word2 in self.model:
            return self.model.similarity(word1, word2)
        else:
            return None

    def get_most_similar(self, word: str, top_n: int = 10):
        if word in self.model:
            return self.model.most_similar(word, topn=top_n)
        else:
            return []

    def embed_document(self, document: str):
        tokens = document.lower().split()
        vectors = []
        for token in tokens:
            if token in self.model:
                vectors.append(self.model[token])

        if len(vectors) == 0:
            return np.zeros(self.model.vector_size)
        else:
            return np.mean(vectors, axis=0)