import tarfile
from gensim.utils import simple_preprocess
from pathlib import Path
from gensim.models import Word2Vec

data_path = Path(r"D:\ScalaProjects\nlp\spark_labs\data\UD_English-EWT.tar.gz")

def read_sentences():
    with tarfile.open(data_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".conllu"):
                f = tar.extractfile(member)
                for line in f:
                    line = line.decode('utf-8')
                    tokens = simple_preprocess(line)
                    if tokens:
                        yield tokens

print("Training Word2Vec model...")
sentences = list(read_sentences())
model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=2, workers=4)
model.save(r"D:\ScalaProjects\nlp\results\word2vec_ewt.model")
print("Model saved!")

print("\nMost similar to 'computer':")
print(model.wv.most_similar("computer", topn=5))
