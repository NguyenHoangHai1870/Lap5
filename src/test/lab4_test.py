# test/lab4_test.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from representations.word_embedder import WordEmbedder

embedder = WordEmbedder("glove-wiki-gigaword-50")

print("Vector for 'king':")
print(embedder.get_vector("king")[:10])  # in 10 phần tử đầu

print("\nSimilarity:")
print("king vs queen:", embedder.get_similarity("king", "queen"))
print("king vs man:", embedder.get_similarity("king", "man"))

print("\nMost similar to 'computer':")
print(embedder.get_most_similar("computer", top_n=10))

print("\nDocument embedding:")
doc_vec = embedder.embed_document("The queen rules the country.")
print(doc_vec[:10])
