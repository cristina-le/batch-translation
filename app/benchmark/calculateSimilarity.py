from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. Tải mô hình embedding
model = SentenceTransformer('jinaai/jina-embeddings-v3')

# 2. Chuẩn bị dữ liệu (ví dụ)
japanese_texts = ["日本語のテキスト例", "もう一つの例文"]
machine_translations = ["Example of Japanese text", "Another example sentence"]
human_translations = ["Example of Japanese text", "Another example sentence"]

# 3. Tạo embedding
japanese_embeddings = model.encode(japanese_texts)
machine_embeddings = model.encode(machine_translations)
human_embeddings = model.encode(human_translations)

# 4. Tính độ tương đồng
machine_similarities = [cosine_similarity([jp_emb], [mt_emb])[0][0] 
                       for jp_emb, mt_emb in zip(japanese_embeddings, machine_embeddings)]
human_similarities = [cosine_similarity([jp_emb], [ht_emb])[0][0] 
                     for jp_emb, ht_emb in zip(japanese_embeddings, human_embeddings)]

# 5. Tính điểm
scores = [m_sim / h_sim * 100 if h_sim > 0 else 0 
         for m_sim, h_sim in zip(machine_similarities, human_similarities)]
average_score = np.mean(scores)

print(f"Điểm chất lượng bản dịch máy: {average_score:.2f}%")
