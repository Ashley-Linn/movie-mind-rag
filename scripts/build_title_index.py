"""
构建标题向量索引
"""

import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

def main():
    # 1. 加载元数据
    metadata_path = Path("index/movie_metadata_tr.pkl")  
    if not metadata_path.exists():
        print(f"错误：元数据文件 {metadata_path} 不存在")
        return

    with open(metadata_path, "rb") as f:
        data = pickle.load(f)

    movie_info = data["movie_info"]
    mvids = list(movie_info.keys())
    titles = [movie_info[mvid].get("Title", "") for mvid in mvids]

    print(f"加载了 {len(mvids)} 部电影")

    # 2. 加载模型（与检索器相同）
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    print("生成标题向量...")
    embeddings = model.encode(titles, normalize_embeddings=True, show_progress_bar=True)

    # 3. 构建 FAISS 索引
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 内积，归一化后即余弦相似度
    index.add(embeddings.astype(np.float32))

    # 4. 保存
    output_dir = Path("index")
    output_dir.mkdir(exist_ok=True)
    faiss.write_index(index, str(output_dir / "title_index.index"))
    with open(output_dir / "title_mvids.pkl", "wb") as f:
        pickle.dump(mvids, f)

    print(f"标题索引已保存到 {output_dir}/title_index.index")
    print(f"mvid 列表已保存到 {output_dir}/title_mvids.pkl")

if __name__ == "__main__":
    main()