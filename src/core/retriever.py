"""
检索器模块
支持：单条搜索 / 批量搜索 / 返回电影 + 完整剧情
"""

import pickle
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Settings


# 全局单例，只加载一次
_model = None
_index = None
_mvid_list = None
_movie_info = None
_title_index = None
_title_mvids = None


def _load_assets(index_dir: Path):
    global _model, _index, _mvid_list, _movie_info, _title_index, _title_mvids
    if _index is not None:
        return

    # 加载索引
    print("[DEBUG] Loading FAISS index...", flush=True)
    _index = faiss.read_index(str(index_dir / "movie_plots.index"))
    print("[DEBUG] FAISS index loaded.", flush=True)
    
    # 加载元数据
    print("[DEBUG] Loading metadata pickle...", flush=True)
    with open(index_dir / "movie_metadata_tr_zh2000.pkl", "rb") as f:
        data = pickle.load(f)
    _mvid_list = data["mvid_list"]
    _movie_info = data["movie_info"]
    print("[DEBUG] Metadata loaded.", flush=True)
    
    # 加载标题索引
    title_index_path = index_dir / "title_index.index"
    title_mvids_path = index_dir / "title_mvids.pkl"
    if title_index_path.exists() and title_mvids_path.exists():
        _title_index = faiss.read_index(str(title_index_path))
        with open(title_mvids_path, "rb") as f:
            _title_mvids = pickle.load(f)
        print("标题索引加载完成")
    else:
        _title_index = None
        _title_mvids = None
        print("警告：标题索引未找到，将只使用剧情检索")

    
    # 加载模型
    model_name = Settings.EMBEDDING_MODEL
    print(f"[INFO] Loading model: {model_name} (will download if not cached)")
    _model = SentenceTransformer(model_name, device='cpu')
    print("[INFO] Model loaded.")
        
def search(
    query: str,
    top_k: int = Settings.TOP_K,
    similarity_threshold: float = Settings.SIMILARITY_THRESHOLD,
    index_dir: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    单条搜索：输入一句话 → 返回最相关的电影（带剧情）
    """
    if index_dir is None:
        index_dir = project_root / "index"

    _load_assets(index_dir)

    # ================== 1. 全局标题精确匹配（完全相等） ==================
    query_lower = query.strip().lower()
    exact_match_mvids = []
    for mvid, info in _movie_info.items():
        title = info.get('Title', '').lower()
        if query_lower == title:
            exact_match_mvids.append(mvid)
    exact_match_mvids = list(dict.fromkeys(exact_match_mvids))
    print(f"完全匹配电影数: {len(exact_match_mvids)}")

    use_hard_boost = len(exact_match_mvids) > 0

    # ========== 2. 剧情向量检索 ==========
    q_vec = _model.encode([query], normalize_embeddings=True)[0]
    q_vec = q_vec.reshape(1, -1).astype(np.float32)

    k_chunks = min(top_k * 5, _index.ntotal)
    scores, indices = _index.search(q_vec, k_chunks)
    scores = scores[0]
    indices = indices[0]

    movie_best = {}
    for idx, score in zip(indices, scores):
        if score < similarity_threshold:
            continue
        mvid = _mvid_list[idx]
        if mvid not in movie_best or score > movie_best[mvid][0]:
            movie_best[mvid] = (score, _movie_info.get(mvid, {}))

    plot_ranked = [mvid for mvid, (score, _) in sorted(movie_best.items(), key=lambda x: x[1][0], reverse=True)]

    # ========== 3. 标题向量检索 ==========
    title_ranked = []
    if _title_index is not None and _title_mvids is not None:
        title_q_vec = _model.encode([query], normalize_embeddings=True)[0].reshape(1, -1).astype(np.float32)
        title_scores, title_indices = _title_index.search(title_q_vec, min(top_k * 3, len(_title_mvids)))
        title_ranked = [_title_mvids[i] for i in title_indices[0]]

    # ========== 4. RRF 融合 ==========
    def rrf_fusion(ranked_lists, exclude=set(), k=60):
        scores = {}
        for rank_list in ranked_lists:
            for rank, mvid in enumerate(rank_list, start=1):
                if mvid in exclude:
                    continue
                scores[mvid] = scores.get(mvid, 0) + 1 / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ========== 5. 构建最终列表（统一使用剧情相似度展示，排序依据 RRF 分数） ==========
    if use_hard_boost:
        exclude_set = set(exact_match_mvids)
        fused = rrf_fusion([plot_ranked, title_ranked], exclude=exclude_set)
        # 硬置顶部分：完全匹配的电影（可能来自 movie_best 或全局）
        exact_list = []
        for mvid in exact_match_mvids:
            if mvid in movie_best:
                score, info = movie_best[mvid]
            else:
                score = 1.0
                info = _movie_info.get(mvid, {})
            exact_list.append((mvid, score, info))
        exact_list.sort(key=lambda x: x[1], reverse=True)
        # 剩余电影：使用剧情相似度作为展示分数（若无则为0.0）
        fused_movies = []
        for mvid, rrf_score in fused:
            plot_score = movie_best.get(mvid, (0.0, None))[0]
            info = _movie_info.get(mvid, {})
            fused_movies.append((mvid, plot_score, info))
        final_movies = exact_list + fused_movies
    else:
        fused = rrf_fusion([plot_ranked, title_ranked], exclude=set())
        final_movies = []
        for mvid, rrf_score in fused:
            plot_score = movie_best.get(mvid, (0.0, None))[0]
            info = _movie_info.get(mvid, {})
            final_movies.append((mvid, plot_score, info))

    final_movies = final_movies[:top_k]

    # ========== 6. 组装结果 ==========
    results = []
    for mvid, score, info in final_movies:
        # cast 处理
        cast_full = info.get("Cast_limited", "")
        if cast_full and cast_full != "Unknown":
            cast_list = [c.strip() for c in cast_full.split(',')[:3]]
            cast_short = ', '.join(cast_list)
        else:
            cast_short = "Unknown"

        # 剧情文本
        plot_text = info.get("Plot_summary_zh", "")
        if not plot_text:
            plot_text = info.get("Plot_summary", "")
        if not plot_text:
            plot_raw = info.get("Plot_cleaned", "")
            plot_text = plot_raw[:350] + ("..." if len(plot_raw) > 350 else "")

        # 相似度显示：完全匹配且无剧情分的显示1.0，否则显示剧情相似度
        if mvid in exact_match_mvids and mvid not in movie_best:
            display_score = "1.0 (标题精准匹配)"
        else:
            display_score = round(float(score), 4)

        results.append({
            "mvid": mvid,
            "title": info.get("Title", "N/A"),
            "year": info.get("Release Year", "N/A"),
            "genre": info.get("Primary_Genre", "N/A"),
            "origin": info.get("Origin/Ethnicity", "Unknown"),
            "director": info.get("Director", "N/A"),
            "cast": cast_short,
            "plot": plot_text,
            "similarity": display_score
        })
    return results


def batch_search(
    queries: List[str],
    top_k: int = Settings.TOP_K,
    **kwargs
) -> List[List[Dict[str, Any]]]:
    """
    批量搜索：一次传入多个问题 → 返回多个结果列表
    """
    return [search(q, top_k, **kwargs) for q in queries]


# 测试
if __name__ == "__main__":
    # 单条搜索
    print("===== 单条搜索测试 =====")
    res = search("time travel science fiction")
    for r in res:
        print(f"\n电影：{r['title']}")
        print(f"剧情：{r['plot'][:100]}...")

    # 批量搜索
    print("\n===== 批量搜索测试 =====")
    queries = [
        "time travel movie",
        "romantic comedy in Paris",
        "horror zombie film"
    ]
    batch_results = batch_search(queries)
    for q, res in zip(queries, batch_results):
        print(f"\n查询：{q}")
        for r in res:
            print(f" - {r['title']} (相似度：{r['similarity']})")