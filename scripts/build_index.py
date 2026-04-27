import sys
import time
import logging
import pickle
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.settings import Settings
from src.core.summarizer import BaseSummarizer, LSASummarizer, TextRankSummarizer


# 配置根日志器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 输出到控制台
    ]
)
logger = logging.getLogger(__name__)


class MovieIndexer:
    """
    电影索引构建类 (对齐版)
    功能：向量化 + 双表对齐 + 关键元数据持久化
    """
    def __init__(self,model_name: Optional[str] = None, 
                summarizer: Optional[BaseSummarizer] = None,
                algorithm: str = "textrank"):
        """
        :param model_name: Embedding 模型名称
        :param summarizer: 摘要生成器实例（如果不提供，默认使用 TextRank）
        :param algorithm: 算法标识，用于输出文件名（如 "lsa" 或 "textrank"）
        """
        self.algorithm = algorithm
        # 1. 模型名称：优先使用传入参数，否则使用配置
        self.model_name = model_name if model_name is not None else Settings.EMBEDDING_MODEL

        # 2. 硬件自动适配    
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"🛰️ 检测到计算设备: {self.device}")

        # 3. 加载Embedding 模型
        logger.info(f"🛰️ 正在加载 Embedding 模型: {self.model_name} | 设备: {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()

        # 4. 获取模型实际维度，并与配置比较（仅警告，以实际为准）
        actual_dim = self.model.get_sentence_embedding_dimension()
        if Settings.EMBEDDING_DIM is not None and Settings.EMBEDDING_DIM != actual_dim:
            logger.warning(f"配置维度 {Settings.EMBEDDING_DIM} 与模型实际维度 {actual_dim} 不一致，使用实际维度")
        self.dimension = actual_dim
        logger.info(f"向量维度: {self.dimension}")

        # 5. 初始化 FAISS 索引 (IndexFlatIP 代表内积索引，归一化后即余弦相似度)
        self.index = faiss.IndexFlatIP(self.dimension)

        # 6.数据对齐的核心容器
        self.mvid_list = []      # 向量对应的 mvid 序列 (顺序必须与 index 一致)
        self.movie_info = {}     # mvid -> 核心元数据 (Title, Year, Director 等)  

        # 初始化摘要生成器
        if summarizer is None:
            self.summarizer = TextRankSummarizer(top_n=5, pos_weight=0.3, max_len=500)
        else:
            self.summarizer = summarizer

    
    def build_from_aligned_data(self, chunks_path: Path, metadata_path: Path):
        """
        从分块表和事实表进行对齐构建
        Args:
            chunks_path: 存储分块文本的 parquet
            metadata_path: 存储核心事实库的 parquet
        """
        # 检查文件存在性
        if not chunks_path.exists():
            raise FileNotFoundError(f"分块文件不存在: {chunks_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")
        
        # 1. 读取双表
        logger.info(f"📂 正在加载数据源进行对齐...")
        df_chunks = pd.read_parquet(chunks_path) 
        df_meta = pd.read_parquet(metadata_path)

        if df_chunks.empty:
            raise ValueError("分块表为空")
        if df_meta.empty:
            raise ValueError("元数据表为空")
        
        # 统一 mvid 类型
        try:
            df_chunks['mvid'] = df_chunks['mvid'].astype(str)
            df_meta['mvid'] = df_meta['mvid'].astype(str)
        except (ValueError, TypeError) as e:
            raise ValueError(f"mvid 列无法转换为字符串类型: {e}")

        # 2. 生成向量
        logger.info(f"🧠 正在为 {len(df_chunks)} 条分块生成 Embedding...")
        texts = df_chunks['chunk_text'].tolist()
        embeddings = self.model.encode(
            texts,
            batch_size=Settings.BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        # 3. 构建索引
        self.index.add(embeddings.astype('float32'))
        self.mvid_list = df_chunks['mvid'].tolist()

        # 4. --- 数据对齐：将事实库中的核心字段注入内存缓存 ---
        logger.info("🔗 正在进行数据对齐：注入核心元数据...")
        core_fields = [
            'mvid', 'Title', 'Release Year', 'Origin/Ethnicity', 'Director', 
            'Cast_limited', 'Primary_Genre','Plot_cleaned'
        ]
        available_fields = [f for f in core_fields if f in df_meta.columns]
        if not available_fields:
            logger.warning("元数据表中没有找到任何核心字段，将只存储 mvid")
            available_fields = ['mvid']

        # 将 metadata 转换为字典，方便检索时秒回传
        for _, row in df_meta[available_fields].iterrows():
            mvid = row['mvid']
            plot_cleaned = row.get('Plot_cleaned', '')
            summary = self.summarizer.generate(plot_cleaned)
            row_dict = row.to_dict()
            row_dict['Plot_summary'] = summary
            self.movie_info[mvid] = row_dict

        logger.info(f"✅ 构建完成！索引条数: {self.index.ntotal}, 缓存电影数: {len(self.movie_info)}")

    def save(self, output_dir: Path):
        """持久化存储所有索引资产"""
        if not self.mvid_list or self.index.ntotal == 0:
            raise RuntimeError("尚未构建索引或索引为空，请先调用 build_from_aligned_data()")

        output_dir.mkdir(parents=True, exist_ok=True)

        # A. 保存 FAISS 索引文件
        index_path = output_dir / "movie_plots.index"
        faiss.write_index(self.index, str(index_path))

        # B. 保存对齐后的元数据资产
        # 根据算法生成元数据文件名
        if self.algorithm == "lsa":
            meta_name = "movie_metadata_lsa.pkl"
        else:  # textrank 或 tr
            meta_name = "movie_metadata_tr.pkl"
        meta_path = output_dir / meta_name
        payload = {
            "mvid_list": self.mvid_list,
            "movie_info": self.movie_info,
            "config": {
                "model": self.model_name,
                "dim": self.dimension,
                "timestamp": time.time(),
                "algorithm": self.algorithm
            }
        }
        with open(meta_path, "wb") as f:
            pickle.dump(payload, f)

        logger.info(f"💾 资产已对齐并保存至: {output_dir}")
        logger.info(f"   - FAISS 索引: {index_path}")
        logger.info(f"   - 电影展示元数据已保存: {meta_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="构建电影 RAG 索引（双表对齐）")
    parser.add_argument("--chunks", type=Path,
                        default=project_root / "data" / "processed" / "chunks.parquet",
                        help="分块文件路径 (parquet)")
    parser.add_argument("--metadata", type=Path,
                        default=project_root / "data" / "processed" / "metadata.parquet",
                        help="电影元数据文件路径 (parquet)")
    parser.add_argument("--output", type=Path,
                        default=project_root / "index",
                        help="输出索引目录")
    parser.add_argument("--model", type=str,
                        default=Settings.EMBEDDING_MODEL,
                        help="Embedding 模型名称")
    parser.add_argument("--algorithm", choices=["lsa", "textrank"], default="textrank",
                        help="摘要算法：lsa 或 textrank")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    # 根据命令行参数选择摘要算法
    if args.algorithm == "lsa":
        from src.core.summarizer import LSASummarizer   
        summarizer = LSASummarizer(sentence_count=3, max_len=500)
        logger.info("使用 LSA 摘要算法")
    else:  # textrank
        from src.core.summarizer import TextRankSummarizer
        summarizer = TextRankSummarizer(top_n=5, pos_weight=0.3, max_len=500)
        logger.info("使用位置加权 TextRank 摘要算法")

    try:
        indexer = MovieIndexer(model_name=args.model,
                               summarizer=summarizer,
                               algorithm=args.algorithm)
        indexer.build_from_aligned_data(args.chunks, args.metadata)
        indexer.save(args.output)
        elapsed = time.time() - start_time
        logger.info(f"🎉 恭喜！索引构建完成，总耗时: {elapsed:.1f} 秒")
    except Exception as e:
        logger.error(f"❌ 构建失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()