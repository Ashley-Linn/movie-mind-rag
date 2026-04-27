"""应用配置设置"""
from pathlib import Path
import os
from dotenv import load_dotenv


# 加载 .env 文件
load_dotenv()

class Settings:
    """应用配置类 """

    # --- 基础路径配置 ---
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    PROCESSED_DIR: Path = DATA_DIR / "processed"  # 专门存放清洗后的 parquet
    INDEX_DIR: Path = BASE_DIR / "index"          # 专门存放 FAISS 索引

    # --- 数据文件路径 ---
    # 核心：分块后的事实库（用于构建索引的文本源）
    CHUNKS_PATH: Path = PROCESSED_DIR / "chunks.parquet"
    # 核心：电影事实库（检索后反查的档案库）
    METADATA_PATH: Path = PROCESSED_DIR / "metadata.parquet"

    # --- 向量化模型配置 ---
    # 建议使用多语言模型，这样支持中文提问搜英文剧情
    EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"  
    EMBEDDING_DIM: int = 384  

    # --- llm配置 ---
    LLM_MODEL: str = "deepseek-chat"
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE")  
    
    # --- 搜索配置 ---
    SIMILARITY_THRESHOLD: float = 0.5  # 相似度阈值，建议根据测试调整
    TOP_K: int = 10                      # 默认检索出的最相关分块数

    # --- 性能配置 ---
    # 4核8线程 CPU 环境下，64 是最稳健的批次大小
    BATCH_SIZE: int = 128  
     

    @classmethod
    def ensure_dirs(cls) -> None:
        """确保 RAG 运行所需的目录存在"""
        directories = [
            cls.DATA_DIR,
            cls.PROCESSED_DIR,
            cls.INDEX_DIR,
            cls.BASE_DIR / "logs"
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# 初始化
settings = Settings()
settings.ensure_dirs()