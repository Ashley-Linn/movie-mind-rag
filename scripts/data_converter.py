import pandas as pd
from pathlib import Path

def convert_csv_to_parquet():
    # 1. 路径设置
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义输入文件路径
    cleaned_csv = data_dir / "movies_cleaned.csv"
    chunks_csv = data_dir / "movie_chunks.csv"
    
    print("🚀 开始数据转换与瘦身...")

    # --- 任务 A: 生成 metadata.parquet (电影事实库) ---
    if cleaned_csv.exists():
        print(f"📦 正在处理主表: {cleaned_csv.name}")
        df_meta = pd.read_csv(cleaned_csv)
        
        # 挑选展示用的核心字段
        metadata_cols = [
            'mvid', 'Title', 'Title_normalized', 'Release Year', 
            'Director', 'Cast_limited', 'Primary_Genre', 'Plot_cleaned',
            'Origin/Ethnicity'  
        ]
        
        # 确保这些列都在，缺失的列会引发错误（可以根据需要处理）
        available_cols = [c for c in metadata_cols if c in df_meta.columns]
        missing_cols = set(metadata_cols) - set(available_cols)
        if missing_cols:
            print(f"⚠️ 警告: 以下列在 CSV 中不存在，将被忽略: {missing_cols}")
        if not available_cols:
            raise ValueError("没有可用的列，请检查 CSV 文件")
        
        metadata = df_meta[available_cols].drop_duplicates(subset=['mvid'])
        
        meta_output = output_dir / "metadata.parquet"
        metadata.to_parquet(meta_output, index=False)
        print(f"✅ 事实库保存成功: {meta_output} ({len(metadata)} 部电影)")
    else:
        print(f"❌ 未找到文件: {cleaned_csv}")

    # --- 任务 B: 生成 chunks.parquet (向量索引素材) ---
    if chunks_csv.exists():
        print(f"📦 正在处理分块表: {chunks_csv.name}")
        # 分块表只需要这些列（节省内存）
        chunk_cols = ['mvid', 'chunk_index', 'chunk_text']
        # 如果存在 parent_text 列，也保留（可选）
        df_chunks = pd.read_csv(chunks_csv, usecols=lambda x: x in chunk_cols or x == 'parent_text')
        
        chunks_output = output_dir / "chunks.parquet"
        df_chunks.to_parquet(chunks_output, index=False)
        print(f"✅ 索引库保存成功: {chunks_output} ({len(df_chunks)} 个分块)")
    else:
        print(f"❌ 未找到文件: {chunks_csv}")

    print("\n✨ 所有任务完成！现在你的 data/processed 目录已准备就绪。")

if __name__ == "__main__":
    convert_csv_to_parquet()