import pandas as pd
import sys
from pathlib import Path
import argparse
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.smart_text_splitter import SmartTextSplitter


def process_movies_csv(
    input_csv_path,
    chunk_size=800,
    chunk_overlap=120,
    short_doc_threshold=200,
    generate_parent_chunks=False,
    parent_size=4000,
    sample_size=None
):
    """
    电影剧情分块脚本
    输入：清洗后的 CSV（必须包含字段：Plot_cleaned 用于分块）
    输出：分块后的 CSV，每行一个 chunk，包含原始所有字段 + 分块专用字段
    """
    input_path = Path(input_csv_path)
    output_path = input_path.parent / "movie_chunks.csv"

    # 读取数据
    df = pd.read_csv(input_path)
    print(f"✅ 成功读取数据，共 {len(df)} 行，字段数：{len(df.columns)}")

    if sample_size and sample_size > 0:
        df = df.head(sample_size)
        print(f"🧪 测试模式：仅处理前 {sample_size} 部电影")

    # 必须校验 Plot_cleaned 字段存在
    if "Plot_cleaned" not in df.columns:
        raise ValueError("输入CSV必须包含 'Plot_cleaned' 字段（清洗后剧情）")

    # 可选：如果不存在 mvid 则自动生成（但清洗脚本应该已生成）
    if "mvid" not in df.columns:
        print("⚠️ 未找到 'mvid' 列，将根据 Title 和 Release Year 自动生成")
        df["mvid"] = df.apply(
            lambda row: f"{row['Title']}_{row['Release Year']}".replace(" ", "_"),
            axis=1
        )

    # 初始化分割器
    splitter = SmartTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_fragment_size=40
    )

    records = []

    # 逐行处理（带进度条）
    for _, row in tqdm(df.iterrows(), total=len(df), desc="分块进度"):
        plot_cleaned = row["Plot_cleaned"]
        if pd.isna(plot_cleaned) or len(str(plot_cleaned).strip()) == 0:
            continue

        plot = str(plot_cleaned).strip()
        original_len = len(plot)

        # 短文档不分块
        if original_len < short_doc_threshold:
            chunks = [plot]
            parent_chunks = [plot] if generate_parent_chunks else None
        else:
            chunks = splitter.split(
                text=plot,
                is_whole_document=True,
                short_threshold=short_doc_threshold
            )
            parent_chunks = None
            if generate_parent_chunks:
                parent_chunks = splitter.create_parent_chunks(
                    text=plot,
                    parent_size=parent_size
                )

        # 将整行数据转为字典（所有原始字段）
        base_record = row.to_dict()

        for idx, chunk in enumerate(chunks):
            # 复制所有原始字段
            record = base_record.copy()
            # 添加分块专用字段
            record["chunk_index"] = idx
            record["chunk_text"] = chunk

            if generate_parent_chunks and parent_chunks:
                # 简单匹配父块（第一个包含 chunk 的父块）
                matched_parent = None
                for p in parent_chunks:
                    if chunk in p:
                        matched_parent = p
                        break
                if not matched_parent and parent_chunks:
                    matched_parent = parent_chunks[0]
                record["parent_text"] = matched_parent

            records.append(record)

    # 输出结果
    result_df = pd.DataFrame(records)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # 统计信息
    print("\n" + "=" * 80)
    print("🎉 分块完成！")
    print(f"📂 输出文件：{output_path}")
    print(f"📦 总分块数：{len(records)}")
    print(f"📊 平均块长度：{result_df['chunk_text'].str.len().mean():.0f} 字符")
    print(f"📈 最大块长度：{result_df['chunk_text'].str.len().max()} 字符")
    print(f"📉 最小块长度：{result_df['chunk_text'].str.len().min()} 字符")
    print(f"📋 输出字段数：{len(result_df.columns)}（原始字段 + 分块字段）")
    if generate_parent_chunks:
        print(f"📚 父块已生成（大小 ≤ {parent_size} 字符）")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="电影剧情分块 - 保留所有原始字段")
    parser.add_argument("--input", required=True, help="清洗后的电影CSV文件路径")
    parser.add_argument("--sample", type=int, help="测试模式：只处理前N部电影")
    parser.add_argument("--chunk_size", type=int, default=800, help="子块最大字符数")
    parser.add_argument("--chunk_overlap", type=int, default=120, help="子块重叠字符数")
    parser.add_argument("--short_doc_threshold", type=int, default=200, help="短文档阈值（字符）")
    parser.add_argument("--generate_parent_chunks", action="store_true", help="是否生成父块")
    parser.add_argument("--parent_size", type=int, default=4000, help="父块最大字符数")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_movies_csv(
        input_csv_path=args.input,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        short_doc_threshold=args.short_doc_threshold,
        generate_parent_chunks=args.generate_parent_chunks,
        parent_size=args.parent_size,
        sample_size=args.sample
    )