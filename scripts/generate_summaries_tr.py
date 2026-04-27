"""
独立生成 extRank 摘要的元数据文件（从 metadata.parquet 直接生成）

"""

import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

project_root = Path(__file__).parent.parent  
sys.path.append(str(project_root))
from src.core.summarizer import TextRankSummarizer

def main():
    parser = argparse.ArgumentParser(description="生成 TextRank 摘要的元数据文件")
    parser.add_argument("--input", type=Path, required=True, help="输入的 pickle 文件路径（如 movie_metadata.pkl）")
    parser.add_argument("--output", type=Path, required=True, help="输出的 pickle 文件路径（如 movie_metadata_tr.pkl）")
    parser.add_argument("--top_n", type=int, default=5, help="摘要句子数（默认5）")
    parser.add_argument("--pos_weight", type=float, default=0.3, help="位置权重（默认0.3）")
    parser.add_argument("--max_len", type=int, default=800, help="摘要最大字符数（默认500）")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"错误：输入文件 {args.input} 不存在")
        return

    print(f"加载 {args.input} ...")
    with open(args.input, "rb") as f:
        data = pickle.load(f)

    movie_info = data["movie_info"]
    print(f"总电影数: {len(movie_info)}")

    # 初始化 TextRank 摘要器
    summarizer = TextRankSummarizer(top_n=args.top_n, pos_weight=args.pos_weight, max_len=args.max_len)

    # 遍历所有电影，生成新摘要
    for mvid, info in tqdm(movie_info.items(), desc="生成 TextRank 摘要"):
        plot_cleaned = info.get("Plot_cleaned", "")
        if plot_cleaned and len(plot_cleaned.strip()) > 50:
            new_summary = summarizer.generate(plot_cleaned)
            # 直接覆盖 Plot_summary 字段（如果你想保留原字段，可以改成 info["Plot_summary_tr"] = new_summary）
            info["Plot_summary"] = new_summary
        else:
            # 如果没有有效剧情，保留原摘要（或置空）
            if "Plot_summary" not in info:
                info["Plot_summary"] = ""

    # 保存新文件
    print(f"保存到 {args.output} ...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(data, f)

    print("完成！")

if __name__ == "__main__":
    main()