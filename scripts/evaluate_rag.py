"""
使用 RAGAS 0.2.2 全面评估 RAG 系统
包含检索指标：context_precision, context_recall
包含生成指标：faithfulness, answer_relevancy
需要 CSV 文件包含列：question, ground_truth
使用本地 embedding 模型（与检索器相同），避免 OpenAI API key 错误
用法：python evaluation/scripts/eval_ragas022.py
"""
import sys
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.core.retriever import search
from src.core.generator import ask

# RAGAS 相关导入
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# LLM 和 Embeddings 包装
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

from config.settings import Settings

# ---------- 1. 配置评估 LLM（裁判）----------
evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model=Settings.LLM_MODEL,
        openai_api_key=Settings.DEEPSEEK_API_KEY,
        openai_api_base=Settings.DEEPSEEK_API_BASE,
        temperature=0
    )
)

# ---------- 2. 配置评估 Embeddings（使用已有的本地模型）----------
# 注意：与检索器使用的模型一致，避免额外下载
embedding_model = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_model)

# ---------- 3. 路径配置 ----------
DATA_DIR = project_root / "evaluation" / "data"
RESULT_DIR = project_root / "evaluation" / "results"
CSV_FILE = DATA_DIR / "golden_manual.csv"   # 请改为你的测试集文件名

def main():
    if not CSV_FILE.exists():
        print(f"错误：文件不存在 {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE, encoding='utf-8')
    if 'question' not in df.columns or 'ground_truth' not in df.columns:
        print("错误：CSV 必须包含 'question' 和 'ground_truth' 列")
        return

    samples = []
    for idx, row in df.iterrows():
        question = row['question']
        print(f"处理 {idx+1}/{len(df)}: {question[:40]}...")

        # 检索
        retrieved = search(question, top_k=3)
        # contexts 是字符串列表（来自检索器的 plot 字段）
        contexts = [r.get('plot', '') for r in retrieved]
        # 生成答案
        answer = ask(question, top_k=3, mode="strict")

        samples.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": row['ground_truth']
        })

    dataset = Dataset.from_list(samples)

    # ---------- 4. 定义所有指标 ----------
    metrics = [
        context_precision,   # 检索精度
        context_recall,      # 检索召回
        faithfulness,        # 答案忠实度
        answer_relevancy,    # 答案相关性
    ]

    # ---------- 5. 执行评估（显式传入 llm 和 embeddings）----------
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    # ---------- 6. 输出平均分 ----------
    print("\n========== RAGAS 0.2.2 完整评估结果 ==========")
    for metric in metrics:
        metric_name = metric.name
        scores = [s for s in result[metric_name] if s is not None]
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"{metric_name}: {avg:.4f}")
    print("=" * 50)

    # ---------- 7. 保存详细结果（每个样本的分数）----------
    detail_df = pd.DataFrame({
        'question': [s['question'] for s in samples],
        'generated_answer': [s['answer'] for s in samples],
        'ground_truth': [s['ground_truth'] for s in samples],
        'context_precision': result['context_precision'],
        'context_recall': result['context_recall'],
        'faithfulness': result['faithfulness'],
        'answer_relevancy': result['answer_relevancy'],
    })
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULT_DIR / "ragas_complete_eval.csv"
    detail_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存至 {output_path}")

if __name__ == "__main__":
    main()