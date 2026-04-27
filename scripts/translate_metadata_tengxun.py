"""
腾讯云翻译 API - 随机采样翻译电影剧情摘要
- 免费额度：500万字符/月
- 每翻译一条立即保存，支持断点续传
- 随机采样指定数量（默认500部）
"""

import pickle
import random
import time
from pathlib import Path
import os
from dotenv import load_dotenv
import json

from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.tmt.v20180321 import tmt_client, models

import sys
import io
# 强制标准输出使用 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ==================== 配置区域 ====================
load_dotenv()
TENCENT_SECRET_ID = os.getenv("TENCENT_SECRET_ID")
TENCENT_SECRET_KEY = os.getenv("TENCENT_SECRET_KEY")
INPUT_PATH = Path("index/movie_metadata_tr.pkl")
OUTPUT_PATH = Path("index/movie_metadata_tr_zh2000.pkl")
SAMPLE_SIZE = 2000
# =================================================

# 初始化腾讯云客户端（全局，避免重复初始化）
cred = credential.Credential(TENCENT_SECRET_ID, TENCENT_SECRET_KEY)
httpProfile = HttpProfile()
httpProfile.endpoint = "tmt.tencentcloudapi.com"
clientProfile = ClientProfile()
clientProfile.httpProfile = httpProfile
client = tmt_client.TmtClient(cred, "ap-beijing", clientProfile)

def tencent_translate(text, source="en", target="zh"):
    """腾讯云翻译"""
    if not text or len(text.strip()) < 10:
        return text
    req = models.TextTranslateRequest()
    # 限制长度（腾讯云限制 6000 字符）
    q = text[:6000]
    params = {
        "SourceText": q,
        "Source": source,
        "Target": target,
        "ProjectId": 0
    }
     # 使用 json.dumps 生成标准 JSON 字符串
    req.from_json_string(json.dumps(params, ensure_ascii=False))
    try:
        resp = client.TextTranslate(req)
        return resp.TargetText
    except Exception as e:
        print(f"腾讯云翻译失败: {e}")
        return text

def main():
    # 1. 加载原始元数据
    print("加载原始元数据...")
    with open(INPUT_PATH, 'rb') as f:
        data = pickle.load(f)
    movie_info = data["movie_info"]
    total_movies = len(movie_info)
    print(f"总电影数: {total_movies}")

    # 2. 加载已有输出（断点续传）
    translated_mvids = set()
    if OUTPUT_PATH.exists():
        print(f"发现已有输出文件 {OUTPUT_PATH}，加载已翻译的记录...")
        with open(OUTPUT_PATH, 'rb') as f:
            existing = pickle.load(f)
        for mvid, info in existing["movie_info"].items():
            if info.get("Plot_summary_zh"):
                movie_info[mvid]["Plot_summary_zh"] = info["Plot_summary_zh"]
                translated_mvids.add(mvid)
        print(f"已加载 {len(translated_mvids)} 条已翻译记录")

    # 3. 确定未翻译的电影列表
    untranslated = [(mvid, info) for mvid, info in movie_info.items()
                    if mvid not in translated_mvids and info.get('Plot_summary') and len(info['Plot_summary'].strip()) >= 10]
    untranslated_count = len(untranslated)
    print(f"未翻译电影数: {untranslated_count}")

    # 4. 随机采样
    sample_count = min(SAMPLE_SIZE, untranslated_count)
    if sample_count == 0:
        print("没有需要翻译的电影，退出。")
        return
    sample_items = random.sample(untranslated, sample_count)
    print(f"本次计划翻译 {sample_count} 部电影（随机采样）")

    # 5. 逐条翻译并立即保存
    new_translated = 0
    for idx, (mvid, info) in enumerate(sample_items, 1):
        if info.get('Plot_summary_zh'):
            continue
        summary = info.get('Plot_summary', '')
        if not summary:
            continue
        print(f"[{idx}/{sample_count}] 翻译: {mvid} - {summary[:50]}...")
        zh = tencent_translate(summary)
        if zh and zh != summary:
            info['Plot_summary_zh'] = zh
            new_translated += 1
            print(f"  翻译成功")
        else:
            print(f"  翻译失败，保留原文")
        # 立即保存（原子写入：先写临时文件再替换）
        temp_path = OUTPUT_PATH.with_suffix('.tmp')
        with open(temp_path, 'wb') as f:
            pickle.dump(data, f)
        temp_path.replace(OUTPUT_PATH)
        print(f"  累计新增: {new_translated}，总翻译数: {len(translated_mvids) + new_translated}")
        time.sleep(0.2)  # 控制频率

    print(f"翻译完成！本次新增 {new_translated} 条，总翻译数: {len(translated_mvids) + new_translated}")
    print(f"输出文件: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()