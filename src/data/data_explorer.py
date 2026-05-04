"""
电影数据探索工具
功能：加载原始 CSV，分析数据质量，生成 JSON 报告，并在终端打印关键信息。
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import json


class MovieDataExplorer:
    """电影数据探索器：加载数据、统计缺失、检测问题、生成报告"""

    def __init__(self, data_path: str = "data/wiki_movie_plots_deduped.csv"):
        """
        初始化探索器
        Args:
            data_path: 原始 CSV 文件路径
        """
        self.data_path = data_path
        self.df = None

    # ------------------------------------------------------------
    # 1. 数据加载
    # ------------------------------------------------------------
    def load_data(self) -> pd.DataFrame:
        """加载 CSV 数据并打印基本信息"""
        print(f"📂 正在加载数据: {self.data_path}")
        try:
            self.df = pd.read_csv(self.data_path, encoding='utf-8', sep=",", engine='python')
            print(f"✅ 数据加载成功: {self.df.shape[0]} 行, {self.df.shape[1]} 列")
            return self.df
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            raise

    # ------------------------------------------------------------
    # 2. 基本信息
    # ------------------------------------------------------------
    def basic_info(self) -> Dict[str, Any]:
        """获取数据基本信息（形状、列名、类型、内存）"""
        info = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / 1024**2, 2),
        }
        return info

    # ------------------------------------------------------------
    # 3. 缺失值分析（包括空字符串）
    # ------------------------------------------------------------
    def missing_values_analysis(self) -> Dict[str, Dict[str, int]]:
        """
        分析每列的缺失值（NaN 和空字符串）
        返回: {列名: {"nan_count": int, "empty_string_count": int, "total_missing": int, "percentage": float}}
        """
        missing = {}
        for col in self.df.columns:
            # 统计 NaN
            nan_cnt = self.df[col].isna().sum()
            # 统计空字符串（仅对 object 类型列）
            empty_cnt = 0
            if self.df[col].dtype == 'object':
                empty_cnt = (self.df[col].astype(str).str.strip() == '').sum()
            total = nan_cnt + empty_cnt
            if total > 0:
                missing[col] = {
                    "nan_count": int(nan_cnt),
                    "empty_string_count": int(empty_cnt),
                    "total_missing": int(total),
                    "percentage": round(total / len(self.df) * 100, 2)
                }
        return missing

    # ------------------------------------------------------------
    # 4. 摘要统计（发行年份、文本字段唯一值、剧情长度）
    # ------------------------------------------------------------
    def summary_statistics(self) -> Dict[str, Any]:
        """生成数值字段和文本字段的统计摘要"""
        summary = {}

        # 发行年份统计
        if 'Release Year' in self.df.columns:
            years = self.df['Release Year'].dropna()
            if not years.empty:
                summary['release_year'] = {
                    'min': int(years.min()),
                    'max': int(years.max()),
                    'mean': round(years.mean(), 2),
                    'std': round(years.std(), 2),
                    'unique_years': int(years.nunique())
                }

        # 文本字段统计（唯一值数量 + 前5高频值）
        text_fields = ['Title', 'Origin/Ethnicity', 'Director', 'Cast', 'Genre']
        for field in text_fields:
            if field in self.df.columns:
                # 过滤掉 NaN 和空字符串
                non_null = self.df[field].dropna()
                non_null = non_null[non_null.astype(str).str.strip() != '']
                if not non_null.empty:
                    summary[field] = {
                        'unique_count': int(non_null.nunique()),
                        'top_values': non_null.value_counts().head(5).to_dict()
                    }

        # 剧情字段统计（长度）
        if 'Plot' in self.df.columns:
            plots = self.df['Plot'].dropna().astype(str)
            lengths = plots.str.len()
            summary['plot'] = {
                'avg_length': round(lengths.mean(), 2),
                'min_length': int(lengths.min()),
                'max_length': int(lengths.max()),
                'missing_count': int(self.df['Plot'].isna().sum())
            }

        return summary

    # ------------------------------------------------------------
    # 5. 问题检测（年份异常、重复标题+年份、过短剧情）
    # ------------------------------------------------------------
    def detect_issues(self) -> Dict[str, Any]:
        """检测常见数据问题"""
        issues = {}

        # 5.1 异常年份（<1888 或 >2026）
        if 'Release Year' in self.df.columns:
            invalid = self.df[(self.df['Release Year'] < 1888) | (self.df['Release Year'] > 2026)]
            if not invalid.empty:
                issues['invalid_years'] = {
                    'count': len(invalid),
                    'examples': invalid[['Title', 'Release Year']].head(5).to_dict('records')
                }

        # 5.2 重复记录（基于 Title 和 Release Year 的组合）
        if 'Title' in self.df.columns and 'Release Year' in self.df.columns:
            dup = self.df.duplicated(subset=['Title', 'Release Year'], keep=False)
            dup_df = self.df[dup]
            if not dup_df.empty:
                issues['duplicate_title_year'] = {
                    'count': len(dup_df),
                    'examples': dup_df[['Title', 'Release Year']].head(5).to_dict('records')
                }

        # 5.3 过短剧情（长度 < 20 字符）
        if 'Plot' in self.df.columns:
            short = self.df[self.df['Plot'].astype(str).str.len() < 20]
            if not short.empty:
                issues['short_plots'] = {
                    'count': len(short),
                    'examples': short[['Title', 'Plot']].head(5).to_dict('records')
                }

        return issues

    # ------------------------------------------------------------
    # 6. 动态生成清洗建议（基于实际检测到的问题）
    # ------------------------------------------------------------
    def generate_recommendations(self) -> Dict[str, Any]:
        """根据缺失值、异常值等动态生成建议"""
        rec = {"cleaning_steps": [], "search_preparation": []}

        # 缺失值建议
        missing = self.missing_values_analysis()
        if missing:
            fields = list(missing.keys())
            rec["cleaning_steps"].append(f"处理缺失值：{', '.join(fields)} 字段存在缺失或空字符串")

        # 异常年份建议
        issues = self.detect_issues()
        if issues.get('invalid_years'):
            rec["cleaning_steps"].append(f"验证年份：发现 {issues['invalid_years']['count']} 个异常发行年份")

        # 重复记录建议
        if issues.get('duplicate_title_year'):
            rec["cleaning_steps"].append(f"去重处理：发现 {issues['duplicate_title_year']['count']} 条标题+年份重复记录")

        # 过短剧情建议
        if issues.get('short_plots'):
            rec["cleaning_steps"].append(f"文本清理：发现 {issues['short_plots']['count']} 条过短剧情（<20字符）")

        # 格式标准化建议（基于字段实际唯一值数量判断）
        if 'Genre' in self.df.columns:
            genres = self.df['Genre'].dropna().astype(str).str.lower()
            if not genres.empty and genres.nunique() > 20:
                rec["cleaning_steps"].append("标准化格式：Genre 字段值过多，建议映射到标准类型集合")

        if 'Origin/Ethnicity' in self.df.columns:
            origins = self.df['Origin/Ethnicity'].dropna().astype(str)
            if not origins.empty and origins.nunique() > 10:
                rec["cleaning_steps"].append("标准化格式：Origin/Ethnicity 字段存在多种写法，建议统一")

        if not rec["cleaning_steps"]:
            rec["cleaning_steps"].append("数据质量良好，无需额外清洗")

        # 搜索准备建议（通用）
        rec["search_preparation"] = [
            "创建搜索索引字段：结合 Title、Genre、Plot",
            "提取关键词：从 Plot 中提取关键名词和主题",
            "构建向量：为语义搜索准备文本向量化"
        ]
        return rec

    # ------------------------------------------------------------
    # 7. 类型转换辅助（JSON 序列化）
    # ------------------------------------------------------------
    def convert_numpy_to_python(self, obj):
        """递归将 numpy 类型转换为 Python 原生类型，以便 JSON 序列化"""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: self.convert_numpy_to_python(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self.convert_numpy_to_python(i) for i in obj]
        return obj

    # ------------------------------------------------------------
    # 8. 保存 JSON 报告
    # ------------------------------------------------------------
    def save_report(self, output_path: str = "data/data_quality_report.json"):
        """将分析结果保存为 JSON 文件"""
        report = {
            "basic_info": self.basic_info(),
            "missing_values": self.missing_values_analysis(),
            "summary_statistics": self.summary_statistics(),
            "issues": self.detect_issues(),
            "recommendations": self.generate_recommendations()
        }
        report = self.convert_numpy_to_python(report)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📄 数据质量报告已保存到: {output_path}")
        return report


# ------------------------------------------------------------
# 主函数：依次执行探索步骤并打印输出
# ------------------------------------------------------------
def main():
    print("=" * 60)
    print("🎬 电影数据探索工具")
    print("=" * 60)

    # 1. 创建探索器并加载数据
    explorer = MovieDataExplorer()
    df = explorer.load_data()

    # 2. 基本信息
    print("\n=== 1. 数据基本信息 ===")
    info = explorer.basic_info()
    print(f"数据形状: {info['shape']}")
    print(f"列名: {info['columns']}")
    print(f"内存占用: {info['memory_usage_mb']:.2f} MB")

    # 3. 缺失值分析
    print("\n=== 2. 缺失值分析（含空字符串） ===")
    missing = explorer.missing_values_analysis()
    if missing:
        for col, stat in missing.items():
            print(f"  {col}: NaN={stat['nan_count']}, 空字符串={stat['empty_string_count']} (合计 {stat['total_missing']}, {stat['percentage']}%)")
    else:
        print("  无缺失值")

    # 4. 摘要统计
    print("\n=== 3. 数据摘要统计 ===")
    summary = explorer.summary_statistics()
    if 'release_year' in summary:
        y = summary['release_year']
        print(f"  发行年份: {y['min']} ~ {y['max']} (平均 {y['mean']}, 标准差 {y['std']})")
    if 'Title' in summary:
        print(f"  独特电影标题数: {summary['Title']['unique_count']}")
    if 'plot' in summary:
        p = summary['plot']
        print(f"  剧情长度: 最短 {p['min_length']} 字符, 最长 {p['max_length']} 字符, 平均 {p['avg_length']:.1f} 字符")

    # 5. 问题检测
    print("\n=== 4. 检测到的问题 ===")
    issues = explorer.detect_issues()
    if issues:
        for name, issue in issues.items():
            print(f"  {name}: {issue['count']} 个问题")
            if 'examples' in issue and issue['examples']:
                print(f"    示例: {issue['examples'][:2]}")
    else:
        print("  未发现明显问题")

    # 6. 清洗建议
    print("\n=== 5. 数据清洗建议 ===")
    rec = explorer.generate_recommendations()
    for step in rec.get("cleaning_steps", []):
        print(f"  • {step}")

    # 7. 保存报告
    print("\n=== 6. 保存报告 ===")
    explorer.save_report()

    print("\n✅ 数据探索完成！")


if __name__ == "__main__":
    main()