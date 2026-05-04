"""
电影数据清洗工具（最终修正版）
功能：
1. 处理缺失值（NaN 和空字符串，以及各种空值变体）
2. 标准化格式（类型、产地、导演、演员）
3. 生成唯一电影 ID（mvid）
4. 移除重复记录（基于标题+年份）
5. 创建搜索辅助字段
"""

import pandas as pd
import re
import numpy as np
from collections import defaultdict
import logging

# 配置根日志器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # 输出到控制台
    ]
)
logger = logging.getLogger(__name__)

class MovieDataCleaner:
    def __init__(self, df: pd.DataFrame):
        """初始化清洗器，保留原始数据副本"""
        self.df = df.copy()
        self.cleaned_df = None

    # ---------------------- 辅助函数 ----------------------
    def _fill_missing(self, col_name: str, fill_value: str = 'Unknown') -> None:
        """
        通用缺失值填充：将 NaN、None、空字符串、各种空值变体统一替换为 fill_value
        """
        if col_name not in self.df.columns:
            return

        # 定义空值检测函数
        def is_empty(x):
            if pd.isna(x):
                return True
            s = str(x).strip()
            # 匹配各种常见的空值表示
            if s == '' or s.lower() in ('nan', 'none', 'null', 'na', 'n/a', 'unknown', '?', '-'):
                return True
            return False

        # 应用检测并替换
        self.df[col_name] = self.df[col_name].apply(lambda x: np.nan if is_empty(x) else x)
        self.df[col_name] = self.df[col_name].fillna(fill_value)
        self.df[col_name] = self.df[col_name].astype(str).str.strip()

    # ---------------------- 字段清洗方法 ----------------------
    def clean_release_year(self) -> 'MovieDataCleaner':
        """清洗发行年份：转数值、剔除异常值、填充中位数"""
        logger.info("清洗发行年份...")
        self.df['Release Year'] = pd.to_numeric(self.df['Release Year'], errors='coerce')
        current_year = 2026
        self.df['Release Year'] = self.df['Release Year'].apply(
            lambda x: x if (1888 <= x <= current_year) else np.nan
        )
        median_year = self.df['Release Year'].median()
        # 避免全为 NaN 的情况
        if pd.isna(median_year):
            median_year = 1950
        self.df['Release Year'] = self.df['Release Year'].fillna(median_year).astype(int)
        logger.info(f"发行年份范围: {self.df['Release Year'].min()} - {self.df['Release Year'].max()}")
        return self

    def clean_title(self) -> 'MovieDataCleaner':
        """清洗标题：去除首尾空格、清理特殊字符、生成标准化小写版本"""
        logger.info("清洗电影标题...")
        self.df['Title'] = self.df['Title'].astype(str).str.strip()
        # 保留字母、数字、空格、常见标点
        self.df['Title'] = self.df['Title'].apply(
            lambda x: re.sub(r'[^\w\s\-.,:;!?\'"()&]', '', x)
        )
        # 标准化标题（小写，用于去重）
        self.df['Title_normalized'] = self.df['Title'].str.lower().str.strip()
        return self

    def clean_origin_ethnicity(self) -> 'MovieDataCleaner':
        """清洗产地/民族：标准化常见值，统一首字母大写"""
        logger.info("清洗产地/民族...")
        if 'Origin/Ethnicity' in self.df.columns:
            # 先填充缺失
            self._fill_missing('Origin/Ethnicity', 'Unknown')
            # 标准化映射
            origin_mapping = {
                'american': 'American', 'british': 'British', 'canadian': 'Canadian',
                'australian': 'Australian', 'indian': 'Indian', 'chinese': 'Chinese',
                'japanese': 'Japanese', 'korean': 'Korean', 'french': 'French',
                'german': 'German', 'italian': 'Italian', 'spanish': 'Spanish',
                'russian': 'Russian', 'unknown': 'Unknown'
            }
            # 统一小写后映射
            self.df['Origin/Ethnicity'] = self.df['Origin/Ethnicity'].str.lower()
            self.df['Origin/Ethnicity'] = self.df['Origin/Ethnicity'].map(
                lambda x: origin_mapping.get(x, x.title())
            )
        return self

    def clean_director(self) -> 'MovieDataCleaner':
        """清洗导演信息：填充缺失，多个导演用分号分隔"""
        logger.info("清洗导演信息...")
        if 'Director' in self.df.columns:
            self._fill_missing('Director', 'Unknown')
            # 处理多个导演（用分号分隔）
            self.df['Director'] = self.df['Director'].apply(
                lambda x: '; '.join([d.strip() for d in x.split(';')]) if x != 'Unknown' else 'Unknown'
            )
        return self

    def clean_cast(self) -> 'MovieDataCleaner':
        """清洗演员阵容：填充缺失，限制前5位演员，生成 Cast_limited"""
        logger.info("清洗演员阵容...")
        if 'Cast' in self.df.columns:
            self._fill_missing('Cast', 'Unknown')
            # 限制前5位演员
            self.df['Cast_limited'] = self.df['Cast'].apply(
                lambda x: ', '.join([actor.strip() for actor in x.split(',')[:5]]) if x != 'Unknown' else 'Unknown'
            )
        return self

    def clean_genre(self) -> 'MovieDataCleaner':
        """清洗电影类型：标准化映射，提取主要类型"""
        logger.info("清洗电影类型...")
        if 'Genre' in self.df.columns:
            # 先填充缺失为 'unknown'（临时占位）
            self._fill_missing('Genre', 'unknown')
            # 确保没有 NaN（上面已经填充，但以防万一）
            self.df['Genre'] = self.df['Genre'].fillna('unknown')
            # 标准化映射表
            genre_mapping = {
                'drama': 'Drama', 'comedy': 'Comedy', 'action': 'Action',
                'romance': 'Romance', 'horror': 'Horror', 'sci-fi': 'Science Fiction',
                'science fiction': 'Science Fiction', 'thriller': 'Thriller',
                'adventure': 'Adventure', 'fantasy': 'Fantasy', 'mystery': 'Mystery',
                'crime': 'Crime', 'animation': 'Animation', 'family': 'Family',
                'musical': 'Musical', 'biography': 'Biography', 'history': 'History',
                'war': 'War', 'western': 'Western', 'film-noir': 'Film Noir',
                'unknown': 'Unknown'
            }
            # 统一小写后映射
            self.df['Genre'] = self.df['Genre'].str.lower()
            self.df['Genre'] = self.df['Genre'].apply(
                lambda x: genre_mapping.get(x, x.title())
            )
            # 提取主要类型（取第一个逗号前的部分）
            self.df['Primary_Genre'] = self.df['Genre'].apply(
                lambda x: x.split(',')[0].strip() if ',' in x else x
            )
        return self

    def clean_plot(self) -> 'MovieDataCleaner':
        """清洗剧情：缺失用标题填充，清理特殊字符，计算长度"""
        logger.info("清洗电影情节...")
        if 'Plot' in self.df.columns:
            # 缺失值用标题替代
            self.df['Plot'] = self.df.apply(
                lambda row: row['Title'] if pd.isna(row['Plot']) or str(row['Plot']).strip() == '' else row['Plot'],
                axis=1
            )
            self.df['Plot'] = self.df['Plot'].astype(str).str.strip()
            # 保留常用标点，删除特殊字符
            self.df['Plot_cleaned'] = self.df['Plot'].apply(
                lambda x: re.sub(r'[^\w\s\.,!?\'"-]', '', x)
            )
            self.df['Plot_Length'] = self.df['Plot_cleaned'].apply(len)
            # 记录过短情节
            min_plot_length = 50
            short_mask = self.df['Plot_Length'] < min_plot_length
            if short_mask.any():
                logger.warning(f"发现 {short_mask.sum()} 个过短情节（<{min_plot_length}字符）")
        return self

    def clean_wiki_page(self) -> 'MovieDataCleaner':
        """清洗 Wiki Page：填充空值，清理特殊字符"""
        logger.info("清洗 Wiki Page...")
        if 'Wiki Page' in self.df.columns:
            self._fill_missing('Wiki Page', '')
            self.df['Wiki Page'] = self.df['Wiki Page'].apply(
                lambda x: re.sub(r'[^\w\s:/\.\-_]', '', x) if x else ''
            )
        return self

    # ---------------------- 后处理 ----------------------
    def generate_mvid(self) -> 'MovieDataCleaner':
        """生成唯一电影ID：格式为 Title_Year，重复时加 _v2, _v3 等后缀"""
        logger.info("生成唯一电影ID...")
        base_ids = self.df['Title'].astype(str) + "_" + self.df['Release Year'].astype(str)
        counter = defaultdict(int)
        final_ids = []
        for base in base_ids:
            counter[base] += 1
            if counter[base] == 1:
                final_ids.append(base)
            else:
                final_ids.append(f"{base}_v{counter[base]}")
        self.df['mvid'] = final_ids
        logger.info(f"生成 {self.df['mvid'].nunique()} 个唯一ID")
        return self

    def remove_duplicates(self) -> 'MovieDataCleaner':
        """移除重复记录：基于标准化标题和年份，保留第一条"""
        logger.info("移除重复记录...")
        before = len(self.df)
        dup_mask = self.df.duplicated(subset=['Title_normalized', 'Release Year'], keep='first')
        if dup_mask.any():
            logger.info(f"发现 {dup_mask.sum()} 条重复行（共 {before - dup_mask.sum()} 组重复）")
            self.df = self.df[~dup_mask].reset_index(drop=True)
            after = len(self.df)
            logger.info(f"移除了 {before - after} 个重复记录")
        return self

    def create_search_fields(self) -> 'MovieDataCleaner':
        """创建搜索文本和关键词字段，用于混合检索"""
        logger.info("创建搜索字段...")
        self.df['Search_Text'] = self.df.apply(
            lambda row: ' '.join([
                str(row['Title']),
                str(row.get('Primary_Genre', '')),
                str(row.get('Origin/Ethnicity', '')),
                str(row.get('Director', '')).split(';')[0],  # 只取第一个导演
                str(row.get('Cast_limited', ''))
            ]).strip(),
            axis=1
        )
        self.df['Keywords'] = self.df['Search_Text']  # 可与 Search_Text 相同，也可简化
        return self

    # ---------------------- 主流程 ----------------------
    def clean_all(self) -> pd.DataFrame:
        """执行所有清洗步骤"""
        logger.info("开始完整数据清洗流程...")
        (self.clean_release_year()
         .clean_title()
         .clean_origin_ethnicity()
         .clean_director()
         .clean_cast()
         .clean_genre()
         .clean_plot()
         .clean_wiki_page()
         .remove_duplicates()
         .create_search_fields()
         .generate_mvid())
        self.cleaned_df = self.df
        logger.info(f"清洗完成: {self.cleaned_df.shape[0]} 条记录")
        return self.cleaned_df

    def save_cleaned_data(self, output_path: str = "data/movies_cleaned.csv") -> str:
        """保存清洗后的数据"""
        if self.cleaned_df is not None:
            self.cleaned_df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"清洗后数据已保存到: {output_path}")
        else:
            logger.warning("无数据可保存，请先运行 clean_all()")
        return output_path


# ---------------------- 使用示例 ----------------------
def main():
    df = pd.read_csv("data/wiki_movie_plots_deduped.csv", encoding='utf-8', sep=",", engine='python')
    cleaner = MovieDataCleaner(df)
    cleaned_df = cleaner.clean_all()
    cleaner.save_cleaned_data()
    print("\n清洗结果摘要:")
    print(f"原始行数: {df.shape[0]}, 清洗后: {cleaned_df.shape[0]}")
    print(cleaned_df[['mvid', 'Title', 'Release Year', 'Primary_Genre', 'Plot_Length']].head())


if __name__ == "__main__":
    main()