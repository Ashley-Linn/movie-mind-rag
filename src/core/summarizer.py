# summarizer.py
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from nltk.tokenize import sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# 确保 nltk 的 punkt 已下载
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class BaseSummarizer:
    """摘要生成器基类"""
    def generate(self, text: str) -> str:
        raise NotImplementedError

class LSASummarizer(BaseSummarizer):
    """LSA 抽取式摘要       
        - 短文本 (<500): 返回原文
        - 中文本 (500-10000): 用 LSA 抽取 sentence_count 句
        - 长文本 (>10000): 分段抽取（每段 2000 字符），每段抽 1-2 句，合并后取前 6 句
        """
    def __init__(self, sentence_count: int = 3, max_len: int = 500):
        self.sentence_count = sentence_count
        self.max_len = max_len

    def generate(self, text: str) -> str:
        if not text or len(text.strip()) < 50:
            return text or ""

        # 短文本直接返回
        if len(text) <= 500:
            return text

        # 中等长度文本：LSA 抽取
        if len(text) <= 10000:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LsaSummarizer()
            summary_sentences = summarizer(parser.document, self.sentence_count)
            summary = " ".join(str(s) for s in summary_sentences)
            if len(summary) > self.max_len:
                summary = summary[:self.max_len].rsplit(' ', 1)[0] + "..."
            return summary

        # 超长文本：分段 LSA
        chunk_size = 2000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        extracted = []
        for chunk in chunks:
            if len(chunk.strip()) < 200:
                continue
            parser = PlaintextParser.from_string(chunk, Tokenizer("english"))
            summarizer = LsaSummarizer()
            sentences = summarizer(parser.document, min(2, self.sentence_count))
            extracted.extend(str(s) for s in sentences)
        # 去重保持顺序
        seen = set()
        unique = []
        for s in extracted:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        summary = " ".join(unique[:6])
        if len(summary) > self.max_len:
            summary = summary[:self.max_len].rsplit(' ', 1)[0] + "..."
        return summary

class TextRankSummarizer(BaseSummarizer):
    """位置加权 TextRank 抽取式摘要（推荐用于剧情）"""
    def __init__(self, top_n: int = 5, pos_weight: float = 0.3, max_len: int = 500):
        """
        :param top_n: 输出句子数量（约5句对应150-300词）
        :param pos_weight: 位置先验权重 (0~1)，越大越偏向靠前句子
        :param max_len: 最终摘要最大字符数
        """
        self.top_n = top_n
        self.pos_weight = pos_weight
        self.max_len = max_len

    def _textrank_with_position(self, sentences):
        """对句子列表执行位置加权 TextRank，返回排序后的句子列表（已按原文顺序）"""
        if len(sentences) <= self.top_n:
            return sentences

        # TF-IDF 向量化
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = vectorizer.fit_transform(sentences)
        sim_matrix = cosine_similarity(tfidf)
        np.fill_diagonal(sim_matrix, 0)

        # PageRank
        graph = nx.from_numpy_array(sim_matrix)
        pr_scores = nx.pagerank(graph, alpha=0.85)

        # 位置先验（指数衰减）
        N = len(sentences)
        position_prior = [np.exp(-i / (N/2)) for i in range(N)]
        position_prior = np.array(position_prior) / np.sum(position_prior)

        # 融合得分
        final_scores = {}
        for i in range(N):
            final_scores[i] = (1 - self.pos_weight) * pr_scores[i] + self.pos_weight * position_prior[i]

        # 取 top_n 个索引并按原文顺序返回
        top_indices = sorted(final_scores, key=final_scores.get, reverse=True)[:self.top_n]
        top_indices.sort()
        return [sentences[i] for i in top_indices]

    def generate(self, text: str) -> str:
        if not text or len(text.strip()) < 50:
            return text or ""

        # 短文本直接返回
        if len(text) <= 500:
            return text

        # 分句
        sentences = sent_tokenize(text)
        if len(sentences) <= self.top_n:
            summary = text
        
        # 对于超长文本（>20000字符），建议先截取前 20000 字符，避免计算过慢
        if len(text) > 20000:
            truncated = text[:20000]
            sentences = sent_tokenize(truncated)

        selected = self._textrank_with_position(sentences)
        
          # ===== 强制包含第一句 =====
        first_sent = sentences[0]
        if first_sent not in selected:
            if len(selected) >= self.top_n:
                # 替换掉最后一个（即得分最低的）
                selected[-1] = first_sent
            else:
                selected.append(first_sent)
            # 重新按原文顺序排序
            selected = sorted(selected, key=lambda s: sentences.index(s))
         # =========================
            
        summary = " ".join(selected)

        # 长度限制
        if len(summary) > self.max_len:
            summary = summary[:self.max_len].rsplit(' ', 1)[0] + "..."
        return summary