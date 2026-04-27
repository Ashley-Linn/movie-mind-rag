class SmartTextSplitter:
    """
    智能文本分割器（迭代实现，无递归）
    标准执行流程：
    1. 判断短文档豁免 → 直接整块不切
    2. 判断文本小于块上限 → 不切割
    3. 按段落、句子、标点优先级做语义切割（全程无重叠）
    4. 合并末尾短小碎片，避免语义零散
    5. 所有分块完成后，【仅最后统一加一次重叠】绝不重复、不冲突
    6. 支持生成大尺寸父块（用于LLM长上下文推理）
    """

    def __init__(
        self,
        chunk_size: int,             # 单个文本块最大字符长度，必须手动传入
        chunk_overlap: int = 0,      # 块间重叠字符数，默认0=不重叠
        min_fragment_size: int = 40  # 小于此长度的碎片自动合并到前一块
    ):
        """
        参数作用说明：
        chunk_overlap 写在初始化里，【只做参数保存】，不执行重叠逻辑
        初始化本身不会生成重叠内容，重叠只在最后一步单独函数处理
        因此不会出现逻辑冲突、重复叠加、块无限变长问题
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_fragment_size = min_fragment_size

        # 语义分隔符优先级：从大语义单位到小语义单位
        self.separators = [
            "\n\n",   # 段落分隔
            "\n",     # 换行分隔
            ". ",     # 英文句子结尾
            "! ",
            "? ",
            "; ",
            ", ",
            " "       # 单词分隔（最后一级）
        ]

    # ------------------------------------------------------------
    # 对外统一入口
    # ------------------------------------------------------------
    def split(self, text: str, is_whole_document: bool = True, short_threshold: int = 200) -> list[str]:
        """对外统一分块入口函数"""
        # 过滤空文本、空白无效内容
        if not text or not text.strip():
            return []

        # 规则1：完整文档 + 长度小于短文档阈值 → 直接返回整块，不做任何切割
        if is_whole_document and len(text) < short_threshold:
            return [text.strip()]

        # 规则2：文本本身长度 ≤ 单块上限 → 无论如何都不切割
        if len(text) <= self.chunk_size:
            return [text.strip()]

        # 三步标准流水线：切割 → 合并碎片 → 最后单独加重叠
        chunks = self._split_by_semantic_iterative(text)   # 纯语义切割，无重叠，迭代实现
        chunks = self._merge_small(chunks)                 # 合并短小碎片，无重叠
        chunks = self._add_overlap_once(chunks)            # 全流程唯一一次加重叠

        # 清理多余空白、过滤空块
        return [c.strip() for c in chunks if c.strip()]

    # ------------------------------------------------------------
    # 核心切割：迭代实现（无递归）
    # ------------------------------------------------------------
    def _split_by_semantic_iterative(self, text: str) -> list[str]:
        """
        迭代式语义切割，完全替代递归
        逻辑：
        1. 使用队列处理文本段
        2. 按分隔符优先级从大到小切割
        3. 尽量拼接，不超过 chunk_size 才保存
        4. 绝不把句子切碎
        """
        final_chunks = []
        queue = [(text, 0)]  # (待处理文本, 当前分隔符级别)

        while queue:
            segment, sep_idx = queue.pop(0)

            # 满足长度 → 直接加入结果
            if len(segment) <= self.chunk_size:
                final_chunks.append(segment)
                continue

            # 分隔符已用完 → 暴力切割
            if sep_idx >= len(self.separators):
                final_chunks.extend(self._force_split(segment))
                continue

            current_sep = self.separators[sep_idx]

            # 当前分隔符不存在 → 下一级
            if current_sep not in segment:
                queue.append((segment, sep_idx + 1))
                continue

            # ========================
            # 正确按分隔符拆分 + 尽量拼接（不会碎）
            # ========================
            parts = segment.split(current_sep)
            current = ""
            temp_chunks = []

            for part in parts:
                # 构造候选拼接文本
                candidate = current + current_sep + part if current else part

                if len(candidate) <= self.chunk_size:
                    # 没超限 → 继续拼接
                    current = candidate
                else:
                    # 超限 → 保存当前块，用下一级分隔符处理剩余部分
                    if current:
                        temp_chunks.append(current)
                    current = part

            # 加入最后一段
            if current:
                temp_chunks.append(current)

            # 超长块继续下一级切割
            for chunk in temp_chunks:
                if len(chunk) > self.chunk_size:
                    queue.append((chunk, sep_idx + 1))
                else:
                    final_chunks.append(chunk)

        return final_chunks

    # ------------------------------------------------------------
    # 兜底暴力切割
    # ------------------------------------------------------------
    def _force_split(self, text: str) -> list[str]:
        """
        暴力按固定长度切分（无任何可用分隔符时使用）
        修复：step = chunk_size，无重叠，避免重复
        """
        step = self.chunk_size
        return [text[i:i+step] for i in range(0, len(text), step)]

    # ------------------------------------------------------------
    # 合并短小碎片
    # ------------------------------------------------------------
    def _merge_small(self, chunks: list[str]) -> list[str]:
        """
        合并切割后产生的超短句碎片，保证语义连贯不零散。
        限制：合并后不超过 chunk_size
        """
        if not chunks:
            return []

        res = [chunks[0]]
        for c in chunks[1:]:
            if len(c) < self.min_fragment_size and len(res[-1]) + len(c) + 1 <= self.chunk_size:
                res[-1] += " " + c
            else:
                res.append(c)
        return res

    # ------------------------------------------------------------
    # 添加重叠
    # ------------------------------------------------------------
    def _add_overlap_once(self, chunks: list[str]) -> list[str]:
        """
        全流程唯一一处添加重叠，只执行一次。
        自动在空格处断开，绝不切断单词。
        """
        if self.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = result[-1]
            curr = chunks[i]

            # 实际重叠长度
            ol = min(self.chunk_overlap, len(prev))
            start = len(prev) - ol

            # 向前找第一个空格，保证不切单词
            while start < len(prev) and prev[start] != ' ':
                start += 1

            # 截取重叠内容
            overlap_text = prev[start:]
            # 拼接（自然带空格，不重复）
            result.append(overlap_text + curr)

        return result

    # ------------------------------------------------------------
    # 父块生成（用于LLM）
    # ------------------------------------------------------------
    def create_parent_chunks(self, text: str, parent_size: int = 4000) -> list[str]:
        """
        生成父块（大尺寸上下文块）
        用途：给LLM大模型做长文本总结、整体剧情理解
        特点：强制不重叠、大块切割、关闭短文档保护
        """
        if len(text) <= parent_size:
            return [text.strip()]

        parent_splitter = SmartTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=0,
            min_fragment_size=0
        )

        parent_chunks = parent_splitter.split(text, is_whole_document=False)
        return parent_chunks