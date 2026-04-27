"""
RAG 问答脚本:提供构建好的 LCEL 链，支持检索 + 生成
运行后选择模式：
1. 固定查询（示例：“时间旅行科幻电影”）
2. 交互输入（可连续提问，输入 exit 退出）
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Settings
from src.core.retriever import search


# ------------------------------------------------------------------
# Pydantic 模型（用于参数校验）
# ------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., description="用户问题", min_length=1)
    top_k: int = Field(default=getattr(Settings, 'TOP_K', 5), description="返回电影数量")
    mode: str = Field(default="strict", description="回答模式: strict 或 free")


# ------------------------------------------------------------------
# 动态构建提示词（根据模式）
# ------------------------------------------------------------------
def build_prompt(mode: str):
    if mode == "strict":
        return ChatPromptTemplate.from_messages([
            ("system", """你是专业的电影问答助手, 你必须严格遵循以下规则：

1. **信息来源**：只根据下方【相关电影信息】回答，不得编造任何内容。如果检索结果中没有相关信息，不要尝试推理或联想。
2. **信息不足时**：如果【相关电影信息】中找不到答案，请直接回复“相关信息找不到答案”。
3. **空结果处理**：如果【相关电影信息】为空，请直接回复“无相关信息，无法回答”，不要自行发挥或编造。
4. 即使某些电影的相似度显示为 0.0，它们仍然可能因标题匹配而被提供，请将其视为有效信息。
5. **支持二次推理**：你可以对检索结果进行对比、总结、排名等分析，例如：
   - 找出其中年份最早/最晚的电影。
   - 总结这些电影的共同类型或主题。
   - 回答复合问题（如“列出剧情中包含时间旅行的电影中，导演是谁？”）。
   但所有结论必须严格基于提供的信息，不得引入外部知识。
6. **输出格式**：
   - 只列出与问题最相关的3部电影（若不足3部则全部列出)   
   - 每部电影格式：`《标题》(年份) - 剧情简介（限350字内）`
   - 剧情简介尽量要以句号结尾，同时保持剧情通顺
   - 不同电影之间信息要隔开一行，不许放在一起，便于区分
   - 不要输出额外的解释、评价，直接按上述格式列出电影
   - 如果问题要求分析、比较或总结，请用简洁的自然语言段落回答，不要添加无关内容。
   - 如果无法完成推理（例如缺少必要字段），请说明“根据现有信息无法进行该项分析”。
7. **示例**：
   - 列表示例：
     《肖申克的救赎》(1994) - 银行家越狱复仇。
     《阿甘正传》(1994) - 傻人自有傻福。
   - 分析示例：
     问题：“这些电影中哪一部年份最早？”
     回答：根据提供的信息，年份最早的是《肖申克的救赎》(1994)，其他电影均为1995年之后。
"""),
            ("human", """
【用户问题】
{query}

【相关电影信息】
{context}

请根据上述规则回答。
""")
        ])
    else:  # free mode
        return ChatPromptTemplate.from_messages([
            ("system", """你是电影问答助手。请优先使用下方【相关电影信息】中的内容进行回答。如果信息不足以完全回答用户问题，你可以适当结合自己的知识进行补充，但需保持与电影相关。回答风格可以自然、有创意，不需要固定格式，但不要过度编造。"""),
            ("human", """
【用户问题】
{query}

【相关电影信息】
{context}

请根据上述规则回答。
""")
        ])


# ------------------------------------------------------------------
# 构建 LCEL 链（支持模式选择）
# ------------------------------------------------------------------
llm = ChatOpenAI(
    model=Settings.LLM_MODEL,
    api_key=Settings.DEEPSEEK_API_KEY,
    base_url=Settings.DEEPSEEK_API_BASE,
    temperature=0.1,
    max_tokens=2048,
    max_retries=3,
    timeout=60,
)


def create_rag_chain(mode: str = "strict"):
    """根据模式创建 LCEL 链"""
    prompt = build_prompt(mode)

    def format_context(query: str, top_k: int = 5) -> str:
        """检索并格式化为上下文字符串"""
        movies = search(query, top_k=top_k)
        if not movies:
            return ""  # 空字符串，让 prompt 处理
        return "\n\n".join([
            f"🎬 电影：{m['title']} ({m['year']})\n"
            f"类型：{m['genre']}\n"
            f"导演：{m['director']}\n"
            f"剧情：{m['plot'][:300]}..."
            for m in movies
        ])

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_context(x["query"], x.get("top_k", 5))
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ------------------------------------------------------------------
# 封装调用函数（对外统一接口）
# ------------------------------------------------------------------
def ask(query: str, top_k: int = None, mode: str = "strict") -> str:
    """调用 RAG 链生成答案"""
    if top_k is None:
        top_k = getattr(Settings, 'TOP_K', 5)
    chain = create_rag_chain(mode)
    return chain.invoke({"query": query, "top_k": top_k})


# ------------------------------------------------------------------
# 命令行入口：两种模式（测试用）
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("电影 RAG 问答系统 (模块测试)")
    print("=" * 50)
    mode_choice = input("请选择模式：1-严格模式  2-自由模式\n请输入数字: ").strip()
    if mode_choice == "1":
        mode = "strict"
    elif mode_choice == "2":
        mode = "free"
    else:
        mode = "strict"

    chain = create_rag_chain(mode)

    mode_desc = "严格模式" if mode == "strict" else "自由模式"
    print(f"\n当前模式: {mode_desc}")

    op_mode = input("请选择：1-固定查询  2-交互输入\n请输入数字: ").strip()

    if op_mode == "1":
        request = QueryRequest(query="时间旅行科幻电影", top_k=3, mode=mode)
        answer = chain.invoke({"query": request.query, "top_k": request.top_k})
        print("\n问题:", request.query)
        print("回答:", answer)

    elif op_mode == "2":
        print("输入问题（输入 'exit' 或 'quit' 退出）")
        while True:
            user_input = input("\n请输入问题: ").strip()
            if user_input.lower() in ('exit', 'quit'):
                break
            if not user_input:
                continue
            request = QueryRequest(query=user_input, mode=mode)
            answer = chain.invoke({"query": request.query, "top_k": request.top_k})
            print("\n回答:", answer)
    else:
        print("无效选择，退出。")