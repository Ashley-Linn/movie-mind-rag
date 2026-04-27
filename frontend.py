"""
Streamlit 前端 - 电影知识库
支持：
- 语义搜索（返回电影列表，包含标题、年份、产地、导演、主演、类型、剧情摘要）
- 智能问答（调用 RAG 链）
"""

import streamlit as st
import requests

# 后端 API 地址
API_BASE = "http://localhost:8000"


st.set_page_config(page_title="🎬 电影知识库", layout="wide")
st.title("🎬 电影知识库 - 智能搜索与问答")

# 侧边栏设置
with st.sidebar:
    st.header("⚙️ 设置")
    mode = st.radio("模式", ["🔍 语义搜索", "💬 智能问答"])
    top_k = st.slider("返回电影数量", 1, 10, 5)
    st.markdown("**📖 模式说明**")
    st.caption("- **语义搜索**：直接返回相关电影列表，展示相似度分数。")
    st.markdown("""
- **智能问答**：用自然语言回答，支持两种子模式：
    - **严格模式**：回答格式固定（列表形式），必须基于检索信息，信息不足时明确告知。
    - **自由模式**：回答无固定格式，可结合自身知识自由发挥，更有创意。
""")
      
    # 🆕 回答模式选择（仅在智能问答模式下显示）
    if mode == "💬 智能问答":
        answer_mode = st.radio(
            "回答模式",
            ["严格模式 (基于检索结果)", "自由模式 (可发挥创意)"],
            index=0,
            help="严格模式：只根据检索到的电影信息回答，不编造；自由模式：可结合自身知识自由发挥"
        )
        # 转换为后端需要的参数值
        mode_value = "strict" if answer_mode == "严格模式 (基于检索结果)" else "free"
    else:
        mode_value = "strict"  # 语义搜索不需要，但保留默认值
    
    st.markdown("---")
    st.markdown("**📖 相似度说明**")
    st.caption("• **0.0**：剧情不相关，但因标题语义匹配被召回，排序由混合检索决定（可能靠前也可能靠后）")
    st.caption("• **0.xx**：剧情相关度（越高越匹配）")
    st.caption("• **1.0**：标题精准匹配，剧情不相关（硬性置顶）")

# 主输入框
query = st.text_input("输入你感兴趣的电影情节、关键词或问题", placeholder="例如：时间旅行科幻电影")

if query:
    if mode == "🔍 语义搜索":
        with st.spinner("搜索中..."):
            try:
                resp = requests.get(
                    f"{API_BASE}/search",
                    params={"q": query, "top_k": top_k},
                    timeout=10
                )
                if resp.status_code == 200:
                    data = resp.json()
                    movies = data.get("movies", [])
                    if not movies:
                        st.warning("未找到相关电影")
                    else:
                        for i, m in enumerate(movies, 1):
                            # 获取相似度显示内容（可能是数字或带备注的字符串）
                            sim_display = m.get('similarity', 0)
                            # 如果 similarity 是数字，格式化为两位小数；如果是字符串直接使用
                            if isinstance(sim_display, (int, float)):
                                sim_text = f"{sim_display:.2f}"
                            else:
                                sim_text = str(sim_display)
                            with st.expander(f"{i}. {m['title']} ({m['year']}) - 相似度 {sim_display}"):
                                col1, col2 = st.columns([1, 3])
                                with col1:

                                    st.markdown(f"**类型 (官方标签)**\n{m.get('genre', '未知')}")
                                    st.caption("⚠️ 官方标签仅供参考，可能与剧情不完全匹配")
                                    st.markdown(f"**产地**\n{m.get('origin', '未知')}")
                                    st.markdown(f"**导演**\n{m.get('director', '未知')}")
                                    st.markdown(f"**主演**\n{m.get('cast', '未知')}")
                                with col2:
                                    st.markdown(f"**关键情节摘录**\n{m.get('plot', '暂无剧情')[:300]}...")
                                    st.caption("⚠️ 剧情摘录基于原文自动生成，可能不完全连贯或准确")
                                    # 添加说明
                                    if isinstance(sim_display, str) and "仅标题匹配" in sim_display:
                                        st.caption("ℹ️ 该电影仅因标题匹配被召回，剧情不相关（相似度标记为1.0）。")
                else:
                    st.error(f"后端错误: {resp.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("无法连接到后端服务，请确保已启动 API (uvicorn api:app --reload)")
            except Exception as e:
                st.error(f"请求失败: {e}")

    else:  # 智能问答模式
        with st.spinner("AI 思考中..."):
            try:
                # 🆕 修改：请求体中增加 mode 参数
                resp = requests.post(
                    f"{API_BASE}/ask",
                    json={"query": query, "top_k": top_k, "mode": mode_value},
                    timeout=30
                )
                if resp.status_code == 200:
                    answer = resp.json().get("answer", "")
                    st.markdown("### 💡 回答")
                    st.success(answer)
                else:
                    st.error(f"后端错误: {resp.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("无法连接到后端服务，请确保已启动 API")
            except Exception as e:
                st.error(f"请求失败: {e}")
else:
    st.info("👆 输入你想查询的内容，开始探索电影世界")
