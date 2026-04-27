"""
FastAPI 后端：提供搜索和问答接口（工业最佳实践版）
- 使用 lifespan 预热模型和索引
- 保留异步端点，确保高并发
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import asyncio
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import Settings
from src.core.retriever import search
from src.core.generator import ask


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时预热模型，关闭时清理资源"""
    # 启动时执行
    print("🔄 预热检索器（加载模型和索引）...")
    search("warmup", top_k=1)
    print("✅ 预热完成，服务已就绪")
    yield
    # 关闭时执行
    print("🛑 应用关闭")


app = FastAPI(
    title="Movie RAG API",
    version="1.0",
    lifespan=lifespan
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== API 端点 ====================
class SearchResponse(BaseModel):
    movies: List[Dict[str, Any]]


# 🆕 修改：AskRequest 增加 mode 字段
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=getattr(Settings, 'TOP_K', 5), ge=1, le=20)
    mode: str = Field(default="strict", description="回答模式: strict 或 free")


class AskResponse(BaseModel):
    answer: str


@app.get("/search", response_model=SearchResponse)
async def search_movies(
    q: str = Query(..., min_length=1),
    top_k: int = Query(default=getattr(Settings, 'TOP_K', 5), ge=1, le=20)
):
    """语义搜索电影（异步）"""
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, search, q, top_k)
    return {"movies": results}


@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """RAG 问答（异步），支持 mode 参数"""
    loop = asyncio.get_event_loop()
    # 🆕 修改：调用 ask 函数并传递 mode 参数
    answer = await loop.run_in_executor(
        None,
        lambda: ask(query=req.query, top_k=req.top_k, mode=req.mode)
    )
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)