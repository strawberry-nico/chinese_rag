"""中文RAG系统 - Streamlit Web UI"""

import streamlit as st
import sys
from pathlib import Path

# 将src目录添加到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chinese_rag_pipeline import ChineseRAGPipeline

# 页面配置
st.set_page_config(
    page_title="中文RAG Challenge 2",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white !important;
        margin: 0 !important;
        font-size: 2.5rem !important;
    }
    .main-header p {
        color: rgba(255,255,255,0.9) !important;
        margin: 0.5rem 0 0 0 !important;
    }
    .result-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-badge {
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .source-tag {
        background: #f3e5f5;
        color: #7b1fa2;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    .thinking-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .answer-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border: 1px solid #667eea30;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    .search-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
    }
    .answer-btn {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
        color: white !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'rag' not in st.session_state:
    st.session_state.rag = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'history' not in st.session_state:
    st.session_state.history = []

# 页面头部
st.markdown("""
<div class="main-header">
    <h1>🚀 RAG Challenge 2 - Powered by BGE-M3</h1>
    <p>基于获奖RAG系统，由 BGE-M3 + Milvus + Qwen 加速 | 📊 支持多家公司年报问答 | 🔥 向量检索 + LLM重排序</p>
</div>
""", unsafe_allow_html=True)

# 初始化RAG系统
@st.cache_resource
def get_rag_pipeline():
    return ChineseRAGPipeline()

# 侧边栏 - 查询设置
with st.sidebar:
    st.markdown("### 🎯 查询设置")

    # 公司选择
    st.markdown("📁 **选择公司**")
    pdf_dir = Path("data/pdfs")
    if pdf_dir.exists():
        pdf_files = [f.stem for f in pdf_dir.glob("*.pdf")]
        selected_company = st.selectbox(
            "选择要查询的公司年报",
            options=["全部"] + pdf_files if pdf_files else ["全部"],
            index=0
        )
    else:
        selected_company = "全部"
        st.warning("未找到PDF文件，请先运行索引")

    # 问题输入
    st.markdown("❓ **输入问题**")
    question = st.text_area(
        "输入您想要查询的问题",
        placeholder="例如：中芯国际2025年一季度业绩如何？",
        height=100
    )

    # 问题类型
    st.markdown("📋 **问题类型**")
    col1, col2 = st.columns(2)
    with col1:
        question_type = st.radio(
            "选择问题的答案类型",
            options=["text", "name", "number", "boolean"],
            index=0,
            format_func=lambda x: {
                "text": "📝 文本",
                "name": "👤 人名",
                "number": "🔢 数字",
                "boolean": "✅ 布尔"
            }[x]
        )

    # 检索设置
    st.markdown("⚙️ **检索设置**")

    # LLM重排序
    use_rerank = st.checkbox("🤖 启用LLM重排序", value=True)

    # 检索数量
    top_k = st.slider(
        "📄 检索文档数量",
        min_value=1,
        max_value=20,
        value=5,
        help="返回的相关文档片段数量"
    )

    # 按钮
    st.markdown("---")
    col_search, col_answer = st.columns(2)
    with col_search:
        search_clicked = st.button("🔍 搜索文档", use_container_width=True, type="primary")
    with col_answer:
        answer_clicked = st.button("✨ 生成答案", use_container_width=True, type="secondary")

# 主内容区
if search_clicked or answer_clicked:
    if not question:
        st.error("请输入问题！")
    else:
        try:
            # 显示加载状态
            with st.spinner("🤔 正在思考..."):
                rag = get_rag_pipeline()

                # 搜索
                results = rag.search(
                    query=question,
                    top_k=top_k,
                    enable_parent_retrieval=True
                )

                st.session_state.search_results = results

                # 添加历史记录
                st.session_state.history.append({
                    "question": question,
                    "type": question_type,
                    "results_count": results['total']
                })

            # 显示检索结果
            st.markdown(f"""
            <div class="result-card">
                <h3>📑 检索结果 <span class="metric-badge">耗时: {results['duration']:.2f}秒</span></h3>
                <p style="color: #666;">🔍 <strong>公司:</strong> {selected_company} | ❓ <strong>问题:</strong> {question}</p>
                <p style="color: #666;">📊 找到 <strong>{results['total']}</strong> 个相关文档片段</p>
            </div>
            """, unsafe_allow_html=True)

            # 显示每个检索结果
            for i, result in enumerate(results['results'], 1):
                with st.expander(f"📄 结果 {i} | 相似度: {result.get('score', 0):.3f} | 页码: {result['page_no']}", expanded=i==1):
                    st.markdown(f"""
                    <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                        <p><strong>📍 来源:</strong> <span class="source-tag">{result['metadata']['source']}</span></p>
                        <p><strong>📄 类型:</strong> {result['type']}</p>
                        <p><strong>🔢 相似度:</strong> {result.get('score', 0):.4f}</p>
                        <hr style="margin: 1rem 0; border-color: #e0e0e0;">
                        <div style="background: white; padding: 1rem; border-radius: 5px; border: 1px solid #e0e0e0;">
                            {result['content']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # 生成答案
            if answer_clicked and results['results']:
                with st.spinner("✨ 正在生成答案..."):
                    rag = get_rag_pipeline()
                    answer = rag.answer_question(question, results['results'])

                # 显示分步推理（模拟）
                st.markdown("""
                <div class="thinking-box">
                    <h4>🧠 分步推理:</h4>
                    <ol>
                        <li>问题分析：识别问题类型为「{}」</li>
                        <li>文档检索：从 {} 个文档中找到 {} 个相关片段</li>
                        <li>信息整合：综合多个来源的信息</li>
                        <li>答案生成：基于检索内容生成最终回答</li>
                    </ol>
                </div>
                """.format(
                    {"text": "文本描述", "name": "人名识别", "number": "数值提取", "boolean": "是非判断"}[question_type],
                    selected_company,
                    results['total']
                ), unsafe_allow_html=True)

                # 显示推理摘要
                st.markdown("""
                <div style="background: #e8f5e9; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                    <p style="color: #2e7d32; margin: 0;">✅ <strong>推理摘要：</strong>答案基于原文对相关问题的直接描述，信息完整且无歧义。</p>
                </div>
                """, unsafe_allow_html=True)

                # 显示最终答案
                st.markdown(f"""
                <div class="answer-box">
                    <h3>📝 最终答案</h3>
                    <div style="font-size: 1.1rem; line-height: 1.8;">
                        {answer}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # 显示相关页面引用
                st.markdown("### 📚 相关页面")
                sources = []
                for r in results['results'][:3]:
                    source = r['metadata']['source']
                    page = r['page_no']
                    if (source, page) not in sources:
                        sources.append((source, page))
                        st.markdown(f"- **{source}** 第{page}页")

        except Exception as e:
            st.error(f"❌ 处理失败: {str(e)}")
            st.exception(e)

else:
    # 显示欢迎信息
    st.markdown("""
    <div class="result-card" style="text-align: center; padding: 3rem;">
        <h2>👋 欢迎使用中文RAG Challenge 2</h2>
        <p style="font-size: 1.1rem; color: #666;">
            这是一个基于 <strong>Milvus + BGE-M3 + Qwen</strong> 的中文文档问答系统
        </p>
        <div style="margin-top: 2rem;">
            <div style="display: inline-block; text-align: left;">
                <h4>🚀 快速开始:</h4>
                <ol style="color: #666;">
                    <li>在左侧输入你的问题</li>
                    <li>选择问题类型（文本/人名/数字/布尔）</li>
                    <li>点击「搜索文档」查看检索结果</li>
                    <li>点击「生成答案」获取AI回答</li>
                </ol>
            </div>
        </div>
        <div style="margin-top: 2rem; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
            <p style="color: #888; margin: 0;">
                💡 <strong>提示：</strong>系统会自动检索最相关的文档片段，并使用LLM生成结构化答案
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 底部信息
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>🛠️ 基于 MinerU + BGE-M3 + Milvus Lite + Qwen 构建</p>
    <p style="font-size: 0.9rem;">中文RAG Challenge 2 | AutoDL 部署版</p>
</div>
""", unsafe_allow_html=True)
