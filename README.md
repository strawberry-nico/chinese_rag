# 中文RAG系统

基于MinerU、BGE-M3、Weaviate和Qwen-turbo的中文文档问答系统

## 🎯 系统特点

- **中文优化**: 专为中文PDF文档设计，支持中文语义理解
- **父子文档**: 基于MinerU JSON输出的精确父子文档关系
- **混合检索**: Weaviate内置混合检索，支持自动alpha调节
- **智能重排**: Qwen-turbo重排序，提升检索质量
- **稳定可靠**: 完善的错误处理和降级机制

## 🏗️ 系统架构

```
PDF文档 → MinerU解析 → 父子文档结构 → BGE-M3向量化 → Weaviate存储
    ↑                                                        ↓
用户查询 ← 答案生成 ← Qwen重排序 ← 父文档检索 ← 混合检索
```

## 📦 安装

### 1. 克隆项目
```bash
git clone <repository-url
git clone <repository-url
cd chinese_rag
```

### 2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 安装MinerU（可选）
如果没有MinerU，系统会使用模拟数据
```bash
git clone https://github.com/opendatalab/MinerU.git
cd MinerU
pip install -e .
cd ..
```

### 5. 启动Weaviate
```bash
docker run -d --name weaviate \
  -p 8080:8080 \
  -p 50051:50051 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH="/var/lib/weaviate" \
  semitechnologies/weaviate:latest
```

### 6. 配置环境变量
创建 `.env` 文件：
```bash
# API密钥
DASHSCOPE_API_KEY=your_dashscope_api_key

# Weaviate连接（可选）
WEAVIATE_URL=localhost:8080

# GPU配置（可选）
CUDA_VISIBLE_DEVICES=0
```

## 🚀 快速开始

### 1. 索引文档
```bash
# 索引data/pdfs目录下的所有PDF
python main.py index --pdf-dir data/pdfs --batch-size 32
```

### 2. 搜索文档
```bash
# 基本搜索
python main.py search "人工智能的发展历程"

# 高级搜索选项
python main.py search "机器学习算法" \
  --top-k 10 \
  --no-auto-tune \
  --no-parent-retrieval
```

### 3. 问答模式
```bash
# 基于检索结果生成答案
python main.py qa "什么是深度学习？"
```

### 4. 运行测试
```bash
# 测试所有模块
python main.py test
```

## 🔧 配置

编辑 `config/config.yaml` 文件：

```yaml
# PDF解析
pdf_parser:
  device: "cuda:0"  # GPU设备

# 向量化
embedding:
  model_name: "BAAI/bge-m3"

# 向量数据库
vector_store:
  collection_name: "ChineseDocuments"
  initial_alpha: 0.5  # 混合检索权重

# 自动调节
auto_tuner:
  window_size: 10
  variance_threshold_high: 0.3
  variance_threshold_low: 0.1

# 重排序
reranker:
  model: "qwen-turbo"
```

## 📊 性能优化

### GPU内存优化
- 自动调整batch_size根据文本长度
- 动态GPU内存管理
- 父文档LRU缓存

### 检索优化
- 混合检索alpha自动调节
- 父文档上下文增强
- 智能重排序

### 错误处理
- 完善的降级机制
- 重试策略
- 模拟数据支持

## 🧪 开发

### 项目结构
```
chinese_rag/
├── src/                    # 源代码
│   ├── config.py          # 配置管理
│   ├── pdf_parser_mineru.py  # PDF解析
│   ├── embedding_bge.py   # 向量化
│   ├── vector_store_weaviate.py  # 向量存储
│   ├── auto_tuner.py      # 自动调节
│   ├── reranker_qwen.py   # 重排序
│   ├── parent_document_retriever.py  # 父文档检索
│   └── chinese_rag_pipeline.py  # 主流程
├── config/                # 配置文件
├── tests/                 # 测试
├── logs/                  # 日志
├── data/                  # 数据
└── main.py               # 主入口
```

### 运行测试
```bash
# 测试各个模块
python -m pytest tests/

# 运行集成测试
python main.py test
```

## 🔍 调试

### 日志查看
```bash
# 实时查看日志
tail -f logs/chinese_rag.log

# 调试模式
python main.py --log-level DEBUG search "查询"
```

### 性能分析
```bash
# 导出搜索历史
python -c "from chinese_rag_pipeline import ChineseRAGPipeline; \
           with ChineseRAGPipeline() as rag: \
               rag.export_search_history('search_history.json')"
```

## 🤝 贡献

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

MIT License

## 🙏 致谢

- [MinerU](https://github.com/opendatalab/MinerU) - PDF解析
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) - 向量化模型
- [Weaviate](https://weaviate.io/) - 向量数据库
- [DashScope](https://dashscope.aliyun.com/) - Qwen-turbo API

## 📞 支持

如有问题，请提交Issue或联系维护者。