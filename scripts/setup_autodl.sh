#!/bin/bash
# AutoDL环境设置脚本

set -e

echo "开始设置中文RAG系统AutoDL环境..."

# 1. 检查GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "警告: 未检测到NVIDIA GPU驱动"
else
    echo "GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
fi

# 2. 创建虚拟环境
echo "创建虚拟环境..."
python -m venv venv
source venv/bin/activate

# 3. 升级pip
echo "升级pip..."
pip install --upgrade pip

# 4. 安装依赖
echo "安装基础依赖..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. 安装项目依赖
echo "安装项目依赖..."
pip install -r requirements.txt

# 6. 安装MinerU（如果未安装）
if ! python -c "import magic_pdf" &> /dev/null; then
    echo "安装MinerU..."
    git clone https://github.com/opendatalab/MinerU.git /tmp/mineru
    cd /tmp/mineru
    pip install -e .
    cd -
fi

# 7. 下载BGE-M3模型（如果本地没有）
MODEL_DIR="models/bge-m3"
if [ ! -d "$MODEL_DIR" ]; then
    echo "下载BGE-M3模型..."
    mkdir -p "$MODEL_DIR"
    cd "$MODEL_DIR"

    # 下载必要的文件
    wget https://huggingface.co/BAAI/bge-m3/resolve/main/config.json
    wget https://huggingface.co/BAAI/bge-m3/resolve/main/pytorch_model.bin
    wget https://huggingface.co/BAAI/bge-m3/resolve/main/tokenizer.json
    wget https://huggingface.co/BAAI/bge-m3/resolve/main/tokenizer_config.json
    wget https://huggingface.co/BAAI/bge-m3/resolve/main/vocab.txt

    cd -
fi

# 8. 启动Weaviate（如果未运行）
if ! docker ps | grep -q weaviate; then
    echo "启动Weaviate容器..."
    docker run -d --name weaviate \
        -p 8080:8080 \
        -p 50051:50051 \
        -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
        -e PERSISTENCE_DATA_PATH="/var/lib/weaviate" \
        -v weaviate_data:/var/lib/weaviate \
        semitechnologies/weaviate:latest

    # 等待Weaviate启动
    echo "等待Weaviate启动..."
    sleep 10
fi

# 9. 设置GPU内存限制
echo "设置GPU内存限制..."
cat > .env << EOF
# GPU配置
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 内存优化
GPU_MEMORY_FRACTION=0.8
BATCH_SIZE_EMBEDDING=16
BATCH_SIZE_MINERU=1
EOF

# 10. 创建必要的目录
echo "创建目录结构..."
mkdir -p data/pdfs
mkdir -p data/processed
mkdir -p logs
mkdir -p outputs

# 11. 创建示例PDF（如果没有）
if [ ! -f "data/pdfs/sample.pdf" ]; then
    echo "创建示例PDF..."
    # 这里可以下载一些示例PDF或创建空文件
    echo "请将您的PDF文件放入 data/pdfs/ 目录中"
fi

# 12. 设置文件权限
echo "设置文件权限..."
chmod +x main.py
chmod +x scripts/*.sh

# 13. 验证安装
echo "验证安装..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
if python -c "import magic_pdf" &> /dev/null; then
    echo "MinerU已安装"
else
    echo "MinerU未安装（将使用模拟模式）"
fi

# 14. 性能测试
echo "运行性能测试..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'可用显存: {torch.cuda.memory_allocated() / 1024**3:.1f} GB')
"

# 15. 启动脚本
echo "创建启动脚本..."
cat > start.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
export $(cat .env | xargs)
echo "启动中文RAG系统..."
python main.py "$@"
EOF
chmod +x start.sh

echo ""
echo "AutoDL环境设置完成！"
echo ""
echo "使用方法："
echo "1. 激活环境: source venv/bin/activate"
echo "2. 索引文档: ./start.sh index --pdf-dir data/pdfs"
echo "3. 搜索文档: ./start.sh search '人工智能'"
echo "4. 问答模式: ./start.sh qa '什么是深度学习？'"
echo "5. 运行测试: ./start.sh test"
echo ""
echo "注意事项："
echo "- 确保已设置 DASHSCOPE_API_KEY 环境变量"
echo "- PDF文件请放在 data/pdfs/ 目录下"
echo "- 日志文件在 logs/ 目录中"
echo "- 如果GPU内存不足，请减小BATCH_SIZE_EMBEDDING"
echo ""
echo "查看日志：tail -f logs/chinese_rag.log""""file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/scripts/setup_autodl.sh