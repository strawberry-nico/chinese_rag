"""BGE-M3向量化模块"""

import logging
import torch
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class BGEM3Embedding:
    """BGE-M3多语言embedding模型"""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.dimension = 1024  # BGE-M3输出维度

        # 延迟导入，避免安装时依赖问题
        try:
            from FlagEmbedding import BGEM3FlagModel
            self.model = BGEM3FlagModel(
                model_name,
                use_fp16=True,
                device=device
            )
            logger.info(f"成功加载BGE-M3模型: {model_name}")
        except ImportError:
            logger.warning("FlagEmbedding未安装，将使用模拟embedding")
            self.model = None

    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> List[List[float]]:
        """编码文本为向量"""
        if not texts:
            return []

        if self.model is None:
            # 模拟模式
            return self._generate_mock_embeddings(len(texts))

        try:
            # 处理空文本
            texts = [t if t else " " for t in texts]

            # 动态调整batch_size
            batch_size = self._optimize_batch_size(texts, batch_size)

            logger.info(f"编码 {len(texts)} 个文本，batch_size={batch_size}")

            # 批量编码
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                max_length=8192,
                return_dense=True
            )

            # 返回dense向量
            vectors = embeddings['dense_vecs'].tolist()

            logger.info(f"完成编码，向量维度: {len(vectors[0]) if vectors else 0}")

            return vectors

        except Exception as e:
            logger.error(f"编码失败: {e}")
            # 降级到模拟向量
            return self._generate_mock_embeddings(len(texts))

    def encode_queries(self, queries: List[str], batch_size: int = 32) -> List[List[float]]:
        """编码查询（添加指令）"""
        if not queries:
            return []

        # BGE-M3的查询指令
        instruction = "Represent this sentence for searching relevant passages: "
        queries_with_inst = [instruction + q for q in queries]

        return self.encode(queries_with_inst, batch_size=batch_size)

    def _optimize_batch_size(self, texts: List[str], requested_batch_size: int) -> int:
        """根据文本长度和GPU内存优化batch_size"""
        if not texts:
            return requested_batch_size

        # 计算平均文本长度
        avg_length = sum(len(t) for t in texts) / len(texts)

        # 根据文本长度调整batch_size
        if avg_length > 4000:  # 长文本
            optimal_batch = min(8, requested_batch_size)
        elif avg_length > 2000:  # 中等文本
            optimal_batch = min(16, requested_batch_size)
        else:  # 短文本
            optimal_batch = min(32, requested_batch_size)

        # GPU内存检查（简化版）
        if torch.cuda.is_available():
            free_memory = torch.cuda.mem_get_info()[0] / 1024**3  # GB
            if free_memory < 4:  # 少于4GB可用内存
                optimal_batch = min(optimal_batch, 8)

        if optimal_batch != requested_batch_size:
            logger.info(f"调整batch_size: {requested_batch_size} -> {optimal_batch} (avg_length={avg_length:.0f})")

        return optimal_batch

    def _generate_mock_embeddings(self, count: int) -> List[List[float]]:
        """生成模拟embedding向量（用于测试）"""
        logger.warning(f"使用模拟embedding生成 {count} 个向量")

        # 生成随机但归一化的向量
        vectors = []
        for _ in range(count):
            # 随机向量
            vector = np.random.randn(self.dimension).astype(np.float32)
            # 归一化
            vector = vector / np.linalg.norm(vector)
            # 添加小的随机偏移，使向量有差异
            vector = vector + np.random.randn(self.dimension) * 0.01
            # 再次归一化
            vector = vector / np.linalg.norm(vector)
            vectors.append(vector.tolist())

        return vectors

    def similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def get_dimension(self) -> int:
        """获取向量维度"""
        return self.dimension

# 测试函数
def test_embedding():
    """测试向量化功能"""
    encoder = BGEM3Embedding(device="cpu")  # 测试用CPU

    # 测试文本
    texts = [
        "人工智能是计算机科学的一个分支",
        "机器学习是人工智能的重要技术",
        "深度学习是机器学习的一个子领域",
        "",  # 空文本
        "自然语言处理让计算机理解人类语言" * 100  # 长文本
    ]

    print("编码文本...")
    embeddings = encoder.encode(texts, batch_size=2)

    print(f"生成 {len(embeddings)} 个向量")
    print(f"向量维度: {len(embeddings[0])}")

    # 测试相似度
    if len(embeddings) >= 3:
        sim12 = encoder.similarity(embeddings[0], embeddings[1])
        sim13 = encoder.similarity(embeddings[0], embeddings[2])
        print(f"文本1与文本2相似度: {sim12:.4f}")
        print(f"文本1与文本3相似度: {sim13:.4f}")

    # 测试查询编码
    queries = ["什么是人工智能", "机器学习应用"]
    query_embeddings = encoder.encode_queries(queries)
    print(f"查询向量数量: {len(query_embeddings)}")

    return embeddings

if __name__ == "__main__":
    test_embedding()
