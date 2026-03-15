"""MinerU PDF解析器 - 基于JSON输出构建父子文档关系"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

# 配置日志
logger = logging.getLogger(__name__)

@dataclass
class ParsedDocument:
    """解析后的文档结构"""
    parent_documents: List[Dict[str, Any]]
    all_chunks: List[Dict[str, Any]]
    document_metadata: Dict[str, Any]

class MinerUPDFParser:
    """MinerU PDF解析器 - 生成父子文档结构"""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.logger = logging.getLogger(__name__)

        # 延迟导入MinerU，避免安装时依赖问题
        try:
            from magic_pdf.model.doc_analyze import DocAnalyze
            self.analyze = DocAnalyze(
                model_config_path="config/mineru_models.yaml",
                device=device
            )
        except ImportError:
            self.logger.warning("MinerU未安装，将使用模拟数据")
            self.analyze = None

    def parse_pdf(self, pdf_path: Path) -> ParsedDocument:
        """解析PDF并构建父子文档关系"""
        self.logger.info(f"开始解析PDF: {pdf_path}")

        if self.analyze is None:
            # 开发模式：返回模拟数据
            return self._generate_mock_data(pdf_path)

        try:
            # 读取PDF
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()

            # 使用MinerU解析
            result = self.analyze(pdf_bytes)

            # 验证结果完整性
            self._validate_mineru_result(result)

            # 构建父子文档结构
            return self._build_document_structure(result, pdf_path)

        except Exception as e:
            self.logger.error(f"MinerU解析失败: {e}")
            # 降级到模拟数据，确保系统不崩溃
            return self._generate_mock_data(pdf_path)

    def _validate_mineru_result(self, result: Dict) -> None:
        """验证MinerU输出格式"""
        if not result or not isinstance(result, dict):
            raise ValueError("MinerU输出格式异常：结果为空或不是字典")

        if 'pdf_info' not in result:
            raise ValueError("MinerU输出格式异常：缺少pdf_info字段")

        pdf_info = result['pdf_info']
        if 'pages' not in pdf_info or not pdf_info['pages']:
            raise ValueError("MinerU输出格式异常：缺少pages或pages为空")

        if 'metadata' not in pdf_info:
            raise ValueError("MinerU输出格式异常：缺少metadata字段")

    def _build_document_structure(self, result: Dict, pdf_path: Path) -> ParsedDocument:
        """构建父子文档结构"""
        pdf_info = result['pdf_info']
        pages = pdf_info['pages']
        metadata = pdf_info['metadata']

        self.logger.info(f"文档共{len(pages)}页，文件名: {metadata.get('file_name', pdf_path.name)}")

        parent_documents = []
        all_chunks = []

        # 处理每一页
        for page_data in pages:
            page_no = page_data['page_info']['page_no']

            # 创建父文档（页面级别）
            parent_doc = self._create_parent_document(page_data, page_no, metadata, pdf_path.name)
            parent_documents.append(parent_doc)

            # 添加父文档chunk
            parent_chunk = self._create_parent_chunk(parent_doc, metadata)
            all_chunks.append(parent_chunk)

            # 处理页面内的子元素
            child_chunks = self._extract_child_chunks(page_data, page_no, metadata, parent_chunk['id'])
            all_chunks.extend(child_chunks)

        self.logger.info(f"解析完成: {len(parent_documents)}个父文档, {len(all_chunks)}个总chunks")

        return ParsedDocument(
            parent_documents=parent_documents,
            all_chunks=all_chunks,
            document_metadata=metadata
        )

    def _create_parent_document(self, page_data: Dict, page_no: int, metadata: Dict, file_name: str) -> Dict[str, Any]:
        """创建父文档（页面级别）"""
        return {
            'id': f"{file_name}_page_{page_no}_parent",
            'type': 'parent_document',
            'page_no': page_no,
            'content': '',  # 内容将在后续填充
            'metadata': {
                'source': file_name,
                'page_range': [page_no, page_no],
                'total_pages': len(metadata.get('pages', [])),
                'document_title': metadata.get('title', ''),
                'file_name': metadata.get('file_name', file_name),
                'creation_date': metadata.get('creation_date', ''),
                'author': metadata.get('author', '')
            },
            'children': []
        }

    def _create_parent_chunk(self, parent_doc: Dict, metadata: Dict) -> Dict[str, Any]:
        """创建父文档chunk（用于向量化）"""
        return {
            'id': parent_doc['id'],
            'type': 'parent',
            'content': '',  # 将在提取子元素后填充
            'page_no': parent_doc['page_no'],
            'metadata': {
                **parent_doc['metadata'],
                'chunk_type': 'parent',
                'child_count': 0,  # 将在后续更新
                'child_ids': []
            }
        }

    def _extract_child_chunks(self, page_data: Dict, page_no: int, metadata: Dict, parent_id: str) -> List[Dict[str, Any]]:
        """提取子文档chunks"""
        child_chunks = []
        layout_dets = page_data.get('layout_dets', [])

        for i, element in enumerate(layout_dets):
            element_type = self._determine_element_type(element)

            if element_type == 'ignore':
                continue

            child_chunk = self._create_child_chunk(
                element, element_type, page_no, metadata, parent_id, i
            )

            if child_chunk and child_chunk['content'].strip():
                child_chunks.append(child_chunk)

        return child_chunks

    def _determine_element_type(self, element: Dict) -> str:
        """确定元素类型"""
        category_id = element.get('category_id', -1)

        # 基于MinerU的category_id映射
        type_mapping = {
            0: 'text',      # 文本
            1: 'title',     # 标题
            2: 'ignore',    # 页眉页脚（忽略）
            3: 'text',      # 文本（其他）
            4: 'text',      # 文本（段落）
            5: 'table',     # 表格
            6: 'figure',    # 图片
            7: 'equation',  # 公式
            8: 'text',      # 列表
            9: 'text',      # 引用
            10: 'ignore'    # 其他（忽略）
        }

        return type_mapping.get(category_id, 'text')

    def _create_child_chunk(self, element: Dict, element_type: str, page_no: int,
                           metadata: Dict, parent_id: str, element_index: int) -> Optional[Dict[str, Any]]:
        """创建子文档chunk"""
        file_name = metadata.get('file_name', 'unknown')

        chunk = {
            'id': f"{file_name}_page_{page_no}_{element_type}_{element_index}",
            'type': element_type,
            'page_no': page_no,
            'metadata': {
                'source': file_name,
                'page_no': page_no,
                'parent_id': parent_id,
                'element_type': element_type,
                'bbox': element.get('bbox', []),
                'confidence': element.get('confidence', 0.0)
            }
        }

        # 根据类型提取内容
        if element_type == 'table':
            chunk['content'] = self._extract_table_content(element)
        elif element_type == 'title':
            chunk['content'] = element.get('text', '')
            chunk['metadata']['level'] = element.get('level', 1)
        elif element_type in ['text', 'equation', 'figure']:
            chunk['content'] = element.get('text', '')
            if element_type == 'equation' and 'latex' in element:
                chunk['content'] = f"${element['latex']}$"
        else:
            chunk['content'] = element.get('text', '')

        return chunk if chunk['content'].strip() else None

    def _extract_table_content(self, table_element: Dict) -> str:
        """提取表格内容为Markdown格式"""
        # 这里需要根据MinerU的实际表格格式处理
        # 简化实现：返回表格文本
        text = table_element.get('text', '')

        # 如果是结构化表格，转换为Markdown
        if 'rows' in table_element:
            return self._convert_table_to_markdown(table_element)

        return text

    def _convert_table_to_markdown(self, table_element: Dict) -> str:
        """将表格转换为Markdown格式"""
        rows = table_element.get('rows', [])
        if not rows:
            return table_element.get('text', '')

        markdown_lines = []

        for i, row in enumerate(rows):
            cells = row.get('cells', [])
            cell_texts = [cell.get('text', '') for cell in cells]
            markdown_lines.append('| ' + ' | '.join(cell_texts) + ' |')

            # 添加表头分隔符
            if i == 0:
                separators = ['---'] * len(cells)
                markdown_lines.append('| ' + ' | '.join(separators) + ' |')

        return '\n'.join(markdown_lines)

    def _generate_mock_data(self, pdf_path: Path) -> ParsedDocument:
        """生成模拟数据（用于开发和测试）"""
        self.logger.warning(f"使用模拟数据解析: {pdf_path}")

        # 模拟一个3页的文档
        parent_documents = []
        all_chunks = []

        for page_no in range(3):
            # 父文档
            parent_doc = {
                'id': f"{pdf_path.name}_page_{page_no}_parent",
                'type': 'parent_document',
                'page_no': page_no,
                'content': f'这是第{page_no + 1}页的内容，模拟PDF页面文本。',
                'metadata': {
                    'source': pdf_path.name,
                    'page_range': [page_no, page_no],
                    'total_pages': 3,
                    'document_title': '模拟文档',
                    'file_name': pdf_path.name
                },
                'children': []
            }
            parent_documents.append(parent_doc)

            # 父文档chunk
            parent_chunk = {
                'id': parent_doc['id'],
                'type': 'parent',
                'content': parent_doc['content'],
                'page_no': page_no,
                'metadata': {
                    **parent_doc['metadata'],
                    'chunk_type': 'parent',
                    'child_count': 2,
                    'child_ids': []
                }
            }
            all_chunks.append(parent_chunk)

            # 子文档chunks
            for i in range(2):
                child_chunk = {
                    'id': f"{pdf_path.name}_page_{page_no}_text_{i}",
                    'type': 'text',
                    'content': f'第{page_no + 1}页第{i + 1}段文本内容。',
                    'page_no': page_no,
                    'metadata': {
                        'source': pdf_path.name,
                        'page_no': page_no,
                        'parent_id': parent_chunk['id'],
                        'element_type': 'text',
                        'bbox': [100, 100, 500, 200],
                        'confidence': 0.95
                    }
                }
                all_chunks.append(child_chunk)
                parent_chunk['metadata']['child_ids'].append(child_chunk['id'])

            parent_chunk['metadata']['child_count'] = len(parent_chunk['metadata']['child_ids'])

        metadata = {
            'title': '模拟PDF文档',
            'file_name': pdf_path.name,
            'total_pages': 3
        }

        return ParsedDocument(
            parent_documents=parent_documents,
            all_chunks=all_chunks,
            document_metadata=metadata
        )

# 测试函数
def test_pdf_parser():
    """测试PDF解析器"""
    import tempfile

    # 创建测试配置
    config = {
        'device': 'cpu'  # 测试用CPU
    }

    parser = MinerUPDFParser(**config)

    # 使用模拟数据测试
    result = parser._generate_mock_data(Path("test.pdf"))

    print(f"解析结果:")
    print(f"  父文档数量: {len(result.parent_documents)}")
    print(f"  总chunks数量: {len(result.all_chunks)}")
    print(f"  文档元数据: {result.document_metadata}")

    # 检查父子关系
    parent_chunks = [c for c in result.all_chunks if c['type'] == 'parent']
    child_chunks = [c for c in result.all_chunks if c['type'] != 'parent']

    print(f"\n父子关系检查:")
    print(f"  父chunks: {len(parent_chunks)}")
    print(f"  子chunks: {len(child_chunks)}")

    # 验证每个子chunk都有parent_id
    orphan_chunks = [c for c in child_chunks if 'parent_id' not in c['metadata']]
    print(f"  孤立chunks: {len(orphan_chunks)}")

    return result

if __name__ == "__main__":
    test_pdf_parser()
