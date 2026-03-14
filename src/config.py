"""配置管理模块"""

import yaml
from pathlib import Path
from typing import Dict, Any
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """配置管理器"""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

        # 环境变量覆盖
        self._override_with_env()

    def _override_with_env(self):
        """用环境变量覆盖配置"""
        # API Keys
        if os.getenv("DASHSCOPE_API_KEY"):
            self._config["reranker"]["api_key"] = os.getenv("DASHSCOPE_API_KEY")

        if os.getenv("WEAVIATE_URL"):
            url = os.getenv("WEAVIATE_URL")
            if ":" in url:
                host, port = url.split(":")
                self._config["vector_store"]["host"] = host
                self._config["vector_store"]["port"] = int(port)

    def get(self, key_path: str, default=None) -> Any:
        """获取配置值，支持点分隔路径"""
        keys = key_path.split(".")
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    @property
    def pdf_parser(self) -> Dict[str, Any]:
        return self._config["pdf_parser"]

    @property
    def embedding(self) -> Dict[str, Any]:
        return self._config["embedding"]

    @property
    def vector_store(self) -> Dict[str, Any]:
        return self._config["vector_store"]

    @property
    def retrieval(self) -> Dict[str, Any]:
        return self._config["retrieval"]

    @property
    def reranker(self) -> Dict[str, Any]:
        return self._config["reranker"]

    @property
    def auto_tuner(self) -> Dict[str, Any]:
        return self._config["auto_tuner"]

    @property
    def gpu(self) -> Dict[str, Any]:
        return self._config["gpu"]

# 全局配置实例
config = Config()