import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from typing import Dict, List, Optional

# 加载环境变量
load_dotenv()


class Config:
    """应用配置管理"""
    
    # API配置
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # OpenRouter API配置
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
    LLM_MODEL = "google/gemini-2.5-flash-lite"  # 使用免费模型进行测试
    
    # 模型路径配置 - 使用项目内相对路径
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    MODEL_PATH = os.getenv(
        "MODEL_PATH",
        str(PROJECT_ROOT / "models" / "pnn_mmoe_model.pth")
    )
    PREPROCESSOR_PATH = os.getenv(
        "PREPROCESSOR_PATH", 
        str(PROJECT_ROOT / "models" / "pnn_mmoe_preprocessors.pkl")
    )
    
    # 离线训练代码路径
    OFFLINE_TRAINING_PATH = Path(__file__).parent.parent.parent / "offline_training"
    
    # Taxonomy知识库路径
    TAXONOMY_CSV_PATH = "/Users/liuchao/AI/xhs-ctr-project/xhs_mtl_model_offline_training/docs/taxonomy_knowledge.csv"
    
    # 缓存配置
    ENABLE_CACHE = True
    CACHE_TTL = 3600  # 1小时
    
    @classmethod
    def validate(cls):
        """验证配置"""
        errors = []
        
        if not cls.OPENROUTER_API_KEY:
            errors.append("OPENROUTER_API_KEY not set")
        
        if not Path(cls.MODEL_PATH).exists():
            errors.append(f"Model file not found: {cls.MODEL_PATH}")
        
        if not Path(cls.PREPROCESSOR_PATH).exists():
            errors.append(f"Preprocessor file not found: {cls.PREPROCESSOR_PATH}")
        
        if not Path(cls.TAXONOMY_CSV_PATH).exists():
            errors.append(f"Taxonomy CSV file not found: {cls.TAXONOMY_CSV_PATH}")
        
        if errors:
            for error in errors:
                logger.error(error)
            return False
        
        return True
    
    @classmethod
    def load_taxonomy_knowledge(cls) -> Dict[str, List[str]]:
        """加载分类知识库"""
        try:
            df = pd.read_csv(cls.TAXONOMY_CSV_PATH)
            
            taxonomy = {}
            for col in df.columns:
                # 获取非空的唯一值
                unique_values = df[col].dropna().unique().tolist()
                # 过滤空字符串
                unique_values = [v for v in unique_values if v and str(v).strip()]
                taxonomy[col] = unique_values
            
            logger.info(f"Loaded taxonomy with {len(taxonomy)} categories")
            for key, values in taxonomy.items():
                logger.info(f"  {key}: {len(values)} unique values")
            
            return taxonomy
            
        except Exception as e:
            logger.error(f"Failed to load taxonomy knowledge: {e}")
            return {}


# 初始化配置
config = Config()