import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from loguru import logger

# 添加离线训练代码到Python路径
offline_path = Path(__file__).parent.parent.parent / "offline_training"
sys.path.insert(0, str(offline_path))

from app.models.input_models import NoteInput, TagPrediction
from app.utils.image_utils import decode_image


class FeaturePipeline:
    """特征工程Pipeline - 复用离线训练代码"""
    
    def __init__(self):
        """初始化特征工程pipeline"""
        self.pipeline = None
        self.preprocessors = None
        
        try:
            # TODO: 导入离线训练的pipeline
            # from pipelines.multimodal_pipeline import MultimodalPipeline
            # self.pipeline = MultimodalPipeline()
            
            logger.info("Feature pipeline initialized (placeholder)")
        except Exception as e:
            logger.error(f"Failed to initialize feature pipeline: {e}")
    
    async def extract_features(
        self, 
        note: NoteInput, 
        tags: TagPrediction
    ) -> Dict[str, Any]:
        """
        提取笔记特征
        
        Args:
            note: 笔记输入
            tags: LLM预测的标签
            
        Returns:
            特征字典
        """
        try:
            features = {}
            
            # 1. 基础文本特征
            features.update(self._extract_text_features(note))
            
            # 2. 标签特征
            features.update(self._extract_tag_features(tags))
            
            # 3. 图像特征（OCR、CN-CLIP等）
            features.update(await self._extract_image_features(note))
            
            # 4. 多模态融合特征
            # TODO: 调用离线训练的multimodal pipeline
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return self._get_default_features()
    
    def _extract_text_features(self, note: NoteInput) -> Dict[str, Any]:
        """提取文本特征"""
        features = {
            'title_length': len(note.title),
            'content_length': len(note.content),
            'title': note.title,
            'content': note.content,
            # TODO: 添加更多文本特征
        }
        return features
    
    def _extract_tag_features(self, tags: TagPrediction) -> Dict[str, Any]:
        """提取标签特征"""
        features = {
            'intention_lv1': tags.intention_lv1,
            'intention_lv2': tags.intention_lv2,
            'taxonomy1': tags.taxonomy1,
            'taxonomy2': tags.taxonomy2,
            'taxonomy3': tags.taxonomy3,
            'note_marketing_integrated_level': tags.note_marketing_integrated_level,
        }
        return features
    
    async def _extract_image_features(self, note: NoteInput) -> Dict[str, Any]:
        """提取图像特征"""
        features = {}
        
        try:
            # 处理封面图
            if note.cover_image:
                cover_img = decode_image(note.cover_image)
                if cover_img:
                    # TODO: 调用OCR
                    # TODO: 调用CN-CLIP
                    features['has_cover_image'] = 1
                else:
                    features['has_cover_image'] = 0
            else:
                features['has_cover_image'] = 0
            
            # 处理内部图片
            features['num_inner_images'] = len(note.inner_images) if note.inner_images else 0
            
            # TODO: 处理inner_images的OCR和CN-CLIP特征
            
        except Exception as e:
            logger.error(f"Image feature extraction failed: {e}")
        
        return features
    
    def _get_default_features(self) -> Dict[str, Any]:
        """获取默认特征（用于错误情况）"""
        # TODO: 根据实际模型输入要求构建默认特征
        return {
            'title_length': 0,
            'content_length': 0,
            'has_cover_image': 0,
            'num_inner_images': 0,
            # 添加其他必需的特征
        }
    
    def transform_to_model_input(self, features: Dict[str, Any]) -> np.ndarray:
        """
        将特征字典转换为模型输入格式
        
        Args:
            features: 特征字典
            
        Returns:
            模型输入数组
        """
        # TODO: 根据实际模型输入要求转换特征
        # 这里需要与离线训练的预处理器保持一致
        
        # 临时返回随机数组
        return np.random.randn(1, 100)  # 假设模型输入是100维