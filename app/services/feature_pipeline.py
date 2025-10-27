import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
import asyncio
from typing import Dict, List, Optional, Any
from loguru import logger

# 添加离线训练代码到Python路径
offline_path = Path(__file__).parent.parent.parent / "offline_training"
sys.path.insert(0, str(offline_path))


try:
    from offline_training.pipelines.multimodal_processors import ChineseCLIPProcessor, OCRProcessor, ImageDownloader, cleanup_memory
    OFFLINE_PROCESSORS_AVAILABLE = True
    logger.info("Successfully imported offline training processors")
except ImportError as e:
    OFFLINE_PROCESSORS_AVAILABLE = False
    logger.warning(f"Failed to import offline training processors: {e}")

from app.models.input_models import NoteInput, TagPrediction
from app.utils.image_utils import decode_image


def extract_hashtag_keywords(content: str) -> str:
    """从content中提取#话题关键词（对齐离线训练的tag提取逻辑）
    
    Args:
        content: 笔记内容文本
        
    Returns:
        提取的hashtag关键词，用空格分隔
    """
    if not content:
        return ""
    
    try:
        # 使用正则表达式提取#话题
        hashtag_pattern = r'#([^#\s]+)'
        hashtags = re.findall(hashtag_pattern, content)
        
        # 清理和过滤hashtags
        cleaned_hashtags = []
        for tag in hashtags:
            # 移除特殊字符，保留中文、英文、数字
            clean_tag = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', tag)
            if clean_tag and len(clean_tag) > 1:  # 至少2个字符
                cleaned_hashtags.append(clean_tag)
        
        return ' '.join(cleaned_hashtags) if cleaned_hashtags else ""
        
    except Exception as e:
        logger.debug(f"Failed to extract hashtag keywords: {e}")
        return ""


class FeaturePipeline:
    """特征工程Pipeline - 复用离线训练代码"""
    
    def __init__(self):
        """初始化特征工程pipeline"""
        self.clip_processor = None
        self.ocr_processor = None
        self.image_downloader = None
        
        if OFFLINE_PROCESSORS_AVAILABLE:
            try:
                # 初始化Chinese-CLIP处理器
                self.clip_processor = ChineseCLIPProcessor(
                    model_name="ViT-B-16",
                    batch_size=8,
                    target_dim=512
                )
                logger.info("Chinese-CLIP processor initialized")
                
                # 初始化OCR处理器
                self.ocr_processor = OCRProcessor()
                logger.info(f"OCR processor initialized, enabled: {self.ocr_processor.enabled}")
                
                # 初始化图片下载器
                self.image_downloader = ImageDownloader(num_workers=5, timeout=15)
                logger.info("Image downloader initialized")
                
                logger.info("✅ Feature pipeline with offline processors initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize offline processors: {e}")
                self.clip_processor = None
                self.ocr_processor = None
                self.image_downloader = None
        else:
            logger.warning("Offline processors not available, using basic feature extraction")
    
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
            特征字典，包含原始tags信息
        """
        try:
            features = {}
            
            # 1. 基础文本特征
            features.update(self._extract_text_features(note))
            
            # 2. 标签特征（包含hashtag关键词提取）
            features.update(self._extract_tag_features(note, tags))
            
            # 3. 图像特征（OCR、CN-CLIP等）
            features.update(await self._extract_image_features(note))
            
            # 4. 添加原始标签信息（保留sparse特征）
            features.update({
                'note_id': note.note_id,
                'original_tags': {
                    'intention_lv1': tags.intention_lv1,
                    'intention_lv2': tags.intention_lv2,
                    'taxonomy1': tags.taxonomy1,
                    'taxonomy2': tags.taxonomy2,
                    'taxonomy3': tags.taxonomy3,
                    'note_marketing_integrated_level': tags.note_marketing_integrated_level,
                }
            })
            
            # 5. 多模态融合特征
            # TODO: 调用离线训练的multimodal pipeline
            
            logger.debug(f"Feature extraction completed for note {note.note_id}: {len(features)} features")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return self._get_default_features()
    
    def _extract_text_features(self, note: NoteInput) -> Dict[str, Any]:
        """提取文本特征"""
        features = {
            'title_length': len(note.title),
            'content_length': len(note.content),
            'title_clean': note.title,
            'content': note.content,
        }
        
        # 提取CLIP文本特征
        if self.clip_processor:
            try:
                # 处理标题
                if note.title:
                    title_features = self.clip_processor.process_texts([note.title])
                    for i, feat in enumerate(title_features[0]):
                        features[f'title_feat_{i}'] = float(feat)
                    logger.debug(f"Extracted title CLIP features: {title_features.shape}")
                else:
                    # 空标题使用零向量
                    for i in range(512):
                        features[f'title_feat_{i}'] = 0.0
                
                # 处理内容
                if note.content:
                    content_features = self.clip_processor.process_long_content([note.content])
                    for i, feat in enumerate(content_features[0]):
                        features[f'content_feat_{i}'] = float(feat)
                    logger.debug(f"Extracted content CLIP features: {content_features.shape}")
                else:
                    # 空内容使用零向量
                    for i in range(512):
                        features[f'content_feat_{i}'] = 0.0
                
            except Exception as e:
                logger.error(f"CLIP text feature extraction failed: {e}")
                # 使用零向量
                for i in range(512):
                    features[f'title_feat_{i}'] = 0.0
                    features[f'content_feat_{i}'] = 0.0
        else:
            # 无CLIP处理器时使用零向量
            for i in range(512):
                features[f'title_feat_{i}'] = 0.0
                features[f'content_feat_{i}'] = 0.0
        
        return features
    
    def _extract_tag_features(self, note: NoteInput, tags: TagPrediction) -> Dict[str, Any]:
        """提取标签特征 - 正确提取hashtag关键词而非LLM预测标签"""
        features = {}
        
        # 1. 存储LLM预测的分类标签（用于稀疏特征）
        features.update({
            'intention_lv1': tags.intention_lv1,
            'intention_lv2': tags.intention_lv2,
            'taxonomy1': tags.taxonomy1,
            'taxonomy2': tags.taxonomy2,
            'taxonomy3': tags.taxonomy3,
            'note_marketing_integrated_level': tags.note_marketing_integrated_level,
        })
        
        # 2. 从content中提取hashtag关键词（对齐离线训练的tag特征）
        hashtag_keywords = extract_hashtag_keywords(note.content)
        features['tag_keywords'] = hashtag_keywords
        
        # 3. 对hashtag关键词提取CLIP特征（这才是真正的tag_feat）
        if self.clip_processor:
            try:
                if hashtag_keywords and hashtag_keywords.strip():
                    # 使用hashtag关键词提取CLIP特征
                    tag_features = self.clip_processor.process_texts([hashtag_keywords])
                    for i, feat in enumerate(tag_features[0]):
                        features[f'tag_feat_{i}'] = float(feat)
                    logger.debug(f"Extracted tag CLIP features from hashtag keywords: '{hashtag_keywords}'")
                else:
                    # 无hashtag关键词时使用零向量
                    for i in range(512):
                        features[f'tag_feat_{i}'] = 0.0
                    logger.debug("No hashtag keywords found, using zero vector for tag features")
                
            except Exception as e:
                logger.error(f"Tag CLIP feature extraction failed: {e}")
                # 使用零向量
                for i in range(512):
                    features[f'tag_feat_{i}'] = 0.0
        else:
            # 无CLIP处理器时使用零向量
            for i in range(512):
                features[f'tag_feat_{i}'] = 0.0
        
        return features
    
    async def _extract_image_features(self, note: NoteInput) -> Dict[str, Any]:
        """提取图像特征 - 对齐multimodal_pipeline的处理流程"""
        features = {}
        
        try:
            # 按照multimodal_pipeline的方式：先收集URLs，再统一下载，最后处理bytes
            cover_urls = []
            inner_image_urls = []
            cover_image_bytes = []
            inner_images_bytes_list = []
            
            # 1. 收集封面图URL和base64数据
            if note.cover_image:
                if note.cover_image.startswith(('http://', 'https://')):
                    cover_urls.append(note.cover_image)
                    features['has_cover_image'] = 1
                else:
                    # base64图片直接解码为bytes
                    cover_img = decode_image(note.cover_image)
                    if cover_img:
                        import io
                        img_bytes = io.BytesIO()
                        cover_img.save(img_bytes, format='PNG')
                        cover_image_bytes.append(img_bytes.getvalue())
                        features['has_cover_image'] = 1
                    else:
                        features['has_cover_image'] = 0
            else:
                features['has_cover_image'] = 0
            
            # 2. 收集内部图片URLs和base64数据
            features['num_inner_images'] = len(note.inner_images) if note.inner_images else 0
            for inner_img in (note.inner_images or []):
                if inner_img.startswith(('http://', 'https://')):
                    inner_image_urls.append(inner_img)
                else:
                    inner_img_pil = decode_image(inner_img)
                    if inner_img_pil:
                        import io
                        img_bytes = io.BytesIO()
                        inner_img_pil.save(img_bytes, format='PNG')
                        inner_images_bytes_list.append(img_bytes.getvalue())
            
            # 3. 统一下载URL图片（对齐multimodal_pipeline）
            if self.image_downloader:
                # 下载封面图
                if cover_urls:
                    logger.info(f"Downloading {len(cover_urls)} cover images from URLs")
                    downloaded_cover = await self.image_downloader.download_batch(cover_urls)
                    cover_image_bytes.extend(downloaded_cover)
                
                # 下载内部图片
                if inner_image_urls:
                    logger.info(f"Downloading {len(inner_image_urls)} inner images from URLs")
                    downloaded_inner = await self.image_downloader.download_batch(inner_image_urls)
                    inner_images_bytes_list.extend(downloaded_inner)
            
            # 4. 提取CLIP特征（对齐multimodal_pipeline的处理方式）
            if self.clip_processor:
                # 处理封面图特征
                if cover_image_bytes:
                    # 过滤None值
                    valid_cover_images = [img for img in cover_image_bytes if img is not None]
                    if valid_cover_images:
                        logger.info(f"Extracting CLIP features from {len(valid_cover_images)} cover images")
                        cover_features = self.clip_processor.process_cover_images(valid_cover_images)
                        for i, feat in enumerate(cover_features[0]):
                            features[f'cover_image_feat_{i}'] = float(feat)
                    else:
                        # 无有效封面图时使用零向量
                        for i in range(512):
                            features[f'cover_image_feat_{i}'] = 0.0
                else:
                    # 无封面图时使用零向量
                    for i in range(512):
                        features[f'cover_image_feat_{i}'] = 0.0
                
                # 处理内部图片特征
                if inner_images_bytes_list:
                    # 过滤None值
                    valid_inner_images = [img for img in inner_images_bytes_list if img is not None]
                    if valid_inner_images:
                        logger.info(f"Extracting CLIP features from {len(valid_inner_images)} inner images")
                        inner_features, _ = self.clip_processor.process_inner_images_batch(
                            [valid_inner_images], pooling_strategy="mean"
                        )
                        for i, feat in enumerate(inner_features[0]):
                            features[f'inner_image_feat_{i}'] = float(feat)
                    else:
                        # 无有效内部图片时使用零向量
                        for i in range(512):
                            features[f'inner_image_feat_{i}'] = 0.0
                else:
                    # 无内部图片时使用零向量
                    for i in range(512):
                        features[f'inner_image_feat_{i}'] = 0.0
            
            # 5. OCR文字提取（对齐multimodal_pipeline的分离处理）
            if self.ocr_processor and self.ocr_processor.enabled:
                # 处理封面图OCR
                cover_ocr_texts = []
                cover_ocr_confidences = []
                if cover_image_bytes:
                    cover_ocr_texts, cover_ocr_confidences = self.ocr_processor.extract_batch_texts(cover_image_bytes)
                
                # 处理内部图片OCR
                inner_ocr_texts = []
                inner_ocr_confidences = []
                if inner_images_bytes_list:
                    inner_ocr_texts, inner_ocr_confidences = self.ocr_processor.extract_inner_images_ocr([inner_images_bytes_list])
                    # extract_inner_images_ocr返回的是列表的列表，需要展平
                    if inner_ocr_texts and len(inner_ocr_texts) > 0:
                        inner_ocr_texts = inner_ocr_texts[0] if isinstance(inner_ocr_texts[0], list) else inner_ocr_texts
                        inner_ocr_confidences = inner_ocr_confidences[0] if isinstance(inner_ocr_confidences[0], list) else inner_ocr_confidences
                
                # 分别存储OCR结果（对齐multimodal_pipeline）
                features['cover_image_ocr_texts'] = ' '.join(cover_ocr_texts) if cover_ocr_texts else ''
                features['inner_images_ocr_texts'] = ' '.join(inner_ocr_texts) if inner_ocr_texts else ''
                features['cover_image_ocr_confidences'] = cover_ocr_confidences[0] if cover_ocr_confidences else 0.0
                features['inner_images_ocr_confidences'] = inner_ocr_confidences[0] if inner_ocr_confidences else 0.0
                
                logger.info(f"Extracted OCR text - Cover: {len(cover_ocr_texts)} segments, Inner: {len(inner_ocr_texts)} segments")
            else:
                features['cover_image_ocr_texts'] = ''
                features['inner_images_ocr_texts'] = ''
                features['cover_image_ocr_confidences'] = 0.0
                features['inner_images_ocr_confidences'] = 0.0
            
        except Exception as e:
            logger.error(f"Image feature extraction failed: {e}", exc_info=True)
            # 返回默认特征
            features['has_cover_image'] = 0
            features['num_inner_images'] = 0
            features['ocr_text'] = ''
            features['ocr_text_length'] = 0
            # 默认零向量
            for i in range(512):
                features[f'cover_image_feat_{i}'] = 0.0
                features[f'inner_image_feat_{i}'] = 0.0
        
        return features
    
    def _get_default_features(self) -> Dict[str, Any]:
        """获取默认特征（用于错误情况）"""
        features = {
            # 基础特征
            'title_length': 0,
            'content_length': 0,
            'title_clean': '',
            'content': '',
            'has_cover_image': 0,
            'num_inner_images': 0,
            'ocr_text': '',
            'ocr_text_length': 0,
            
            # 标签特征
            'intention_lv1': '',
            'intention_lv2': '',
            'taxonomy1': '',
            'taxonomy2': '',
            'taxonomy3': '',
            'note_marketing_integrated_level': '',
        }
        
        # CLIP特征（零向量）
        for i in range(512):
            features[f'title_feat_{i}'] = 0.0
            features[f'content_feat_{i}'] = 0.0
            features[f'cover_image_feat_{i}'] = 0.0
            features[f'inner_image_feat_{i}'] = 0.0
            features[f'tag_feat_{i}'] = 0.0
        
        return features
    
    def transform_to_model_input(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        将特征字典转换为模型输入格式
        
        Args:
            features: 特征字典
            
        Returns:
            处理后的特征字典（保持原格式，供模型推理使用）
        """
        # 确保所有CLIP特征都存在
        feature_types = ['title_feat', 'content_feat', 'cover_image_feat', 'inner_image_feat', 'tag_feat']
        
        for feat_type in feature_types:
            for i in range(512):
                feat_name = f'{feat_type}_{i}'
                if feat_name not in features:
                    features[feat_name] = 0.0
        
        # 确保基础特征存在
        basic_features = {
            'title_length': 0,
            'content_length': 0,
            'has_cover_image': 0,
            'num_inner_images': 0,
            'ocr_text_length': 0
        }
        
        for key, default_value in basic_features.items():
            if key not in features:
                features[key] = default_value
        
        logger.debug(f"Transformed features ready with {len(features)} dimensions")
        return features
    
    async def cleanup(self):
        """清理资源"""
        try:
            if self.image_downloader:
                await self.image_downloader.close()
                logger.info("Image downloader session closed")
            
            if OFFLINE_PROCESSORS_AVAILABLE:
                cleanup_memory()
                logger.info("Memory cleanup completed")
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'image_downloader') and self.image_downloader:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.cleanup())
                else:
                    loop.run_until_complete(self.cleanup())
            except Exception:
                pass