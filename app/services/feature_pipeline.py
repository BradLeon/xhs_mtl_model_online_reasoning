import sys
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
            
            # TODO, 丢失了 tag等原始sparse特征
            
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
    
    def _extract_tag_features(self, tags: TagPrediction) -> Dict[str, Any]:
        """提取标签特征"""
        # TODO, 此tag非彼tag。 离线训练的tag是content末尾的#话题。 
        features = {
            'intention_lv1': tags.intention_lv1,
            'intention_lv2': tags.intention_lv2,
            'taxonomy1': tags.taxonomy1,
            'taxonomy2': tags.taxonomy2,
            'taxonomy3': tags.taxonomy3,
            'note_marketing_integrated_level': tags.note_marketing_integrated_level,
        }
        
        # 提取标签的CLIP特征
        if self.clip_processor:
            try:
                # 合并所有标签为文本
                tag_texts = [
                    tags.intention_lv1,
                    tags.intention_lv2, 
                    tags.taxonomy1,
                    tags.taxonomy2,
                    tags.taxonomy3,
                    tags.note_marketing_integrated_level
                ]
                # 过滤空标签
                valid_tags = [tag for tag in tag_texts if tag and tag.strip()]
                
                if valid_tags:
                    # 组合标签文本
                    combined_tags = ' '.join(valid_tags)
                    tag_features = self.clip_processor.process_texts([combined_tags])
                    for i, feat in enumerate(tag_features[0]):
                        features[f'tag_feat_{i}'] = float(feat)
                    logger.debug(f"Extracted tag CLIP features from {len(valid_tags)} tags")
                else:
                    # 无有效标签时使用零向量
                    for i in range(512):
                        features[f'tag_feat_{i}'] = 0.0
                
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
        """提取图像特征"""
        features = {}
        
        try:
            # 准备图片数据
            image_urls = []
            image_bytes_list = []
            
            # 处理封面图
            # TODO: 为了什么和multimodal_pipeline不同？  为什么要先decode再download？ 
            if note.cover_image:
                if note.cover_image.startswith(('http://', 'https://')):
                    image_urls.append(note.cover_image)
                else:
                    # base64图片直接解码
                    cover_img = decode_image(note.cover_image)
                    if cover_img:
                        import io
                        img_bytes = io.BytesIO()
                        cover_img.save(img_bytes, format='PNG')
                        image_bytes_list.append(img_bytes.getvalue())
                        features['has_cover_image'] = 1
                    else:
                        features['has_cover_image'] = 0
                        image_bytes_list.append(None)
            else:
                features['has_cover_image'] = 0
                image_bytes_list.append(None)
            
            # 处理内部图片
            features['num_inner_images'] = len(note.inner_images) if note.inner_images else 0
            for inner_img in (note.inner_images or []):
                if inner_img.startswith(('http://', 'https://')):
                    image_urls.append(inner_img)
                else:
                    inner_img_pil = decode_image(inner_img)
                    if inner_img_pil:
                        import io
                        img_bytes = io.BytesIO()
                        inner_img_pil.save(img_bytes, format='PNG')
                        image_bytes_list.append(img_bytes.getvalue())
                    else:
                        image_bytes_list.append(None)
            
            # 下载URL图片
            if image_urls and self.image_downloader:
                logger.info(f"Downloading {len(image_urls)} images from URLs")
                downloaded_images = await self.image_downloader.download_batch(image_urls)
                image_bytes_list.extend(downloaded_images)
            
            # 提取CLIP特征
            if self.clip_processor and image_bytes_list:
                # 过滤None值
                valid_images = [img for img in image_bytes_list if img is not None]
                
                if valid_images:
                    logger.info(f"Extracting CLIP features from {len(valid_images)} images")
                    
                    # 封面图特征
                    if len(valid_images) > 0:
                        cover_features = self.clip_processor.process_cover_images([valid_images[0]])
                        for i, feat in enumerate(cover_features[0]):
                            features[f'cover_image_feat_{i}'] = float(feat)
                    else:
                        # 无图片时使用零向量
                        for i in range(512):
                            features[f'cover_image_feat_{i}'] = 0.0
                    
                    # 内部图片特征（如果有）
                    if len(valid_images) > 1:
                        inner_images = valid_images[1:]
                        inner_features, _ = self.clip_processor.process_inner_images_batch(inner_images, pooling_strategy="mean")
                        for i, feat in enumerate(inner_features[0]):
                            features[f'inner_image_feat_{i}'] = float(feat)
                    else:
                        # 无内部图片时使用零向量
                        for i in range(512):
                            features[f'inner_image_feat_{i}'] = 0.0
                else:
                    logger.warning("No valid images found for CLIP processing")
                    # 使用零向量
                    for i in range(512):
                        features[f'cover_image_feat_{i}'] = 0.0
                        features[f'inner_image_feat_{i}'] = 0.0
            
            # OCR文字提取
            if self.ocr_processor and self.ocr_processor.enabled and image_bytes_list:
                ocr_texts = []
                for img_bytes in image_bytes_list:
                    if img_bytes:
                        text, confidence = self.ocr_processor.extract_text(img_bytes)
                        if text and confidence > 0.5:
                            ocr_texts.append(text)
            # TODO, 离线训练pipeline, cover_image_ocr_texts , inner_images_ocr_texts，而不是ocr_text

                features['ocr_text'] = ' '.join(ocr_texts)
                features['ocr_text_length'] = len(features['ocr_text'])
                logger.info(f"Extracted OCR text: {len(ocr_texts)} segments")
            else:
                features['ocr_text'] = ''
                features['ocr_text_length'] = 0
            
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