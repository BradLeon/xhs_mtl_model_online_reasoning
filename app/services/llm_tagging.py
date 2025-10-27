import json
import httpx
from typing import Dict, List, Optional
from loguru import logger

from app.utils.config import config
from app.models.input_models import NoteInput, TagPrediction


class LLMTaggingService:
    """LLM标签预测服务"""
    
    def __init__(self):
        """初始化LLM标签服务"""
        self.api_key = config.OPENROUTER_API_KEY
        self.api_url = config.OPENROUTER_API_URL
        self.model = config.LLM_MODEL
        self.taxonomy = config.load_taxonomy_knowledge()
        
        if not self.api_key:
            logger.warning("OpenRouter API key not configured")
        
        logger.info(f"Initialized LLM tagging service with model: {self.model}")
    
    def _build_prompt(self, note: NoteInput) -> str:
        """构建LLM提示词"""
        prompt = f"""你是一个专业的内容标签分类助手。请根据小红书笔记的标题、内容为其预测6个维度的标签。

笔记信息：
标题：{note.title}
内容：{note.content}

请从以下每个维度中选择最合适的一个标签（如果没有合适的，可以留空）：

1. intention_lv1（一级意图）：
可选值：{', '.join(self.taxonomy.get('intention_lv1', [])[:20])}...

2. intention_lv2（二级意图）：
可选值：{', '.join(self.taxonomy.get('intention_lv2', [])[:20])}...

3. taxonomy1（一级分类）：
可选值：{', '.join(self.taxonomy.get('taxonomy1', [])[:20])}...

4. taxonomy2（二级分类）：
可选值：{', '.join(self.taxonomy.get('taxonomy2', [])[:30])}...

5. taxonomy3（三级分类）：
可选值：{', '.join(self.taxonomy.get('taxonomy3', [])[:30])}...

6. note_marketing_integrated_level（营销集成度）：
可选值：{', '.join(self.taxonomy.get('note_marketing_integrated_level', []))}

请以JSON格式返回预测结果，格式如下：
{{
    "intention_lv1": "选择的标签",
    "intention_lv2": "选择的标签",
    "taxonomy1": "选择的标签",
    "taxonomy2": "选择的标签",
    "taxonomy3": "选择的标签",
    "note_marketing_integrated_level": "选择的标签"
}}

注意：
- 每个维度只选择一个最合适的标签
- 如果某个维度没有合适的标签，可以返回空字符串
- 请仅返回JSON格式，不要包含其他说明文字
"""
        return prompt
    
    async def predict_tags(self, note: NoteInput) -> TagPrediction:
        """
        预测笔记的标签
        
        Args:
            note: 笔记输入
            
        Returns:
            标签预测结果
        """
        try:
            if not self.api_key:
                # 如果没有API key，返回默认标签
                logger.warning("Using default tags due to missing API key")
                return self._get_default_tags()
            
            prompt = self._build_prompt(note)
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "https://github.com/xhs-ctr-project",
                        "X-Title": "XHS Content Tagging"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant that classifies content."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 500
                    }
                )
                
                response.raise_for_status()
                result = response.json()
                
                # 解析LLM返回的内容
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '{}')
                
                # 尝试解析JSON
                try:
                    # 清理可能的markdown代码块标记
                    if '```json' in content:
                        content = content.split('```json')[1].split('```')[0]
                    elif '```' in content:
                        content = content.split('```')[1].split('```')[0]
                    
                    tags_dict = json.loads(content.strip())
                    
                    # 验证并填充缺失的字段
                    return TagPrediction(
                        intention_lv1=tags_dict.get('intention_lv1', ''),
                        intention_lv2=tags_dict.get('intention_lv2', ''),
                        taxonomy1=tags_dict.get('taxonomy1', ''),
                        taxonomy2=tags_dict.get('taxonomy2', ''),
                        taxonomy3=tags_dict.get('taxonomy3', ''),
                        note_marketing_integrated_level=tags_dict.get('note_marketing_integrated_level', '')
                    )
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM response as JSON: {e}, content: {content}")
                    return self._get_default_tags()
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error during LLM tagging: {e}")
            return self._get_default_tags()
        except Exception as e:
            logger.error(f"Unexpected error during LLM tagging: {e}")
            return self._get_default_tags()
    
    def _get_default_tags(self) -> TagPrediction:
        """获取默认标签（用于错误情况或测试）"""
        return TagPrediction(
            intention_lv1="分享",
            intention_lv2="分享日常",
            taxonomy1="生活记录",
            taxonomy2="",
            taxonomy3="",
            note_marketing_integrated_level="生活记录"
        )
    
    def validate_tags(self, tags: TagPrediction) -> bool:
        """
        验证标签是否有效
        
        Args:
            tags: 标签预测结果
            
        Returns:
            是否有效
        """
        # TODO: 实现标签验证逻辑
        # 检查标签是否在允许的范围内
        return True