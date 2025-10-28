import json
import time
import random
import asyncio
from typing import Dict, List, Optional
from loguru import logger

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available, LLM service will use fallback mode")

from app.utils.config import config
from app.models.input_models import NoteInput, TagPrediction


class LLMTaggingService:
    """LLM标签预测服务 - 使用OpenAI SDK with 重试机制"""
    
    def __init__(self):
        """初始化LLM标签服务"""
        self.api_key = config.OPENROUTER_API_KEY
        self.model = config.LLM_MODEL
        self.taxonomy = config.load_taxonomy_knowledge()
        
        # 重试配置
        self.max_retries = 3
        self.base_delay = 1.0  # 基础延迟（秒）
        self.max_delay = 60.0  # 最大延迟（秒）
        
        # 初始化OpenAI客户端
        if OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
            )
            logger.info(f"✅ Initialized OpenAI client for model: {self.model}")
            logger.info(f"OpenAI API key: {self.api_key}") 
        else:
            self.client = None
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI library not available")
            if not self.api_key:
                logger.warning("OpenRouter API key not configured")
            logger.warning("LLM service will use fallback mode (default tags)")
        
        logger.info(f"LLM tagging service initialized - Retry config: max={self.max_retries}, base_delay={self.base_delay}s")
    
    def _build_prompt(self, note: NoteInput) -> str:
        """构建LLM提示词"""
        prompt = f"""你是一个专业的内容标签分类助手。请根据小红书笔记的标题、内容为其预测6个维度的标签。

笔记信息：
标题：{note.title}
内容：{note.content}

请从以下每个维度中选择最合适的一个标签（如果没有合适的，可以留空）：

1. intention_lv1（一级意图）：
可选值：{', '.join(self.taxonomy.get('intention_lv1', [])[:30])}...

2. intention_lv2（二级意图）：
可选值：{', '.join(self.taxonomy.get('intention_lv2', [])[:30])}...

3. taxonomy1（一级分类）：
可选值：{', '.join(self.taxonomy.get('taxonomy1', [])[:50])}...

4. taxonomy2（二级分类）：
可选值：{', '.join(self.taxonomy.get('taxonomy2', [])[:300])}...

5. taxonomy3（三级分类）：
可选值：{', '.join(self.taxonomy.get('taxonomy3', [])[:500])}...

6. note_marketing_integrated_level（内容营销感）：
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
        预测笔记的标签 - 使用OpenAI SDK with 重试机制
        
        Args:
            note: 笔记输入
            
        Returns:
            标签预测结果
        """
        start_time = time.time()
        logger.info(f"🏷️ Starting LLM tag prediction for note: {note.note_id or 'unnamed'}")
        
        # 如果客户端不可用，直接返回默认标签
        if not self.client:
            logger.warning("LLM client not available, using default tags")
            return self._get_default_tags()
        
        # 构建prompt
        prompt = self._build_prompt(note)
        
        # 打印详细的prompt信息
        logger.info("=" * 60)
        logger.info("📝 LLM PROMPT")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model}")
        logger.info(f"Note ID: {note.note_id}")
        logger.info(f"Title: {note.title}")
        logger.info(f"Content: {note.content[:200]}...")
        #logger.info("--- Full Prompt ---")
        #logger.info(prompt)
        logger.info("=" * 60)
        
        # 进行重试调用
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"🔄 LLM API call attempt {attempt + 1}/{self.max_retries + 1}")
                
                # 调用OpenAI API
                response = await self._call_openai_api(prompt)
                
                # 解析响应
                result = self._parse_llm_response(response)
                
                # 记录成功信息
                elapsed_time = time.time() - start_time
                #logger.info(f"✅ LLM tag prediction successful in {elapsed_time:.2f}s after {attempt + 1} attempts")
                
                return result
                
            except Exception as e:
                error_type = type(e).__name__
                
                # 检查是否是可重试的错误
                if self._is_retryable_error(e) and attempt < self.max_retries:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(f"⚠️ LLM API call failed ({error_type}: {e}), retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    # 最终失败或不可重试的错误
                    elapsed_time = time.time() - start_time
                    logger.error(f"❌ LLM tag prediction failed after {attempt + 1} attempts in {elapsed_time:.2f}s")
                    logger.error(f"Final error: {error_type}: {e}")
                    return self._get_default_tags()
        
        # 应该不会到达这里，但为了安全
        return self._get_default_tags()
    
    async def _call_openai_api(self, prompt: str) -> str:
        """调用OpenAI API"""
        try:
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/xhs-ctr-project",
                    "X-Title": "XHS Content Tagging",
                },
                extra_body={},
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that classifies content accurately. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500,
                timeout=30.0
            )
            
            # 获取响应内容
            content = completion.choices[0].message.content
            
            # 打印详细的响应信息
            logger.info("=" * 60)
            logger.info("📨 LLM RESPONSE")
            logger.info("=" * 60)
            #logger.info(f"Model: {completion.model}")
            #logger.info(f"Usage: {completion.usage}")
            #logger.info(f"Finish reason: {completion.choices[0].finish_reason}")
            logger.info("--- Response Content ---")
            logger.info(content)
            logger.info("=" * 60)
            
            return content
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {type(e).__name__}: {e}")
            raise
    
    def _parse_llm_response(self, content: str) -> TagPrediction:
        """解析LLM响应内容"""
        try:
            # 清理可能的markdown代码块标记
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            # 解析JSON
            tags_dict = json.loads(content.strip())
            
            # 创建TagPrediction对象
            result = TagPrediction(
                intention_lv1=tags_dict.get('intention_lv1', ''),
                intention_lv2=tags_dict.get('intention_lv2', ''),
                taxonomy1=tags_dict.get('taxonomy1', ''),
                taxonomy2=tags_dict.get('taxonomy2', ''),
                taxonomy3=tags_dict.get('taxonomy3', ''),
                note_marketing_integrated_level=tags_dict.get('note_marketing_integrated_level', '')
            )
            
            #logger.info(f"📊 Parsed tags: {result.dict()}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Raw content: {content}")
            raise ValueError(f"Invalid JSON response: {e}")
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """判断错误是否可重试"""
        error_message = str(error).lower()
        
        # 可重试的错误类型
        retryable_errors = [
            '429',  # Too Many Requests
            'too many requests',
            'rate limit',
            'timeout',
            'connection',
            'temporary',
            'server error',
            '500',
            '502',
            '503',
            '504'
        ]
        
        for retryable in retryable_errors:
            if retryable in error_message:
                return True
        
        return False
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """计算指数退避延迟"""
        # 指数退避：base_delay * 2^attempt + 随机抖动
        delay = self.base_delay * (2 ** attempt)
        
        # 添加随机抖动（避免同时重试）
        jitter = random.uniform(0, delay * 0.1)
        delay += jitter
        
        # 限制最大延迟
        delay = min(delay, self.max_delay)
        
        return delay
    
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