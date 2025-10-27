from pydantic import BaseModel, Field
from typing import List, Optional


class NoteInput(BaseModel):
    """小红书笔记输入模型"""
    title: str = Field(..., description="笔记标题")
    cover_image: str = Field(..., description="封面图片(base64编码或URL)")
    content: str = Field(..., description="笔记内容")
    inner_images: Optional[List[str]] = Field(default=[], description="内部图片列表(base64编码或URL)")
    note_id: Optional[str] = Field(default=None, description="笔记ID(可选，用于批量预测时标识)")


class BatchNoteInput(BaseModel):
    """批量笔记输入模型"""
    notes: List[NoteInput] = Field(..., description="笔记列表")


class PredictionOutput(BaseModel):
    """预测结果输出模型"""
    note_id: Optional[str] = Field(default=None, description="笔记ID")
    ctr: float = Field(..., description="点击率预测值")
    like_rate: float = Field(..., description="点赞率预测值")
    fav_rate: float = Field(..., description="收藏率预测值")
    comment_rate: float = Field(..., description="评论率预测值")
    share_rate: float = Field(..., description="分享率预测值")
    follow_rate: float = Field(..., description="关注率预测值")
    interaction_rate: float = Field(..., description="互动率预测值")
    ces_rate: float = Field(..., description="CES率预测值")
    impression: float = Field(..., description="曝光量预测值")
    sort_score2: float = Field(..., description="排序分数2预测值")


class BatchPredictionOutput(BaseModel):
    """批量预测结果输出模型"""
    predictions: List[PredictionOutput] = Field(..., description="预测结果列表")
    total_notes: int = Field(..., description="总笔记数")


class HealthCheckResponse(BaseModel):
    """健康检查响应模型"""
    status: str = Field(default="healthy", description="服务状态")
    model_loaded: bool = Field(..., description="模型是否已加载")
    version: str = Field(default="1.0.0", description="API版本")


class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str = Field(..., description="错误信息")
    detail: Optional[str] = Field(default=None, description="详细错误信息")
    code: Optional[int] = Field(default=500, description="错误码")


class TagPrediction(BaseModel):
    """LLM标签预测结果"""
    intention_lv1: str = Field(..., description="一级意图")
    intention_lv2: str = Field(..., description="二级意图")
    taxonomy1: str = Field(..., description="一级分类")
    taxonomy2: str = Field(..., description="二级分类")
    taxonomy3: str = Field(..., description="三级分类")
    note_marketing_integrated_level: str = Field(..., description="内容营销感")