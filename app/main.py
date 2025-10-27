from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from loguru import logger

from app.models.input_models import (
    NoteInput, 
    BatchNoteInput,
    PredictionOutput,
    BatchPredictionOutput,
    HealthCheckResponse,
    ErrorResponse
)
from app.services.llm_tagging import LLMTaggingService
from app.services.feature_pipeline import FeaturePipeline
from app.services.model_inference import ModelInferenceService


# 全局变量存储模型和服务
model_service = None
llm_service = None
feature_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("Starting up the application...")
    
    # 初始化各个服务
    global model_service, llm_service, feature_service
    
    try:
        model_service = ModelInferenceService()
        llm_service = LLMTaggingService()
        feature_service = FeaturePipeline()
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
    
    yield
    
    logger.info("Shutting down the application...")
    # 清理资源
    model_service = None
    llm_service = None
    feature_service = None
    

# 创建FastAPI应用
app = FastAPI(
    title="小红书笔记多维度预测API",
    description="提供小红书笔记的CTR、互动率等多维度预测服务",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """根路径"""
    return {
        "message": "小红书笔记多维度预测API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """健康检查接口"""
    try:
        # 检查各个服务的状态
        model_loaded = model_service is not None
        
        return HealthCheckResponse(
            status="healthy",
            model_loaded=model_loaded,
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )


@app.post("/predict", response_model=PredictionOutput)
async def predict_single(note: NoteInput):
    """单个笔记预测接口"""
    try:
        logger.info(f"Received prediction request for note: {note.note_id or 'unnamed'}")
        
        # 检查服务是否初始化
        if not all([llm_service, feature_service, model_service]):
            logger.warning("Services not fully initialized, using mock data")
            return model_service._get_mock_prediction(note.note_id) if model_service else PredictionOutput(
                note_id=note.note_id,
                ctr=0.05,
                like_rate=0.1,
                fav_rate=0.08,
                comment_rate=0.03,
                share_rate=0.02,
                follow_rate=0.01,
                interaction_rate=0.15,
                ces_rate=0.06,
                impression=10000.0,
                sort_score2=0.75
            )
        
        # 1. LLM标签预测
        logger.info("Step 1: Predicting tags with LLM...")
        tags = await llm_service.predict_tags(note)
        logger.info(f"Tags predicted: {tags.dict()}")
        
        # 2. 特征工程
        logger.info("Step 2: Extracting features...")
        features = await feature_service.extract_features(note, tags)
        features['note_id'] = note.note_id  # 添加note_id到特征中
        logger.info(f"Features extracted: {len(features)} features")
        
        # 3. 模型预测
        logger.info("Step 3: Running model inference...")
        prediction = model_service.predict(features)
        prediction.note_id = note.note_id  # 确保note_id正确设置
        logger.info(f"Prediction completed: CTR={prediction.ctr:.4f}")
        
        return prediction
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/predict_batch", response_model=BatchPredictionOutput)
async def predict_batch(batch: BatchNoteInput):
    """批量笔记预测接口"""
    try:
        logger.info(f"Received batch prediction request for {len(batch.notes)} notes")
        
        predictions = []
        for note in batch.notes:
            # TODO: 优化为批量处理
            pred = await predict_single(note)
            predictions.append(pred)
        
        return BatchPredictionOutput(
            predictions=predictions,
            total_notes=len(batch.notes)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    log_level = os.getenv("LOG_LEVEL", "INFO").lower()
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level=log_level
    )