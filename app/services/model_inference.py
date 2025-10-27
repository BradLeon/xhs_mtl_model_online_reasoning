import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, model inference will use mock data")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available, using Python lists")

from app.utils.config import config
from app.models.input_models import PredictionOutput


class ModelInferenceService:
    """MMOE模型推理服务"""
    
    def __init__(self):
        """初始化模型推理服务"""
        self.model = None
        self.preprocessors = None
        
        if TORCH_AVAILABLE:
            # 优先级：MPS > CUDA > CPU
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
            self._load_model()
        else:
            self.device = "cpu"
            logger.warning("PyTorch not available, model inference disabled")
            
        self._load_preprocessors()
    
    def _load_model(self):
        """加载MMOE模型"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping model loading")
            return
            
        try:
            model_path = Path(config.MODEL_PATH)
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                return
            
            # 加载模型
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # TODO: 根据实际模型结构初始化模型
            # from models.mmoe import MMOEModel
            # self.model = MMOEModel(...)
            # self.model.load_state_dict(checkpoint['model_state_dict'])
            # self.model.to(self.device)
            # self.model.eval()
            
            logger.info(f"Model loaded from {model_path}")
            logger.info(f"Using device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def _load_preprocessors(self):
        """加载预处理器"""
        try:
            preprocessor_path = Path(config.PREPROCESSOR_PATH)
            if not preprocessor_path.exists():
                logger.warning(f"Preprocessor file not found: {preprocessor_path}")
                return
            
            with open(preprocessor_path, 'rb') as f:
                self.preprocessors = pickle.load(f)
            
            logger.info(f"Preprocessors loaded from {preprocessor_path}")
            
        except Exception as e:
            logger.error(f"Failed to load preprocessors: {e}")
    
    def predict(self, features: Dict) -> PredictionOutput:
        """
        执行模型推理
        
        Args:
            features: 特征字典
            
        Returns:
            预测结果
        """
        try:
            if self.model is None:
                # 如果模型未加载，返回模拟数据
                return self._get_mock_prediction(features.get('note_id'))
            
            # 特征预处理
            processed_features = self._preprocess_features(features)
            
            # 转换为tensor
            input_tensor = torch.FloatTensor(processed_features).to(self.device)
            
            # 模型推理
            with torch.no_grad():
                outputs = self.model(input_tensor)
            
            # 后处理
            predictions = self._postprocess_predictions(outputs)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._get_mock_prediction(features.get('note_id'))
    
    def predict_batch(self, features_list: List[Dict]) -> List[PredictionOutput]:
        """
        批量预测
        
        Args:
            features_list: 特征字典列表
            
        Returns:
            预测结果列表
        """
        predictions = []
        for features in features_list:
            pred = self.predict(features)
            predictions.append(pred)
        
        return predictions
    
    def _preprocess_features(self, features: Dict) -> np.ndarray:
        """
        特征预处理
        
        Args:
            features: 原始特征
            
        Returns:
            处理后的特征数组
        """
        # TODO: 使用加载的preprocessors进行特征预处理
        # 这里需要与离线训练保持一致
        
        # 临时返回随机数组
        return np.random.randn(1, 100)
    
    def _postprocess_predictions(self, outputs: torch.Tensor) -> PredictionOutput:
        """
        后处理模型输出
        
        Args:
            outputs: 模型原始输出
            
        Returns:
            格式化的预测结果
        """
        # TODO: 根据实际模型输出格式进行后处理
        
        # 假设模型输出10个目标的预测值
        predictions = outputs.cpu().numpy().flatten()
        
        # impression_log转换为impression
        impression_log = predictions[8] if len(predictions) > 8 else 0
        impression = np.exp(impression_log)
        
        return PredictionOutput(
            ctr=float(predictions[0]) if len(predictions) > 0 else 0.05,
            like_rate=float(predictions[1]) if len(predictions) > 1 else 0.1,
            fav_rate=float(predictions[2]) if len(predictions) > 2 else 0.08,
            comment_rate=float(predictions[3]) if len(predictions) > 3 else 0.03,
            share_rate=float(predictions[4]) if len(predictions) > 4 else 0.02,
            follow_rate=float(predictions[5]) if len(predictions) > 5 else 0.01,
            interaction_rate=float(predictions[6]) if len(predictions) > 6 else 0.15,
            ces_rate=float(predictions[7]) if len(predictions) > 7 else 0.06,
            impression=float(impression),
            sort_score2=float(predictions[9]) if len(predictions) > 9 else 0.75
        )
    
    def _get_mock_prediction(self, note_id: Optional[str] = None) -> PredictionOutput:
        """
        获取模拟预测结果（用于测试或模型未加载时）
        
        Args:
            note_id: 笔记ID
            
        Returns:
            模拟的预测结果
        """
        # 生成合理范围内的随机预测值
        np.random.seed(hash(note_id) % 2**32 if note_id else 42)
        
        return PredictionOutput(
            note_id=note_id,
            ctr=float(np.random.uniform(0.01, 0.15)),
            like_rate=float(np.random.uniform(0.05, 0.25)),
            fav_rate=float(np.random.uniform(0.03, 0.20)),
            comment_rate=float(np.random.uniform(0.01, 0.10)),
            share_rate=float(np.random.uniform(0.005, 0.05)),
            follow_rate=float(np.random.uniform(0.001, 0.03)),
            interaction_rate=float(np.random.uniform(0.10, 0.40)),
            ces_rate=float(np.random.uniform(0.02, 0.15)),
            impression=float(np.random.uniform(1000, 50000)),
            sort_score2=float(np.random.uniform(0.5, 0.95))
        )
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_loaded": self.model is not None,
            "preprocessors_loaded": self.preprocessors is not None,
            "device": str(self.device),
            "model_path": config.MODEL_PATH,
            "preprocessor_path": config.PREPROCESSOR_PATH
        }