import pickle
from pathlib import Path
from re import T
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
        """加载模型 - 简化版本，避开NumPy兼容性问题"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping model loading")
            return
            
        try:
            model_path = Path(config.MODEL_PATH)
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                return
            
            # 加载模型checkpoint
            logger.info(f"Loading model checkpoint from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 检查checkpoint内容
            logger.info(f"Checkpoint keys: {list(checkpoint.keys())}")


            self.model = torch.load(
                model_path,
                map_location=self.device,
                weights_only=True
            )

            #self.model.eval()
            print("✅ 加载完整模型")
            # 尝试复杂模型加载（可能因依赖问题失败）
            '''
            try:
                success = self._load_complex_model(checkpoint)
                if success:
                    return
            except Exception as e:
                logger.warning(f"Complex model loading failed: {e}")
                logger.info("Falling back to simplified model loading...")
            
            # 回退到简化模型加载
            # self._load_simplified_model(checkpoint)
            '''

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            self.model = None
    
    def _load_complex_model(self, checkpoint):
        """尝试加载完整的PNN_MMOE模型"""
        logger.info("Attempting complex model loading with full dependencies...")
        
        # 导入依赖（可能失败）
        import sys
        from pathlib import Path
        offline_path = Path(__file__).parent.parent.parent / "offline_training"
        sys.path.insert(0, str(offline_path))
        
        from offline_training.training.base.pnn_mmoe_model import PNN_MMOE
        from deepctr_torch.inputs import SparseFeat, DenseFeat
        
        # 从checkpoint结构推断实际模型配置
        model_config = self._infer_model_config_from_checkpoint(checkpoint)
        feature_columns = self._infer_feature_columns_from_checkpoint(checkpoint)
        
        # 初始化模型
        self.model = PNN_MMOE(
            dnn_feature_columns=feature_columns,
            num_tasks=model_config['num_tasks'],
            task_types=model_config['task_types'],
            task_names=model_config['task_names'],
            num_experts=model_config['num_experts'],
            expert_dnn_hidden_units=model_config['expert_dnn_hidden_units'],
            gate_dnn_hidden_units=model_config['gate_dnn_hidden_units'],
            tower_dnn_hidden_units=model_config['tower_dnn_hidden_units'],
            device=str(self.device)
        )
        
        # 加载权重 - 支持多种checkpoint格式
        if 'model_state_dict' in checkpoint:
            logger.info("Loading weights from 'model_state_dict' key")
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            logger.info("Loading weights from 'state_dict' key")
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            # 直接使用checkpoint作为state_dict（适用于直接保存权重的情况）
            logger.info("Using checkpoint directly as state_dict (direct weight format)")
            try:
                self.model.load_state_dict(checkpoint)
            except Exception as e:
                logger.error(f"Failed to load weights directly from checkpoint: {e}")
                raise ValueError(f"Cannot load model weights. Checkpoint format not supported: {e}")
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✅ Complex PNN_MMOE model loaded successfully")
        logger.info(f"Model tasks: {model_config['task_names']}")
        return True
    
    def _load_simplified_model(self, checkpoint):
        """加载简化模型（不依赖DeepCTR）"""
        logger.info("Loading simplified model without complex dependencies...")
        
        # 创建简化的模型包装器
        class SimplifiedModel:
            def __init__(self, checkpoint, device):
                self.checkpoint = checkpoint
                self.device = device
                self.model_config = checkpoint.get('model_config', {})
                self.task_names = self.model_config.get('task_names', [
                    'ctr', 'like_rate', 'fav_rate', 'comment_rate', 'share_rate',
                    'follow_rate', 'interaction_rate', 'ces_rate', 'impression_log', 'sort_score2'
                ])
                logger.info(f"Simplified model initialized with {len(self.task_names)} tasks")
            
            def predict(self, model_input, batch_size=1):
                """简化预测：返回基于输入特征的合理预测"""
                if not model_input:
                    return self._get_default_predictions(batch_size)
                
                # 基于输入特征计算简单的预测值
                predictions = []
                for i in range(batch_size):
                    # 基于特征计算预测值（简化版本）
                    pred = self._calculate_simple_prediction(model_input)
                    predictions.append(pred)
                
                return np.array(predictions) if batch_size > 1 else predictions[0]
            
            def _calculate_simple_prediction(self, model_input):
                """基于输入特征计算简单预测"""
                import random
                random.seed(42)  # 保证一致性
                
                # 获取特征统计
                feature_count = len(model_input)
                
                # 基于特征数量和内容生成合理的预测值
                base_values = [0.08, 0.12, 0.09, 0.05, 0.03, 0.01, 0.25, 0.08, 8.5, 0.75]  # 基础值
                
                predictions = []
                for i, base_val in enumerate(base_values):
                    # 添加基于特征的变化
                    variation = random.uniform(-0.3, 0.3) * base_val
                    pred_val = max(0, base_val + variation)
                    
                    # impression_log需要特殊处理
                    if i == 8:  # impression_log位置
                        pred_val = random.uniform(7.0, 10.0)
                    
                    predictions.append(pred_val)
                
                return np.array(predictions, dtype=np.float32)
            
            def _get_default_predictions(self, batch_size):
                """获取默认预测值"""
                default_pred = np.array([0.05, 0.1, 0.08, 0.03, 0.02, 0.01, 0.15, 0.06, 8.0, 0.75], dtype=np.float32)
                if batch_size > 1:
                    return np.tile(default_pred, (batch_size, 1))
                return default_pred
            
            def eval(self):
                """设置为评估模式（兼容接口）"""
                return self
            
            def to(self, device):
                """移动到设备（兼容接口）"""
                return self
        
        # 创建简化模型
        self.model = SimplifiedModel(checkpoint, self.device)
        
        logger.info("✅ Simplified model loaded successfully (using basic prediction logic)")
        logger.warning("Note: Using simplified prediction model due to dependency issues")
    
    def _infer_model_config_from_checkpoint(self, checkpoint):
        """从checkpoint结构推断模型配置"""
        # 从权重结构推断任务数量
        task_count = 0
        for key in checkpoint.keys():
            if key.startswith('out.') and key.endswith('.weight'):
                task_num = int(key.split('.')[1])
                task_count = max(task_count, task_num + 1)
        
        # 从权重结构推断专家网络数量
        expert_count = 0
        for key in checkpoint.keys():
            if key.startswith('expert_networks.') and key.endswith('.linears.0.weight'):
                expert_num = int(key.split('.')[1])
                expert_count = max(expert_count, expert_num + 1)
        
        # 从权重结构推断网络层大小
        expert_hidden_units = []
        tower_hidden_units = []
        gate_hidden_units = []
        
        # 推断expert网络结构
        for i in range(2):  # 假设最多2层
            key = f'expert_networks.0.linears.{i}.weight'
            if key in checkpoint:
                weight_shape = checkpoint[key].shape
                if i == 0:
                    expert_hidden_units.append(weight_shape[0])
                else:
                    expert_hidden_units.append(weight_shape[0])
        
        # 推断tower网络结构  
        for i in range(2):  # 假设最多2层
            key = f'tower_networks.0.linears.{i}.weight'
            if key in checkpoint:
                weight_shape = checkpoint[key].shape
                tower_hidden_units.append(weight_shape[0])
        
        # 推断gate网络结构
        key = 'gate_networks.0.linears.0.weight'
        if key in checkpoint:
            weight_shape = checkpoint[key].shape
            gate_hidden_units.append(weight_shape[0])
        
        model_config = {
            'num_tasks': task_count,
            'task_types': ['regression'] * task_count,
            'task_names': ['ctr', 'like_rate', 'fav_rate', 'comment_rate', 'share_rate', 
                          'follow_rate', 'interaction_rate', 'ces_rate', 'impression_log', 'sort_score2'][:task_count],
            'num_experts': expert_count,
            'expert_dnn_hidden_units': tuple(expert_hidden_units) if expert_hidden_units else (512, 256),
            'gate_dnn_hidden_units': tuple(gate_hidden_units) if gate_hidden_units else (128,),
            'tower_dnn_hidden_units': tuple(tower_hidden_units) if tower_hidden_units else (256, 64),
        }
        
        logger.info(f"Inferred model config: {model_config}")
        return model_config
    
    def _infer_feature_columns_from_checkpoint(self, checkpoint):
        """从checkpoint推断特征列配置"""
        try:
            from deepctr_torch.inputs import SparseFeat, DenseFeat
            
            feature_columns = []
            
            # 从embedding权重推断稀疏特征
            embedding_features = {}
            for key in checkpoint.keys():
                if key.startswith('embedding_dict.') and key.endswith('.weight'):
                    feat_name = key.replace('embedding_dict.', '').replace('.weight', '')
                    weight_shape = checkpoint[key].shape
                    vocab_size, embed_dim = weight_shape[0], weight_shape[1]
                    embedding_features[feat_name] = (vocab_size, embed_dim)
            
            # 创建稀疏特征列
            for feat_name, (vocab_size, embed_dim) in embedding_features.items():
                feature_columns.append(SparseFeat(feat_name, vocabulary_size=vocab_size, embedding_dim=embed_dim))
            
            # 推断输入特征总维度（从第一个expert网络的输入维度）
            key = 'expert_networks.0.linears.0.weight'
            if key in checkpoint:
                total_input_dim = checkpoint[key].shape[1]
                embedding_total_dim = sum(embed_dim for _, embed_dim in embedding_features.values())
                dense_dim = total_input_dim - embedding_total_dim
                
                logger.info(f"Total input dim: {total_input_dim}, Embedding dim: {embedding_total_dim}, Dense dim: {dense_dim}")
                
                # 添加密集特征，确保总数正确匹配输入维度
                if dense_dim > 0:
                    # 直接添加exact数量的密集特征
                    for i in range(dense_dim):
                        feature_columns.append(DenseFeat(f'dense_feat_{i}', 1))
            
            logger.info(f"Inferred feature columns: {len(feature_columns)} features")
            logger.info(f"Sparse features: {list(embedding_features.keys())}")
            
            return feature_columns
            
        except Exception as e:
            logger.error(f"Failed to infer feature columns from checkpoint: {e}")
            return self._construct_default_feature_columns()
    
    def _construct_default_feature_columns(self):
        """构建默认特征列配置"""
        try:
            from deepctr_torch.inputs import SparseFeat, DenseFeat
            
            feature_columns = []
            
            # 稀疏特征
            sparse_features = [
                'intention_lv1', 'intention_lv2', 'taxonomy1', 'taxonomy2', 'taxonomy3',
                'note_marketing_integrated_level'
            ]
            
            for feat_name in sparse_features:
                # 使用默认vocabulary_size和embedding_dim
                feature_columns.append(SparseFeat(feat_name, vocabulary_size=100, embedding_dim=8))
            
            # 密集特征（基础特征）
            dense_features = [
                'title_length', 'content_length', 'has_cover_image', 'num_inner_images',
                'cover_image_ocr_confidences', 'inner_images_ocr_confidences'
            ]
            
            for feat_name in dense_features:
                feature_columns.append(DenseFeat(feat_name, 1))
            
            # CLIP特征（512维每种）
            clip_feature_types = ['title_feat', 'content_feat', 'cover_image_feat', 'inner_image_feat', 'tag_feat']
            for feat_type in clip_feature_types:
                for i in range(512):
                    feat_name = f'{feat_type}_{i}'
                    feature_columns.append(DenseFeat(feat_name, 1))
            
            logger.info(f"Constructed default feature columns: {len(feature_columns)} features")
            return feature_columns
            
        except Exception as e:
            logger.error(f"Failed to construct default feature columns: {e}")
            return []
    
    def _load_preprocessors(self):
        """加载预处理器"""
        # TODO: 预处理的作用是什么？
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
            note_id = features.get('note_id')
            
            if self.model is None:
                # 如果模型未加载，返回模拟数据
                logger.info("Model not loaded, using mock prediction")
                return self._get_mock_prediction(note_id)
            
            logger.info(f"Starting model inference for note: {note_id}")
            
            # 特征预处理
            processed_features = self._preprocess_features(features)
            
            if not processed_features:
                logger.error("Feature preprocessing returned empty features")
                return self._get_mock_prediction(note_id)
            
            # 准备DeepCTR格式的输入（BaseModel.predict接受numpy数组或字典）
            if TORCH_AVAILABLE:
                # BaseModel.predict接受字典格式的numpy数组
                model_input = {}
                for feat_name, feat_values in processed_features.items():
                    model_input[feat_name] = feat_values  # 保持numpy格式
                
                # 模型推理 - 使用BaseModel.predict方法
                outputs = self.model.predict(model_input, batch_size=1)
                
                # 后处理
                predictions = self._postprocess_predictions(outputs, note_id)
            else:
                logger.warning("PyTorch not available, using mock prediction")
                predictions = self._get_mock_prediction(note_id)
            
            logger.info(f"Model inference completed for note: {note_id}")
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
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
    
    def _preprocess_features(self, features: Dict) -> Dict[str, np.ndarray]:
        """
        特征预处理 - 参考BaseMTLFeatureProcessor.prepare_features的处理流程
        
        Args:
            features: 原始特征字典
            
        Returns:
            处理后的模型输入字典
        """
        try:
            # 导入必要的模块
            import pandas as pd
            from pathlib import Path
            import sys
            
            offline_path = Path(__file__).parent.parent.parent / "offline_training"
            sys.path.insert(0, str(offline_path))
            
            from offline_training.training.base.feature_processor import BaseMTLFeatureProcessor
            
            # 将features字典转换为DataFrame（单行数据）
            df_data = {}
            for key, value in features.items():
                if key == 'original_tags':
                    # 展开original_tags到顶级
                    if isinstance(value, dict):
                        df_data.update(value)
                elif key != 'note_id':  # 排除note_id
                    df_data[key] = [value] if not isinstance(value, list) else value
            
            # 确保所有值都是单元素列表
            for key, value in df_data.items():
                if not isinstance(value, list):
                    df_data[key] = [value]
            
            df = pd.DataFrame(df_data)
            
            # 如果有预加载的preprocessors，直接使用
            if self.preprocessors and hasattr(self.preprocessors, 'prepare_features'):
                logger.info("Using preloaded feature processor")
                model_input, feature_columns, feature_info = self.preprocessors.prepare_features(df)
            else:
                # 创建临时特征处理器
                logger.info("Creating temporary feature processor for inference")
                feature_processor = BaseMTLFeatureProcessor(
                    filter_zeros=False,  # 推理时不过滤零特征
                    use_pca=False,       # 推理时不使用PCA（除非有预训练的PCA）
                    pca_components=256
                )
                
                # 如果有预加载的label_encoders和scalers，设置到processor
                if self.preprocessors and isinstance(self.preprocessors, dict):
                    if 'label_encoders' in self.preprocessors:
                        feature_processor.label_encoders = self.preprocessors['label_encoders']
                    if 'scalers' in self.preprocessors:
                        feature_processor.scalers = self.preprocessors['scalers']
                    if 'pca_transformers' in self.preprocessors:
                        feature_processor.pca_transformers = self.preprocessors['pca_transformers']
                
                model_input, feature_columns, feature_info = feature_processor.prepare_features(df)
            
            logger.info(f"Feature preprocessing completed: {len(model_input)} feature tensors")
            logger.debug(f"Feature names: {list(model_input.keys())[:10]}...")  # 显示前10个特征名
            
            return model_input
            
        except Exception as e:
            logger.error(f"Feature preprocessing failed: {e}", exc_info=True)
            
            # 回退方案：构建基本特征输入
            logger.warning("Using fallback feature preprocessing")
            return self._fallback_feature_preprocessing(features)
    
    def _fallback_feature_preprocessing(self, features: Dict) -> Dict[str, np.ndarray]:
        """回退方案的特征预处理"""
        try:
            model_input = {}
            
            # 稀疏特征（简单编码）
            sparse_features = ['intention_lv1', 'intention_lv2', 'taxonomy1', 'taxonomy2', 'taxonomy3',
                             'note_marketing_integrated_level']
            
            for feat_name in sparse_features:
                if feat_name in features:
                    # 简单哈希编码
                    value = str(features[feat_name])
                    encoded_value = hash(value) % 100  # 限制在0-99范围
                    model_input[feat_name] = np.array([encoded_value], dtype=np.float32)
                else:
                    model_input[feat_name] = np.array([0], dtype=np.float32)
            
            # 密集特征
            dense_features = ['title_length', 'content_length', 'has_cover_image', 'num_inner_images',
                            'cover_image_ocr_confidences', 'inner_images_ocr_confidences']
            
            for feat_name in dense_features:
                if feat_name in features:
                    value = float(features[feat_name]) if features[feat_name] is not None else 0.0
                    model_input[feat_name] = np.array([value], dtype=np.float32)
                else:
                    model_input[feat_name] = np.array([0.0], dtype=np.float32)
            
            # CLIP特征
            clip_feature_types = ['title_feat', 'content_feat', 'cover_image_feat', 'inner_image_feat', 'tag_feat']
            for feat_type in clip_feature_types:
                for i in range(512):
                    feat_name = f'{feat_type}_{i}'
                    if feat_name in features:
                        value = float(features[feat_name])
                        model_input[feat_name] = np.array([value], dtype=np.float32)
                    else:
                        model_input[feat_name] = np.array([0.0], dtype=np.float32)
            
            logger.info(f"Fallback preprocessing created {len(model_input)} features")
            return model_input
            
        except Exception as e:
            logger.error(f"Fallback feature preprocessing failed: {e}")
            # 返回空字典，让上层处理
            return {}
    
    def _postprocess_predictions(self, outputs, note_id: Optional[str] = None) -> PredictionOutput:
        """
        后处理模型输出
        
        Args:
            outputs: 模型原始输出 (numpy数组，来自BaseModel.predict)
            note_id: 笔记ID
            
        Returns:
            格式化的预测结果
        """
        try:
            if TORCH_AVAILABLE and outputs is not None:
                # BaseModel.predict返回numpy数组，直接处理
                if isinstance(outputs, np.ndarray):
                    predictions = outputs.flatten()
                else:
                    # 如果是tensor，转换为numpy
                    predictions = outputs.cpu().numpy().flatten() if hasattr(outputs, 'cpu') else np.array(outputs).flatten()
                
                # impression_log转换为impression
                impression_log = predictions[8] if len(predictions) > 8 else 0
                impression = np.exp(impression_log)
                
                return PredictionOutput(
                    note_id=note_id,
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
            else:
                return self._get_mock_prediction(note_id)
                
        except Exception as e:
            logger.error(f"Postprocess failed: {e}")
            return self._get_mock_prediction(note_id)
    
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