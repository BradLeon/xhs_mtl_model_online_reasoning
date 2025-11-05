#!/usr/bin/env python3
"""
模型推理服务

借鉴离线训练的MTLPredictor逻辑，但在本项目内实现完整的模型加载和推理功能
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger
import numpy as np

try:
    import torch
    import torch.nn as nn
    from deepctr_torch.inputs import SparseFeat, DenseFeat
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch or DeepCTR-Torch not available")


# 添加offline_training路径以导入模型定义
offline_path = Path(__file__).parent.parent.parent / "offline_training"
sys.path.insert(0, str(offline_path))

from app.utils.config import config
from app.models.input_models import PredictionOutput


# ========== MPS Compatibility Patch ==========
# Fix for MPS (Apple Silicon) devices: DeepCTR-Torch's BaseModel.predict()
# has multiple float64 issues that break MPS compatibility
if TORCH_AVAILABLE:
    from deepctr_torch.models.basemodel import BaseModel
    from torch.utils.data import DataLoader
    from torch.utils import data as Data

    _original_deepctr_predict = BaseModel.predict

    def _mps_safe_predict(self, x, batch_size=256):
        """MPS-compatible reimplementation of BaseModel.predict()

        Fixes multiple float64 issues:
        1. np.concatenate defaults to float64
        2. torch.from_numpy preserves float64
        3. Final .astype("float64") conversion
        """
        model = self.eval()

        # Convert dict to list of arrays
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        # Ensure all inputs are float32 numpy arrays
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
            # Force float32 for MPS compatibility
            if x[i].dtype == np.float64:
                x[i] = x[i].astype(np.float32)

        # Concatenate with explicit float32 dtype
        concatenated = np.concatenate(x, axis=-1)
        if concatenated.dtype == np.float64:
            logger.debug("🔄 Converting concatenated features from float64 to float32")
            concatenated = concatenated.astype(np.float32)

        # Create tensor dataset
        tensor_data = Data.TensorDataset(torch.from_numpy(concatenated))
        test_loader = DataLoader(dataset=tensor_data, shuffle=False, batch_size=batch_size)

        # Run predictions
        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x_batch = x_test[0].to(self.device).float()
                y_pred = model(x_batch).cpu().data.numpy()
                pred_ans.append(y_pred)

        # Return float32 predictions instead of float64
        result = np.concatenate(pred_ans).astype(np.float32)
        logger.debug(f"✅ MPS-safe predict completed, output dtype: {result.dtype}")
        return result

    BaseModel.predict = _mps_safe_predict
    logger.info("✅ Applied comprehensive MPS compatibility patch to DeepCTR BaseModel")
# ========== End MPS Compatibility Patch ==========


class ModelInferenceService:
    """模型推理服务
    
    借鉴MTLPredictor的实现逻辑，支持标准checkpoint加载和特征预处理
    """
    
    def __init__(self):
        """初始化模型推理服务"""
        self.model = None
        self.preprocessors = None
        self.feature_columns = []
        self.feature_names = []
        self.tasks = []
        self.task_column_mapping = {}
        self.label_normalizer = None
        self.training_info = {}
        
        # 设置设备
        self.device = self._get_device()
        
        # checkpoint目录
        self.checkpoint_dir = Path(config.MODEL_PATH).parent
        
        logger.info("="*60)
        logger.info("🚀 Initializing Model Inference Service")
        logger.info(f"Checkpoint dir: {self.checkpoint_dir}")
        logger.info(f"Device: {self.device}")
        logger.info("="*60)
        
        if not TORCH_AVAILABLE:
            logger.error("❌ PyTorch not available, cannot load model")
            return
        
        # 按照MTLPredictor的逻辑初始化
        try:
            # 1. 加载checkpoint元数据
            self._load_metadata()
            
            # 2. 加载模型
            logger.info("Loading model...")
            self._load_model()
            
            # 3. 加载预处理器
            logger.info("Loading preprocessors...")
            self._load_preprocessors()
            
            # 4. 加载特征列定义
            logger.info("Loading feature columns...")
            self._load_feature_columns()
            
            # 5. 加载标签归一化器
            logger.info("Loading label normalizer...")
            self._load_label_normalizer()

            # 验证 label_normalizer 状态
            if self.label_normalizer is None:
                logger.error("❌ CRITICAL: Label normalizer is None!")
                logger.error("   Predictions will NOT be denormalized - this will cause wrong predictions!")
                logger.error("   Expected file: models/label_normalizer.pkl")
            else:
                logger.info("✅ Label normalizer validation:")
                logger.info(f"   Normalization method: {getattr(self.label_normalizer, 'normalization_method', 'unknown')}")
                if hasattr(self.label_normalizer, 'fitted_tasks'):
                    logger.info(f"   Fitted tasks: {self.label_normalizer.fitted_tasks}")
                if hasattr(self.label_normalizer, 'normalizers'):
                    logger.info(f"   Number of normalizers: {len(self.label_normalizer.normalizers)}")

            # 6. 加载训练信息
            logger.info("Loading training info...")
            self._load_training_info()
            
            logger.info("✅ Model Inference Service initialized successfully")
            logger.info(f"Model type: {self.training_info.get('model_type', 'unknown')}")
            logger.info(f"Tasks: {', '.join(self.tasks)}")
            
            # 7. 预热模型
            if self.model:
                self._warmup()
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize model inference service: {e}", exc_info=True)
    
    def _get_device(self) -> str:
        """获取推理设备"""
        if not TORCH_AVAILABLE:
            return 'cpu'
        
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    
    def _load_metadata(self):
        """加载checkpoint元数据"""
        metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        if metadata_file.exists():
            # 打印文件详细信息
            file_size = metadata_file.stat().st_size
            file_mtime = metadata_file.stat().st_mtime
            import time
            mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_mtime))
            logger.info(f"📄 Loading: {metadata_file}")
            logger.info(f"   Size: {file_size} bytes, Modified: {mtime_str}")

            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"   ✅ Loaded checkpoint metadata: version {self.metadata.get('version', 'unknown')}")
        else:
            logger.warning("⚠️  No metadata file found in checkpoint")
            self.metadata = {}
    
    def _load_model(self):
        """加载模型，借鉴ModelLoader的逻辑"""
        # 首先尝试加载完整模型
        complete_model_path = self.checkpoint_dir / "complete_model.pth"
        if complete_model_path.exists():
            try:
                logger.info(f"Loading complete model from {complete_model_path}")
                self.model = torch.load(complete_model_path, map_location=self.device)
                self.model.eval()
                logger.info(f"✅ Complete model loaded successfully: {self.model.__class__.__name__}")
                return
            except Exception as e:
                logger.warning(f"Failed to load complete model: {e}")
                logger.info("Falling back to rebuild method...")
        
        # 回退到从配置重建模型
        self._rebuild_and_load_model()
    
    def _rebuild_and_load_model(self):
        """从配置重建模型并加载权重"""
        try:
            # 1. 加载模型配置
            model_config = self._load_model_config()
            
            # 2. 临时加载特征列用于模型重建
            temp_feature_columns = self._load_feature_columns_for_model()
            
            # 3. 重建模型
            self.model = self._create_model(model_config, temp_feature_columns)
            
            # 4. 加载权重
            weights_file = self.checkpoint_dir / "model.pth"
            if weights_file.exists():
                # 打印文件详细信息
                file_size = weights_file.stat().st_size
                file_mtime = weights_file.stat().st_mtime
                import time
                mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_mtime))
                logger.info(f"📄 Loading model weights: {weights_file}")
                logger.info(f"   Size: {file_size / 1024 / 1024:.2f} MB, Modified: {mtime_str}")

                state_dict = torch.load(weights_file, map_location=self.device)

                # 打印权重信息
                logger.info(f"   State dict keys: {len(state_dict.keys())}")
                # 显示前5个权重的形状
                for i, (key, tensor) in enumerate(list(state_dict.items())[:5]):
                    logger.info(f"   [{i}] {key}: {tensor.shape}")

                self.model.load_state_dict(state_dict)
                logger.info("   ✅ Model weights loaded successfully")
            else:
                logger.error("❌ Model weights not found!")
                return
            
            self.model.eval()
            logger.info(f"✅ Model rebuilt successfully: {self.model.__class__.__name__}")
            
        except Exception as e:
            logger.error(f"Failed to rebuild model: {e}", exc_info=True)
            self.model = None
    
    def _load_model_config(self) -> Dict[str, Any]:
        """加载模型配置"""
        config_file = self.checkpoint_dir / "model_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Model config not found: {config_file}")

        # 打印文件详细信息
        file_size = config_file.stat().st_size
        file_mtime = config_file.stat().st_mtime
        import time
        mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_mtime))
        logger.info(f"📄 Loading: {config_file}")
        logger.info(f"   Size: {file_size} bytes, Modified: {mtime_str}")

        with open(config_file, 'r') as f:
            config_data = json.load(f)

        # 打印关键配置信息
        logger.info(f"   ✅ Model type: {config_data.get('model_type', 'unknown')}")
        logger.info(f"   ✅ Tasks: {config_data.get('tasks', [])}")
        logger.info(f"   ✅ L2 reg embedding: {config_data.get('l2_reg_embedding', 'N/A')}")
        logger.info(f"   ✅ L2 reg dnn: {config_data.get('l2_reg_dnn', 'N/A')}")

        return config_data
    
    def _load_feature_columns_for_model(self) -> List:
        """加载特征列定义（用于模型重建）"""
        feature_file = self.checkpoint_dir / "feature_columns.json"
        if not feature_file.exists():
            raise FileNotFoundError(f"Feature columns not found: {feature_file}")

        # 打印文件详细信息
        file_size = feature_file.stat().st_size
        file_mtime = feature_file.stat().st_mtime
        import time
        mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_mtime))
        logger.info(f"📄 Loading: {feature_file}")
        logger.info(f"   Size: {file_size} bytes, Modified: {mtime_str}")

        with open(feature_file, 'r') as f:
            feature_data = json.load(f)

        # 重建特征列对象
        feature_columns = []
        sparse_count = 0
        dense_count = 0
        for feat_info in feature_data:
            if feat_info['type'] == 'SparseFeat':
                feature_columns.append(SparseFeat(
                    name=feat_info['name'],
                    vocabulary_size=feat_info['vocabulary_size'],
                    embedding_dim=feat_info['embedding_dim'],
                    dtype=feat_info.get('dtype', 'int32')
                ))
                sparse_count += 1
            elif feat_info['type'] == 'DenseFeat':
                feature_columns.append(DenseFeat(
                    name=feat_info['name'],
                    dimension=feat_info.get('dimension', 1),
                    dtype=feat_info.get('dtype', 'float32')
                ))
                dense_count += 1

        logger.info(f"   ✅ Loaded {len(feature_columns)} feature columns (Sparse: {sparse_count}, Dense: {dense_count})")
        return feature_columns
    
    def _create_model(self, model_config: Dict[str, Any], feature_columns: List) -> nn.Module:
        """根据配置创建模型"""
        model_type = model_config.get('model_type', 'PNN_MMOE')
        
        if model_type == 'PNN_MMOE':
            # 导入PNN_MMOE模型
            from offline_training.training.base.pnn_mmoe_model import PNN_MMOE

            # 🔍 关键诊断：检查导入的PNN_MMOE模型来源
            import inspect
            pnn_module_file = inspect.getfile(PNN_MMOE)
            logger.info(f"🔍 PNN_MMOE model class loaded from: {pnn_module_file}")
            import os
            if os.path.exists(pnn_module_file):
                pnn_mtime = os.path.getmtime(pnn_module_file)
                import time
                pnn_mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(pnn_mtime))
                logger.info(f"   File modified: {pnn_mtime_str}")

            pnn_mmoe_config = model_config.get('pnn_mmoe_config', {})
            mmoe_config = pnn_mmoe_config.get('mmoe', {})
            pnn_config = pnn_mmoe_config.get('pnn', {})
            
            model = PNN_MMOE(
                dnn_feature_columns=feature_columns,
                num_tasks=len(model_config.get('tasks', [])),
                task_types=['binary'] * len(model_config.get('tasks', [])),
                task_names=model_config.get('tasks', []),
                use_inner_product=pnn_config.get('use_inner_product', True),
                use_outter_product=pnn_config.get('use_outter_product', False),
                num_experts=mmoe_config.get('num_experts', 3),
                expert_dnn_hidden_units=tuple(mmoe_config.get('expert_dims', [128, 64])),
                gate_dnn_hidden_units=tuple(mmoe_config.get('gate_dims', [32])),
                tower_dnn_hidden_units=tuple(mmoe_config.get('tower_dims', [64, 32])),
                dnn_dropout=model_config.get('dropout', 0.1),
                l2_reg_embedding=model_config.get('l2_reg_embedding', 1e-5),
                l2_reg_dnn=model_config.get('l2_reg_dnn', 0),
                device=self.device
            )
            
            logger.info(f"Created PNN_MMOE model with {len(model_config.get('tasks', []))} tasks")
            return model
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _load_preprocessors(self):
        """加载预处理器"""
        preprocessor_file = self.checkpoint_dir / "preprocessors.pkl"
        if preprocessor_file.exists():
            try:
                # 打印文件详细信息
                file_size = preprocessor_file.stat().st_size
                file_mtime = preprocessor_file.stat().st_mtime
                import time
                mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_mtime))
                logger.info(f"📄 Loading: {preprocessor_file}")
                logger.info(f"   Size: {file_size / 1024 / 1024:.2f} MB, Modified: {mtime_str}")

                with open(preprocessor_file, 'rb') as f:
                    self.preprocessors = pickle.load(f)

                # 打印preprocessors内容
                if isinstance(self.preprocessors, dict):
                    logger.info(f"   Keys: {list(self.preprocessors.keys())}")
                    if 'label_encoders' in self.preprocessors:
                        logger.info(f"   Label encoders count: {len(self.preprocessors['label_encoders'])}")
                    if 'scalers' in self.preprocessors:
                        logger.info(f"   Scalers count: {len(self.preprocessors['scalers'])}")

                logger.info("   ✅ Preprocessors loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load preprocessors: {e}")
                self.preprocessors = None
        else:
            logger.warning("⚠️  Preprocessors file not found")
            self.preprocessors = None
    
    def _load_feature_columns(self):
        """加载特征列定义"""
        feature_file = self.checkpoint_dir / "feature_columns.json"
        if feature_file.exists():
            try:
                with open(feature_file, 'r') as f:
                    feature_data = json.load(f)
                
                # 提取特征名称
                self.feature_names = [feat['name'] for feat in feature_data]
                self.feature_columns = feature_data
                
                logger.info(f"✅ Loaded {len(self.feature_names)} feature columns")
            except Exception as e:
                logger.error(f"Failed to load feature columns: {e}")
                self.feature_columns = []
                self.feature_names = []
        else:
            logger.warning("Feature columns file not found")
            self.feature_columns = []
            self.feature_names = []
    
    def _load_label_normalizer(self):
        """加载标签归一化器"""
        normalizer_file = self.checkpoint_dir / "label_normalizer.pkl"
        if normalizer_file.exists():
            try:
                with open(normalizer_file, 'rb') as f:
                    self.label_normalizer = pickle.load(f)
                logger.info("✅ Label normalizer loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load label normalizer: {e}")
                self.label_normalizer = None
        else:
            logger.info("Label normalizer file not found (this is normal)")
            self.label_normalizer = None
    
    def _load_training_info(self):
        """加载训练信息"""
        training_file = self.checkpoint_dir / "training_info.json"
        if training_file.exists():
            try:
                with open(training_file, 'r') as f:
                    self.training_info = json.load(f)
                
                self.tasks = self.training_info.get('tasks', [])
                self.task_column_mapping = self.training_info.get('task_column_mapping', {})
                
                logger.info(f"✅ Training info loaded: {len(self.tasks)} tasks")
            except Exception as e:
                logger.error(f"Failed to load training info: {e}")
                self.training_info = {}
                self.tasks = []
                self.task_column_mapping = {}
        else:
            logger.warning("Training info file not found")
            self.training_info = {}
            self.tasks = []
            self.task_column_mapping = {}
    
    def _warmup(self):
        """预热模型，减少首次推理延迟"""
        if not self.model:
            return
        
        logger.info("🔥 Warming up model...")
        
        try:
            # 创建虚拟输入
            dummy_input = {}
            for feat_name in self.feature_names:
                dummy_input[feat_name] = np.zeros(1, dtype=np.float32)
            
            # 执行一次推理
            with torch.no_grad():
                _ = self.model.predict(dummy_input, batch_size=1)
            
            logger.info("✅ Model warmed up successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Model warmup failed: {e}")
    
    def predict(self, features: Dict) -> PredictionOutput:
        """
        执行模型推理
        
        Args:
            features: 特征字典
            
        Returns:
            预测结果
        """
        note_id = features.get('note_id')
        
        if not self.model:
            logger.warning("Model not loaded, using mock prediction")
            return self._get_mock_prediction(note_id)
        
        try:
            logger.info(f"🔮 Starting model inference for note: {note_id}")

            # 预处理特征
            logger.debug("Step 1: Starting _preprocess_features...")
            processed_features = self._preprocess_features(features)
            logger.debug(f"Step 1 ✅: _preprocess_features completed, feature dtypes: {[f.dtype for f in processed_features.values() if isinstance(f, np.ndarray)][:5]}")

            # 执行预测
            logger.debug("Step 2: Starting model.predict...")
            with torch.no_grad():
                predictions = self.model.predict(processed_features, batch_size=1)

            # 💾 记录模型原始输出（归一化空间）
            logger.info(f"📊 Raw predictions from model (normalized space): {predictions}")
            logger.info(f"   Shape: {predictions.shape if isinstance(predictions, np.ndarray) else 'N/A'}")
            logger.info(f"   Dtype: {predictions.dtype if isinstance(predictions, np.ndarray) else type(predictions)}")
            if isinstance(predictions, np.ndarray) and predictions.ndim > 1:
                logger.info(f"   Values: {predictions[0, :]}")
            elif isinstance(predictions, np.ndarray):
                logger.info(f"   Values: {predictions}")

            logger.debug(f"Step 2 ✅: model.predict completed")

            # Safety check: Convert any remaining float64 to float32 for MPS compatibility
            if isinstance(predictions, np.ndarray) and predictions.dtype == np.float64:
                logger.warning("⚠️ Predictions still in float64 after patch, converting to float32")
                predictions = predictions.astype(np.float32)

            # 后处理预测结果
            logger.debug("Step 3: Starting _postprocess_predictions...")
            result = self._postprocess_predictions(predictions, note_id)
            logger.debug("Step 3 ✅: _postprocess_predictions completed")

            logger.info(f"✅ Model inference completed for note: {note_id}")
            return result

        except Exception as e:
            logger.error(f"❌ Prediction failed for note {note_id}: {e}", exc_info=True)
            return self._get_mock_prediction(note_id)
    
    def predict_batch(self, features_list: List[Dict]) -> List[PredictionOutput]:
        """
        批量预测
        
        Args:
            features_list: 特征字典列表
            
        Returns:
            预测结果列表
        """
        if not self.model:
            logger.warning("Model not loaded, using mock predictions")
            return [self._get_mock_prediction(f.get('note_id')) for f in features_list]
        
        try:
            logger.info(f"🔮 Starting batch inference for {len(features_list)} notes")
            
            # 预处理所有特征
            batch_features = []
            for features in features_list:
                processed = self._preprocess_features(features)
                batch_features.append(processed)
            
            # 合并为批量输入
            batch_input = {}
            for feat_name in self.feature_names:
                feat_values = []
                for processed in batch_features:
                    feat_values.append(processed.get(feat_name, np.array([0.0], dtype=np.float32))[0])
                batch_input[feat_name] = np.array(feat_values, dtype=np.float32)
            
            # 执行批量预测
            with torch.no_grad():
                batch_predictions = self.model.predict(batch_input, batch_size=len(features_list))
            
            # 后处理结果
            results = []
            for i, features in enumerate(features_list):
                note_id = features.get('note_id')
                pred_values = batch_predictions[i] if len(batch_predictions.shape) > 1 else batch_predictions
                result = self._postprocess_single_prediction(pred_values, note_id)
                results.append(result)
            
            logger.info(f"✅ Batch inference completed for {len(features_list)} notes")
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch prediction failed: {e}", exc_info=True)
            return [self._get_mock_prediction(f.get('note_id')) for f in features_list]
    
    def _get_sparse_and_dense_features(self):
        """从 feature_columns 中识别稀疏和密集特征"""
        sparse_features = []
        dense_features = []

        for feat_info in self.feature_columns:
            feat_name = feat_info['name']
            feat_type = feat_info['type']

            if feat_type == 'SparseFeat':
                sparse_features.append(feat_name)
            elif feat_type == 'DenseFeat':
                dense_features.append(feat_name)

        return sparse_features, dense_features

    def _apply_label_encoder(self, feature_name: str, value):
        """应用 LabelEncoder 到稀疏特征"""
        try:
            if not self.preprocessors or 'label_encoders' not in self.preprocessors:
                logger.warning(f"No label encoders available, returning default for {feature_name}")
                return 0

            encoder = self.preprocessors['label_encoders'].get(feature_name)
            if encoder is None:
                logger.warning(f"No encoder found for feature {feature_name}, using default 0")
                return 0

            # 处理字符串值
            if isinstance(value, str):
                try:
                    encoded_value = encoder.transform([value])[0]
                    return int(encoded_value)
                except (ValueError, KeyError) as e:
                    # 未知类别，返回默认值
                    logger.warning(f"Unknown category '{value}' for feature {feature_name}, using default 0")
                    return 0
            elif isinstance(value, (int, float)):
                return int(value)
            else:
                return 0

        except Exception as e:
            logger.error(f"Error encoding feature {feature_name}: {e}")
            return 0

    def _apply_standard_scaler(self, feature_name: str, value: float) -> float:
        """应用 StandardScaler 到密集特征"""
        try:
            # ✅ FIX #2: CLIP 特征已经标准化，跳过 StandardScaler
            # CLIP 模型输出的 embedding 特征已经归一化到 [-1, 1] 范围
            # 如果再应用 StandardScaler 会破坏特征分布
            CLIP_FEATURE_PREFIXES = ['cover_image_feat_', 'title_feat_', 'content_feat_',
                                     'inner_image_feat_', 'tag_feat_']
            if any(feature_name.startswith(prefix) for prefix in CLIP_FEATURE_PREFIXES):
                # CLIP features are already normalized, skip scaling
                return np.float32(value)

            if not self.preprocessors or 'scalers' not in self.preprocessors:
                logger.warning(f"No scalers available, returning raw value for {feature_name}")
                return float(value)

            scaler = self.preprocessors['scalers'].get(feature_name)
            if scaler is None:
                #logger.warning(f"No scaler found for feature {feature_name}, using raw value")
                return float(value)

            # 应用标准化
            if isinstance(value, (int, float)):
                # 强制使用 float32 输入，避免 sklearn 返回 float64
                scaled_value = scaler.transform(np.array([[value]], dtype=np.float32))[0][0]
                logger.info(f"scaled_value: {scaled_value}, raw value: {value}")
                # 确保返回 float32 以兼容 MPS 设备
                return np.float32(scaled_value)
            else:
                logger.warning(f"Non-numeric value for dense feature {feature_name}, using 0.0")
                return np.float32(0.0)

        except Exception as e:
            logger.error(f"Error scaling feature {feature_name}: {e}")
            return np.float32(value)

    def _apply_pca_if_needed(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """如果训练时使用了 PCA，则应用 PCA 转换"""
        try:
            if not self.preprocessors or 'pca_transformers' not in self.preprocessors:
                return features

            pca_transformers = self.preprocessors['pca_transformers']
            if not pca_transformers:
                return features

            # 处理每个 PCA 组
            for pca_name, pca in pca_transformers.items():
                original_features = []
                original_feat_names = []

                for feat_name in self.feature_names:
                    if feat_name.startswith(f"{pca_name}_feat_"):
                        if feat_name in features:
                            # 确保收集的特征是 float32
                            original_features.append(np.float32(features[feat_name][0]))
                            original_feat_names.append(feat_name)

                if original_features:
                    # 应用 PCA - 显式指定 float32 dtype
                    original_array = np.array(original_features, dtype=np.float32).reshape(1, -1)
                    # PCA 返回 float64，立即转换为 float32
                    pca_result = pca.transform(original_array)[0].astype(np.float32)

                    # 替换原始特征为 PCA 特征
                    for feat_name in original_feat_names:
                        del features[feat_name]

                    # 添加 PCA 特征（已经是 float32）
                    n_components = len(pca_result)
                    for i in range(n_components):
                        pca_feat_name = f"{pca_name}_pca_{i}"
                        features[pca_feat_name] = np.array([pca_result[i]], dtype=np.float32)

                    logger.debug(f"Applied PCA for {pca_name}: {len(original_features)} → {n_components} dims")

            return features

        except Exception as e:
            logger.error(f"Error applying PCA: {e}")
            return features

    def _preprocess_features(self, features: Dict) -> Dict[str, np.ndarray]:
        """
        预处理特征（修复版本 - 应用训练时的预处理器）

        应用顺序:
        1. LabelEncoder for sparse features (categorical → integer)
        2. StandardScaler for dense features (normalization)
        3. PCA for CLIP features (optional dimensionality reduction)
        """
        try:
            # 识别稀疏和密集特征
            sparse_features, dense_features = self._get_sparse_and_dense_features()

            # ========== Phase 2: 特征一致性分析 ==========
            # 分析输入特征 vs 预期特征，帮助发现训练和推理的不一致
            expected_features = set(self.feature_names)
            provided_features = set(features.keys())
            missing_features = expected_features - provided_features
            extra_features = provided_features - expected_features

            # 按特征类型分组缺失特征
            missing_sparse = [f for f in missing_features if f in sparse_features]
            missing_dense = [f for f in missing_features if f in dense_features]
            missing_other = [f for f in missing_features if f not in sparse_features and f not in dense_features]

            # 打印特征统计摘要
            logger.info(f"📊 Feature Analysis: {len(expected_features)} expected, {len(provided_features)} provided, {len(missing_features)} missing")

            # 详细显示缺失特征（限制输出数量）
            if missing_sparse:
                logger.warning(f"🔍 Missing {len(missing_sparse)} sparse features (showing first 10): {missing_sparse[:10]}")
            if missing_dense:
                logger.warning(f"🔍 Missing {len(missing_dense)} dense features (showing first 10): {missing_dense[:10]}")
            if missing_other:
                logger.debug(f"🔍 Missing {len(missing_other)} other features (CLIP/embeddings, showing first 10): {missing_other[:10]}")

            # 显示额外提供的特征（可能是新增的）
            if extra_features:
                logger.debug(f"✨ Extra features provided (not in training): {list(extra_features)[:10]}")

            # 显示存在的关键特征
            key_features = ['note_id', 'title', 'content', 'cover_image', 'nickname', 'note_type']
            present_key_features = [f for f in key_features if f in provided_features]
            logger.info(f"✅ Present key features: {present_key_features}")
            # ========== End Phase 2 ==========

            processed = {}

            # 处理每个特征
            for feat_name in self.feature_names:
                if feat_name in sparse_features:
                    # 稀疏特征：应用 LabelEncoder
                    raw_value = features.get(feat_name, '')
                    encoded_value = self._apply_label_encoder(feat_name, raw_value)
                    processed[feat_name] = np.array([encoded_value], dtype=np.int32)

                elif feat_name in dense_features:
                    # 密集特征：应用 StandardScaler
                    raw_value = features.get(feat_name, 0.0)
                    scaled_value = self._apply_standard_scaler(feat_name, raw_value)
                    # 确保使用 float32（MPS 兼容）
                    processed[feat_name] = np.array([float(scaled_value)], dtype=np.float32)

                else:
                    # 其他特征（如 CLIP embeddings）
                    if feat_name in features:
                        value = features[feat_name]
                        if isinstance(value, (list, np.ndarray)):
                            processed[feat_name] = np.array(value, dtype=np.float32).flatten()[:1]
                        elif isinstance(value, (int, float)):
                            processed[feat_name] = np.array([float(value)], dtype=np.float32)
                        else:
                            processed[feat_name] = np.array([0.0], dtype=np.float32)
                    else:
                        processed[feat_name] = np.array([0.0], dtype=np.float32)

            # 应用 PCA（如果训练时使用）
            processed = self._apply_pca_if_needed(processed)

            # ========== Phase 2: 特征处理统计 ==========
            # 统计各类特征的处理情况
            processed_sparse = sum(1 for f in processed.keys() if f in sparse_features)
            processed_dense = sum(1 for f in processed.keys() if f in dense_features)
            processed_other = len(processed) - processed_sparse - processed_dense

            # 统计使用默认值的特征数量
            features_with_defaults = len(missing_features)
            default_rate = (features_with_defaults / len(expected_features) * 100) if expected_features else 0

            logger.info(f"✅ Preprocessed {len(processed)} features: sparse={processed_sparse}/{len(sparse_features)}, "
                       f"dense={processed_dense}/{len(dense_features)}, other={processed_other}")
            logger.info(f"⚠️  Using default values for {features_with_defaults} features ({default_rate:.1f}%)")
            # ========== End Phase 2 ==========

            # ========== 特征详细诊断：保存所有特征名和值用于离在线对比 ==========
            
            import json
            from pathlib import Path
            try:
                features_dict = {}
                for feat_name, feat_value in processed.items():
                    # 将numpy数组转为Python列表
                    if isinstance(feat_value, np.ndarray):
                        features_dict[feat_name] = feat_value.tolist()
                    else:
                        features_dict[feat_name] = float(feat_value) if isinstance(feat_value, (int, float)) else str(feat_value)

                # 保存到JSON文件
                output_file = Path("online_features.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(features_dict, f, indent=2, ensure_ascii=False)

                logger.info(f"📝 Saved {len(features_dict)} features to {output_file}")

                # 打印前10个稀疏特征和前10个密集特征作为样本
                sparse_sample = [f for f in processed.keys() if f in sparse_features][:10]
                dense_sample = [f for f in processed.keys() if f in dense_features][:10]

                logger.info(f"📊 Sample sparse features ({len(sparse_sample)}):")
                for feat in sparse_sample:
                    logger.info(f"   {feat}: {features_dict[feat]}")

                logger.info(f"📊 Sample dense features ({len(dense_sample)}):")
                for feat in dense_sample:
                    logger.info(f"   {feat}: {features_dict[feat]}")

            except Exception as e:
                logger.warning(f"Failed to save features for diagnosis: {e}")
            # ========== End 特征诊断 ==========
            
            return processed

        except Exception as e:
            logger.error(f"Feature preprocessing failed: {e}", exc_info=True)
            # 返回默认特征
            return {feat_name: np.array([0.0], dtype=np.float32) for feat_name in self.feature_names}
    
    def _postprocess_predictions(self, predictions, note_id: Optional[str] = None) -> PredictionOutput:
        """后处理预测结果"""
        try:
            if isinstance(predictions, np.ndarray):
                pred_values = predictions.flatten()
            else:
                pred_values = np.array(predictions).flatten()
            
            return self._postprocess_single_prediction(pred_values, note_id)
            
        except Exception as e:
            logger.error(f"Postprocess failed: {e}")
            return self._get_mock_prediction(note_id)
    
    def _postprocess_single_prediction(self, pred_values, note_id: Optional[str] = None) -> PredictionOutput:
        """后处理单个预测结果"""
        try:
            # 根据任务映射提取预测值
            task_mapping = self.task_column_mapping

            # ✅ FIX #1: 应用 label denormalization（从标准化空间转回原始空间）
            # 训练时对标签做了 StandardScaler 标准化，预测值也是标准化后的
            # 必须进行逆变换才能得到真实的预测值
            logger.info(f"📊 Raw predictions from model (normalized space): {pred_values}")
            logger.info(f"   Shape: {pred_values.shape}, Dtype: {pred_values.dtype}")
            logger.info(f"   Number of tasks: {len(self.tasks)}")
            logger.info(f"   Tasks: {self.tasks}")

            if self.label_normalizer is not None:
                logger.info(f"🔄 Applying label denormalization to predictions")
                logger.info(f"   Using normalizer: {type(self.label_normalizer).__name__}")
                logger.info(f"   Normalizer fitted tasks: {self.label_normalizer.fitted_tasks if hasattr(self.label_normalizer, 'fitted_tasks') else 'N/A'}")

                # pred_values 是 1D 数组，需要 reshape 成 2D (1, n_tasks)
                pred_values_2d = pred_values.reshape(1, -1)
                logger.info(f"   Reshaped to 2D: {pred_values_2d.shape}")

                # 逆标准化：将标准化后的值转回原始尺度
                denormalized = self.label_normalizer.inverse_transform(pred_values_2d, self.tasks)
                logger.info(f"   Denormalized shape: {denormalized.shape}")

                # 转回 1D 数组
                pred_values = denormalized.flatten()

                logger.info(f"✅ Denormalized predictions (original space): {pred_values}")
                # 诊断：显示每个任务的反归一化前后对比
                logger.info("   Per-task denormalization:")
                for i, task in enumerate(self.tasks):
                    if i < len(pred_values_2d[0]) and i < len(pred_values):
                        logger.info(f"     [{i}] {task}: {pred_values_2d[0][i]:.6f} → {pred_values[i]:.6f}")
            else:
                logger.error("❌ CRITICAL: No label_normalizer available!")
                logger.error("   Predictions are in NORMALIZED space (wrong scale)!")
                logger.error("   This will cause negative values and wrong magnitude!")

            # 默认值
            predictions = {
                'ctr': 0.05,
                'like_rate': 0.1,
                'fav_rate': 0.08,
                'comment_rate': 0.03,
                'share_rate': 0.02,
                'follow_rate': 0.01,
                'interaction_rate': 0.15,
                'ces_rate': 0.06,
                'impression': 8.0,
                'sort_score': 0.75
            }

            # 从预测值中提取
            logger.info("📋 Extracting predictions from denormalized values:")
            for i, task in enumerate(self.tasks):
                if i < len(pred_values):
                    predictions[task] = float(pred_values[i])
                    logger.info(f"   [{i}] {task}: {predictions[task]:.6f}")
                else:
                    logger.warning(f"   [{i}] {task}: index out of range (len={len(pred_values)}), using default")

            # ✅ FIX #3: 预测值范围验证和修正
            # 对于率类指标（ctr, like_rate等），确保在合理范围内 [0, 1]
            logger.info("🔄 Applying clip to rate/ctr tasks [0, 1]:")
            rate_tasks = ['ctr', 'like_rate', 'fav_rate', 'comment_rate', 'share_rate',
                         'follow_rate', 'interaction_rate', 'ces_rate', 'sort_score']
            clip_count = 0
            for task in rate_tasks:
                if task in predictions:
                    # 将异常值限制在 [0, 1] 范围内
                    original_value = predictions[task]
                    predictions[task] = max(0.0, min(1.0, predictions[task]))
                    if abs(original_value - predictions[task]) > 0.0001:  # 降低阈值以捕获所有clip
                        logger.warning(f"   ⚠️  {task}: {original_value:.6f} → {predictions[task]:.6f} (CLIPPED!)")
                        clip_count += 1
                    else:
                        logger.info(f"   ✅ {task}: {predictions[task]:.6f} (no clip)")

            if clip_count > 0:
                logger.warning(f"⚠️  Total {clip_count} tasks were clipped!")

            # 处理impression（从log转换）
            impression_log = predictions.get('impression', 8.0)
            impression = np.exp(impression_log) if impression_log > 0 else 1000.0
            logger.info(f"🔄 Impression transformation: {impression_log:.6f} (log) → {impression:.0f} (count)")

            # 最终预测结果诊断
            logger.info("📋 Final predictions (after all processing):")
            logger.info(f"   CTR: {predictions['ctr']:.6f}")
            logger.info(f"   Like rate: {predictions['like_rate']:.6f}")
            logger.info(f"   Fav rate: {predictions['fav_rate']:.6f}")
            logger.info(f"   Comment rate: {predictions['comment_rate']:.6f}")
            logger.info(f"   Share rate: {predictions['share_rate']:.6f}")
            logger.info(f"   Follow rate: {predictions['follow_rate']:.6f}")
            logger.info(f"   Interaction rate: {predictions['interaction_rate']:.6f}")
            logger.info(f"   CES rate: {predictions['ces_rate']:.6f}")
            logger.info(f"   Impression: {impression:.0f}")
            logger.info(f"   Sort score: {predictions['sort_score']:.6f}")

            return PredictionOutput(
                note_id=note_id,
                ctr=predictions['ctr'],
                like_rate=predictions['like_rate'],
                fav_rate=predictions['fav_rate'],
                comment_rate=predictions['comment_rate'],
                share_rate=predictions['share_rate'],
                follow_rate=predictions['follow_rate'],
                interaction_rate=predictions['interaction_rate'],
                ces_rate=predictions['ces_rate'],
                impression=float(impression),
                sort_score2=predictions['sort_score']
            )
            
        except Exception as e:
            logger.error(f"Single prediction postprocess failed: {e}")
            return self._get_mock_prediction(note_id)
    
    def _get_mock_prediction(self, note_id: Optional[str] = None) -> PredictionOutput:
        """获取模拟预测结果"""
        import random
        if note_id:
            random.seed(hash(note_id) % 2**32)
        else:
            random.seed(42)
        
        return PredictionOutput(
            note_id=note_id,
            ctr=float(random.uniform(0.01, 0.15)),
            like_rate=float(random.uniform(0.05, 0.25)),
            fav_rate=float(random.uniform(0.03, 0.20)),
            comment_rate=float(random.uniform(0.01, 0.10)),
            share_rate=float(random.uniform(0.005, 0.05)),
            follow_rate=float(random.uniform(0.001, 0.03)),
            interaction_rate=float(random.uniform(0.10, 0.40)),
            ces_rate=float(random.uniform(0.02, 0.15)),
            impression=float(random.uniform(1000, 50000)),
            sort_score2=float(random.uniform(0.5, 0.95))
        )
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        info = {
            "model_loaded": self.model is not None,
            "device": self.device,
            "checkpoint_dir": str(self.checkpoint_dir),
            "preprocessors_loaded": self.preprocessors is not None,
        }
        
        if self.model:
            info.update({
                "tasks": self.tasks,
                "model_type": self.training_info.get('model_type', 'unknown'),
                "task_column_mapping": self.task_column_mapping,
                "feature_count": len(self.feature_names)
            })
        
        return info