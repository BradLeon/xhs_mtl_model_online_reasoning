# 小红书笔记多维度预测在线推理服务

小红书内容多维度模型在线推理，包括tag predict + features extraction + model prediction

## 功能特性

- 🏷️ **LLM标签预测**: 使用OpenRouter API对笔记进行6个维度的标签分类
- 🔧 **特征工程**: 复用离线训练的pipeline，包括OCR、CN-CLIP等多模态特征提取
- 🎯 **MMOE模型推理**: 预测10个目标变量（CTR、互动率等）
- ⚡ **批量处理**: 支持单笔记和批量笔记预测
- 📊 **RESTful API**: 基于FastAPI的Web服务接口

## 预测维度

1. **ctr**: 点击率
2. **like_rate**: 点赞率  
3. **fav_rate**: 收藏率
4. **comment_rate**: 评论率
5. **share_rate**: 分享率
6. **follow_rate**: 关注率
7. **interaction_rate**: 互动率
8. **ces_rate**: CES率
9. **impression**: 曝光量（从impression_log转换）
10. **sort_score2**: 排序分数

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository_url>
cd xhs_mtl_model_online_reasoning

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置设置

```bash
# 复制配置文件模板
cp .env.example .env

# 编辑.env文件，设置以下配置：
# - OPENROUTER_API_KEY: OpenRouter API密钥
# - MODEL_PATH: MMOE模型文件路径
# - PREPROCESSOR_PATH: 预处理器文件路径
```

### 3. 启动服务

```bash
# 方式1: 使用启动脚本
./run_server.sh

# 方式2: 直接运行
python -m uvicorn app.main:app --reload --port 8000
```

服务启动后可访问：
- API服务: http://localhost:8000
- API文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

## API接口

### 单笔记预测

```bash
POST /predict
Content-Type: application/json

{
  "title": "笔记标题",
  "cover_image": "封面图片URL或base64",
  "content": "笔记内容",
  "inner_images": ["图片1", "图片2"],
  "note_id": "optional_id"
}
```

### 批量预测

```bash
POST /predict_batch
Content-Type: application/json

{
  "notes": [
    {
      "title": "笔记1标题",
      "cover_image": "...",
      "content": "...",
      "inner_images": [],
      "note_id": "note_001"
    },
    ...
  ]
}
```

## 测试

```bash
# 运行测试脚本
python test_api.py
```

## 项目结构

```
xhs_mtl_model_online_reasoning/
├── app/
│   ├── main.py              # FastAPI主应用
│   ├── models/              # 数据模型
│   ├── services/            # 核心服务
│   │   ├── llm_tagging.py   # LLM标签服务
│   │   ├── feature_pipeline.py  # 特征工程
│   │   └── model_inference.py   # 模型推理
│   └── utils/               # 工具类
├── offline_training/        # 离线训练代码（子模块）
├── config/                  # 配置文件
├── tests/                   # 测试文件
├── requirements.txt         # 依赖列表
└── README.md               # 项目说明
