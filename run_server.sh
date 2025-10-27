#!/bin/bash

# 启动服务器脚本

echo "Starting XHS MTL Model Online Reasoning Server..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed"
    exit 1
fi

# 检查环境变量
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Using default configuration."
    echo "Copy .env.example to .env and configure your settings."
fi

# 安装依赖（如果需要）
if [ "$1" == "--install" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# 启动服务器
echo "Starting FastAPI server on http://localhost:8000"
echo "API documentation available at http://localhost:8000/docs"
echo "Press Ctrl+C to stop the server"

# 使用python运行
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload