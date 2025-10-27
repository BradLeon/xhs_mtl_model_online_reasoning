#!/usr/bin/env python3
"""
API测试脚本
"""

import json
import asyncio
import httpx
from typing import Dict, Any
from loguru import logger


BASE_URL = "http://localhost:8000"


async def test_health_check():
    """测试健康检查接口"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")
        logger.info(f"Health check response: {response.status_code}")
        logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.json()


async def test_single_prediction():
    """测试单个笔记预测"""
    test_note = {
        "title": "夏日清凉美食推荐｜自制柠檬茶超解暑",
        "cover_image": "https://example.com/image.jpg",  # 这里可以是真实的图片URL或base64
        "content": """炎炎夏日，来一杯自制的柠檬茶吧！
        
        材料准备：
        - 新鲜柠檬2个
        - 蜂蜜适量
        - 绿茶包2个
        - 冰块
        - 薄荷叶装饰
        
        制作步骤：
        1. 柠檬切片，去籽备用
        2. 绿茶用80度水冲泡5分钟
        3. 茶水晾凉后加入蜂蜜调味
        4. 加入柠檬片和冰块
        5. 最后用薄荷叶装饰
        
        小贴士：柠檬要选择新鲜多汁的，茶水不要太烫避免破坏维C哦～
        
        #夏日饮品 #自制柠檬茶 #消暑解渴 #美食分享
        """,
        "inner_images": [],
        "note_id": "test_001"
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        logger.info("Sending single prediction request...")
        response = await client.post(
            f"{BASE_URL}/predict",
            json=test_note
        )
        
        logger.info(f"Single prediction response: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            logger.info("Prediction results:")
            for key, value in result.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
        else:
            logger.error(f"Error: {response.text}")
        
        return response.json() if response.status_code == 200 else None


async def test_batch_prediction():
    """测试批量笔记预测"""
    test_batch = {
        "notes": [
            {
                "title": "秋季护肤好物分享",
                "cover_image": "",
                "content": "换季护肤很重要，今天分享几个我的爱用好物...",
                "inner_images": [],
                "note_id": "batch_001"
            },
            {
                "title": "周末宅家追剧清单",
                "cover_image": "",
                "content": "最近追的几部剧都超好看，推荐给大家...",
                "inner_images": [],
                "note_id": "batch_002"
            },
            {
                "title": "新手健身入门指南",
                "cover_image": "",
                "content": "作为健身小白，分享一下我的健身入门经验...",
                "inner_images": [],
                "note_id": "batch_003"
            }
        ]
    }
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        logger.info("Sending batch prediction request...")
        response = await client.post(
            f"{BASE_URL}/predict_batch",
            json=test_batch
        )
        
        logger.info(f"Batch prediction response: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Total notes processed: {result['total_notes']}")
            logger.info("Batch prediction results:")
            for i, pred in enumerate(result['predictions'], 1):
                logger.info(f"\n  Note {i} ({pred.get('note_id', 'unknown')}):")
                logger.info(f"    CTR: {pred['ctr']:.4f}")
                logger.info(f"    Like Rate: {pred['like_rate']:.4f}")
                logger.info(f"    Interaction Rate: {pred['interaction_rate']:.4f}")
        else:
            logger.error(f"Error: {response.text}")
        
        return response.json() if response.status_code == 200 else None


async def main():
    """主测试函数"""
    logger.add("test_api.log", rotation="10 MB")
    logger.info("Starting API tests...")
    
    # 1. 健康检查
    logger.info("\n=== Testing Health Check ===")
    health_result = await test_health_check()
    
    # 2. 单个预测
    logger.info("\n=== Testing Single Prediction ===")
    single_result = await test_single_prediction()
    
    # 3. 批量预测
    logger.info("\n=== Testing Batch Prediction ===")
    batch_result = await test_batch_prediction()
    
    logger.info("\n=== All tests completed ===")
    
    # 返回测试结果摘要
    return {
        "health_check": "passed" if health_result else "failed",
        "single_prediction": "passed" if single_result else "failed",
        "batch_prediction": "passed" if batch_result else "failed"
    }


if __name__ == "__main__":
    results = asyncio.run(main())
    logger.info(f"\nTest Summary: {json.dumps(results, indent=2)}")
    
    # 返回状态码
    all_passed = all(v == "passed" for v in results.values())
    exit(0 if all_passed else 1)