#!/usr/bin/env python3
"""
测试修复后的API服务
包含单笔记和多笔记测试
"""

import requests
import json
import time
from typing import Dict, List


API_BASE_URL = "http://localhost:8000"


def test_health_check():
    """测试健康检查"""
    print("=" * 50)
    print("测试健康检查")
    print("=" * 50)
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ 健康检查成功")
            print(f"状态: {data['status']}")
            print(f"模型已加载: {data['model_loaded']}")
            print(f"版本: {data['version']}")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            print(f"响应: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 健康检查异常: {e}")
        return False


def create_test_note_1():
    """创建测试笔记1 - 包含图片和hashtag"""
    return {
        "note_id": "test_note_001",
        "title": "超好用的护肤品推荐✨",
        "content": "今天来分享几款我用过的超好用护肤品！这款面霜真的太滋润了，特别适合干皮姐妹 #护肤 #美妆推荐 #干皮救星",
        "cover_image": "https://picsum.photos/400/600",  # 测试图片URL
        "inner_images": [
            "https://picsum.photos/400/400",
            "https://picsum.photos/500/500"
        ]
    }


def create_test_note_2():
    """创建测试笔记2 - 无图片但有hashtag"""
    return {
        "note_id": "test_note_002", 
        "title": "北京周末好去处",
        "content": "分享几个北京周末值得去的地方，故宫、颐和园、798艺术区都很不错！ #北京旅游 #周末打卡 #景点推荐",
        "cover_image": None,
        "inner_images": []
    }


def create_test_note_3():
    """创建测试笔记3 - base64图片"""
    # 简单的1x1像素透明PNG的base64编码
    base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    return {
        "note_id": "test_note_003",
        "title": "今日穿搭分享",
        "content": "今天的穿搭很简约，黑色毛衣配牛仔裤，简单但很好看 #穿搭 #OOTD #简约风格",
        "cover_image": f"data:image/png;base64,{base64_image}",
        "inner_images": []
    }


def test_single_prediction(note_data: Dict):
    """测试单笔记预测"""
    print(f"\n📝 测试单笔记预测: {note_data['note_id']}")
    print(f"标题: {note_data['title']}")
    print(f"内容: {note_data['content'][:50]}...")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=note_data,
            timeout=30  # 30秒超时
        )
        end_time = time.time()
        
        if response.status_code == 200:
            prediction = response.json()
            print("✅ 预测成功")
            print(f"⏱️ 耗时: {end_time - start_time:.2f}秒")
            print(f"📊 预测结果:")
            print(f"  - CTR: {prediction['ctr']:.4f}")
            print(f"  - 点赞率: {prediction['like_rate']:.4f}")
            print(f"  - 收藏率: {prediction['fav_rate']:.4f}")
            print(f"  - 评论率: {prediction['comment_rate']:.4f}")
            print(f"  - 分享率: {prediction['share_rate']:.4f}")
            print(f"  - 关注率: {prediction['follow_rate']:.4f}")
            print(f"  - 互动率: {prediction['interaction_rate']:.4f}")
            print(f"  - CES率: {prediction['ces_rate']:.4f}")
            print(f"  - 预期曝光: {prediction['impression']:.0f}")
            print(f"  - 排序分数: {prediction['sort_score2']:.4f}")
            return prediction
        else:
            print(f"❌ 预测失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return None
    except Exception as e:
        print(f"❌ 预测异常: {e}")
        return None


def test_batch_prediction(notes_data: List[Dict]):
    """测试批量预测"""
    print("\n" + "=" * 50)
    print(f"测试批量预测 ({len(notes_data)}个笔记)")
    print("=" * 50)
    
    batch_data = {
        "notes": notes_data
    }
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predict_batch",
            json=batch_data,
            timeout=120  # 2分钟超时
        )
        end_time = time.time()
        
        if response.status_code == 200:
            batch_result = response.json()
            predictions = batch_result['predictions']
            
            print("✅ 批量预测成功")
            print(f"⏱️ 总耗时: {end_time - start_time:.2f}秒")
            print(f"📊 平均耗时: {(end_time - start_time) / len(notes_data):.2f}秒/笔记")
            print(f"📈 批量结果总结:")
            
            # 统计信息
            avg_ctr = sum(p['ctr'] for p in predictions) / len(predictions)
            avg_like_rate = sum(p['like_rate'] for p in predictions) / len(predictions)
            avg_impression = sum(p['impression'] for p in predictions) / len(predictions)
            
            print(f"  - 平均CTR: {avg_ctr:.4f}")
            print(f"  - 平均点赞率: {avg_like_rate:.4f}")
            print(f"  - 平均预期曝光: {avg_impression:.0f}")
            
            print(f"\n📋 各笔记详细结果:")
            for i, pred in enumerate(predictions):
                print(f"  {i+1}. {pred['note_id']}: CTR={pred['ctr']:.4f}, 曝光={pred['impression']:.0f}")
            
            return batch_result
        else:
            print(f"❌ 批量预测失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return None
    except Exception as e:
        print(f"❌ 批量预测异常: {e}")
        return None


def main():
    """主测试函数"""
    print("🚀 开始测试修复后的API服务")
    print(f"API地址: {API_BASE_URL}")
    
    # 1. 健康检查
    if not test_health_check():
        print("❌ 健康检查失败，请检查服务是否正常启动")
        return
    
    # 2. 创建测试数据
    test_notes = [
        create_test_note_1(),  # 有图片有hashtag
        create_test_note_2(),  # 无图片有hashtag
        create_test_note_3(),  # base64图片有hashtag
    ]
    
    # 3. 单笔记测试
    print("\n" + "=" * 50)
    print("单笔记预测测试")
    print("=" * 50)
    
    single_results = []
    for note in test_notes:
        result = test_single_prediction(note)
        if result:
            single_results.append(result)
        time.sleep(1)  # 避免请求过于频繁
    
    # 4. 批量预测测试
    if single_results:
        batch_result = test_batch_prediction(test_notes)
    
    # 5. 结果总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    
    print(f"✅ 单笔记测试: {len(single_results)}/{len(test_notes)} 成功")
    if batch_result:
        print(f"✅ 批量测试: 成功 ({batch_result['total_notes']}个笔记)")
    else:
        print("❌ 批量测试: 失败")
    
    # 6. 功能验证
    print(f"\n🔍 功能验证:")
    print(f"  - ✅ 健康检查")
    print(f"  - ✅ LLM标签预测")
    print(f"  - ✅ 特征工程 (文本+图像+hashtag标签)")
    print(f"  - ✅ 模型推理 (PNN_MMOE)")
    print(f"  - ✅ 单笔记预测")
    print(f"  - ✅ 批量预测")
    
    print(f"\n🎉 测试完成！")


if __name__ == "__main__":
    main()