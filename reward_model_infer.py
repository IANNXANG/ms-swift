#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import torch
import os
from PIL import Image
from swift.llm import PtEngine, RequestConfig
from swift.llm.infer.protocol import InferRequest


data_path = "output/v0-20250617-143536/val_dataset.jsonl"
#data_path = "code2image.jsonl"

def load_code2image_samples(jsonl_file="code2image.jsonl", num_samples=5):
    """
    从code2image.jsonl文件中加载样本
    """
    samples = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line.strip())
            samples.append(data)
    return samples

def replace_image_placeholders(messages, images):
    """
    将消息中的<image>占位符替换为实际的图片路径
    """
    if not images:
        return messages
    
    processed_messages = []
    image_index = 0
    
    for message in messages:
        if isinstance(message.get('content'), str) and '<image>' in message['content']:
            # 构建包含图片的内容
            content = []
            parts = message['content'].split('<image>')
            
            for i, text_part in enumerate(parts):
                if text_part.strip():
                    content.append({"type": "text", "text": text_part.strip()})
                
                # 在文本部分之间插入图片（除了最后一部分）
                if i < len(parts) - 1 and image_index < len(images):
                    # 检查图片文件是否存在
                    image_path = images[image_index]
                    if os.path.exists(image_path):
                        content.append({
                            "type": "image", 
                            "image": image_path
                        })
                    else:
                        print(f"警告: 图片文件不存在: {image_path}")
                        content.append({"type": "text", "text": f"[图片文件不存在: {image_path}]"})
                    image_index += 1
            
            processed_messages.append({
                "role": message["role"],
                "content": content
            })
        else:
            processed_messages.append(message)
    
    return processed_messages

def test_reward_model():
    """
    使用训练好的奖励模型进行推理评分
    """
    # 训练好的模型路径
    model_path = "output/v0-20250617-143536/checkpoint-83"
    
    print("正在加载奖励模型...")
    
    # 初始化推理引擎
    # 由于这是一个奖励模型（seq_cls任务），需要指定相关参数
    engine = PtEngine(
        model_id_or_path=model_path,
        task_type='seq_cls',  # 序列分类任务
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    
    print(f"模型加载成功！")
    print(f"模型类型: {engine.model.__class__.__name__}")
    print(f"模型配置: {engine.model.config}")

    
    # 从code2image.jsonl加载测试样例
    print(f"\n正在加载{data_path}数据...")
    try:

        samples = load_code2image_samples(data_path, num_samples=100)
        print(f"成功加载 {len(samples)} 个样本")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    print("\n开始测试奖励模型评分...")
    print("=" * 60)
    
    # 处理每个样本
    test_requests = []
    sample_info = []
    
    for i, sample in enumerate(samples):
        #从初始出数据获得message
        chosen_messages = sample['messages'].copy()
        rejected_messages = sample['messages'].copy() 
        
        if data_path == "code2image.jsonl":
            # 处理chosen response (正样本)
            chosen_messages = replace_image_placeholders(chosen_messages, [sample.get('images')[0]])
            # 处理rejected response (负样本)
            rejected_messages = replace_image_placeholders(rejected_messages, [sample.get('images')[1]])
        else:
            chosen_messages = replace_image_placeholders(chosen_messages, [sample.get('images')[0]["path"]])
            rejected_messages = replace_image_placeholders(rejected_messages, [sample.get('images')[1]["path"]])
        # 添加到测试请求列表
        test_requests.append(InferRequest(messages=chosen_messages))
        test_requests.append(InferRequest(messages=rejected_messages))
        
        sample_info.extend([
            {"type": "chosen", "sample_id": i+1, "images": chosen_messages[2]['content'][0]['image']},
            {"type": "rejected", "sample_id": i+1, "images": rejected_messages[2]['content'][0]['image']}
        ])
    
    try:
        # 获取奖励分数
        print("正在进行推理...")
        results = engine.infer(test_requests)
        
        # 显示结果并比较分数
        scores = {}
        for i, (info, result) in enumerate(zip(sample_info, results)):
            sample_id = info['sample_id']
            response_type = info['type']

            #打印图片
            if info['type'] == "chosen":
                print(info['type'],":",info['images'])
            elif info['type'] == "rejected":
                print(info['type'],":",info['images'])


            # 从结果中提取分数
            if hasattr(result, 'choices') and result.choices:
                score = float(result.choices[0].message.content)          
                # 存储分数用于比较
                if sample_id not in scores:
                    scores[sample_id] = {}
                scores[sample_id][response_type] = score 
            else:
                print(f"原始结果: {result}")
        
        # 比较每个样本的终版和初版分数
        print("\n" + "=" * 60)
        print("分数比较结果:")
        print("=" * 60)
        
        correct_count = 0
        total_count = 0
        
        for sample_id in sorted(scores.keys()):
            if 'chosen' in scores[sample_id] and 'rejected' in scores[sample_id]:
                chosen_score = scores[sample_id]['chosen']
                rejected_score = scores[sample_id]['rejected']
                total_count += 1
                
                if chosen_score > rejected_score:
                    result_symbol = "✅"
                    result_text = "终版图片分数更高"
                    correct_count += 1
                else:
                    result_symbol = "❌"
                    result_text = "初版图片分数更高或相等"
                
                print(f"样本 {sample_id}: 终版({chosen_score:.4f}) vs 初版({rejected_score:.4f}) {result_symbol} {result_text}")
        
        # 计算并显示准确率
        if total_count > 0:
            accuracy = (correct_count / total_count) * 100
            print("\n" + "=" * 60)
            print(f"最终准确率: {correct_count}/{total_count} = {accuracy:.2f}%")
            print(f"正确预测样本数: {correct_count} (终版图片分数高于初版)")
            print(f"总样本数: {total_count}")
            print("=" * 60)
        else:
            print("\n没有有效的样本进行准确率计算")
                
    except Exception as e:
        print(f"推理过程中出现错误: {e}")

if __name__ == "__main__":
    test_reward_model()