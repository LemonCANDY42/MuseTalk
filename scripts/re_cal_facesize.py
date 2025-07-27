import json
from pathlib import Path
import os

# 指定包含JSON文件的文件夹路径
folder_path = Path("dataset/HDTF/meta")

# 删除数据不一致的文件列表
delete_list = []

# 遍历文件夹中的所有JSON文件
for json_file in folder_path.glob("*.json"):
    try:
        # 读取JSON内容
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查必要字段是否存在
        if "face_list" not in data or "frames" not in data:
            print(f"跳过 {json_file.name}，缺少必要字段")
            continue
        
        bbox_list = data["face_list"]
        total_frames = data["frames"]
        
        # 验证数据一致性
        if len(bbox_list) != total_frames:
            print(f"警告: {json_file.name}中帧数不一致 ({total_frames} vs {len(bbox_list)})，即将删除")
            # 删除数据不一致的文件
            os.remove(json_file)
            delete_list.append(json_file)
            continue
        
        # 计算所有bbox的平均尺寸
        height_total = 0
        width_total = 0
        
        for bbox in bbox_list:
            if not bbox:  # 跳过空bbox
                continue
            x1, y1, x2, y2 = bbox
            height_total += (y2 - y1)
            width_total += (x2 - x1)
        
        # 计算平均值（避免除零错误）
        avg_height = int(height_total / max(len(bbox_list), 1))
        avg_width = int(width_total / max(len(bbox_list), 1))
        
        # 打印更新内容
        print(f"更新 {json_file.name} 的 face_size :由 {data['face_size']} 变为 {avg_height} {avg_width}")
        
        # 更新数据
        data["face_size"] = [avg_height, avg_width]
        
        # 写回原始文件
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        
        print(f"已更新: {json_file.name}")

    except Exception as e:
        print(f"处理 {json_file.name} 时出错: {str(e)}")

print(f"删除数据不一致的文件总数: {len(delete_list)}\n文件列表：{delete_list}")

print("处理完成！")