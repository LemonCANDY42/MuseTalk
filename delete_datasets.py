import os
import re
import argparse

def delete_files_by_number_range(folder_path, dry_run=False):
    """
    删除指定文件夹中文件名最后一个下划线后数字在1-40或135-214范围内的文件
    
    Args:
        folder_path (str): 目标文件夹路径
        dry_run (bool): 若为True，仅打印待删除文件但不实际删除
    """
    deleted_count = 0
    skipped_count = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 跳过目录，只处理文件
        if not os.path.isfile(file_path):
            continue

        # 提取最后一个下划线后的数字（直到扩展名前）
        match = re.search(r'_(\d+)(?=\.[^.]+$)', filename)
        if not match:
            skipped_count += 1
            continue
            
        number_str = match.group(1)
        
        try:
            number = int(number_str)
            # 检查数字是否在目标范围
            if (1 <= number <= 40) or (135 <= number <= 214):
                if dry_run:
                    print(f"[Dry Run] 待删除: {filename} (数字: {number})")
                else:
                    os.remove(file_path)
                    print(f"已删除: {filename} (数字: {number})")
                    deleted_count += 1
            else:
                skipped_count += 1
        except ValueError:
            skipped_count += 1

    print(f"\n操作完成: 删除 {deleted_count} 个文件, 跳过 {skipped_count} 个文件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="删除文件名中特定数字范围的文件")
    parser.add_argument("folder", help="目标文件夹路径")
    parser.add_argument("--dry-run", action="store_true", help="模拟运行（不实际删除文件）")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"错误: 路径 {args.folder} 不是有效文件夹")
        exit(1)

    delete_files_by_number_range(args.folder, args.dry_run)