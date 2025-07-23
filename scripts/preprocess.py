import os
import argparse
import subprocess
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from typing import Tuple, List, Union
import decord
import json
import cv2
from musetalk.utils.face_detection import FaceAlignment,LandmarksType
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
import sys

import concurrent.futures
import psutil

def get_cpu_usage():
    """获取当前CPU使用率"""
    return psutil.cpu_percent(interval=1)

def has_nvidia_encoder():
    """检查系统是否支持NVIDIA硬件编码器（如h264_nvenc）"""
    try:
        # 检查ffmpeg是否支持nvenc
        result = subprocess.run(["ffmpeg", "-hide_banner", "-codecs"], 
                              capture_output=True, text=True)
        return "h264_nvenc" in result.stdout
    except:
        return False


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

ffmpeg_path = "./ffmpeg-4.4-amd64-static/"
if not fast_check_ffmpeg():
    print("Adding ffmpeg to PATH")
    # Choose path separator based on operating system
    path_separator = ';' if sys.platform == 'win32' else ':'
    os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
    if not fast_check_ffmpeg():
        print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

class AnalyzeFace:
    def __init__(self, device: Union[str, torch.device], config_file: str, checkpoint_file: str):
        """
        Initialize the AnalyzeFace class with the given device, config file, and checkpoint file.

        Parameters:
        device (Union[str, torch.device]): The device to run the models on ('cuda' or 'cpu').
        config_file (str): Path to the mmpose model configuration file.
        checkpoint_file (str): Path to the mmpose model checkpoint file.
        """
        self.device = device
        self.dwpose = init_model(config_file, checkpoint_file, device=self.device)
        self.facedet = FaceAlignment(LandmarksType._2D, flip_input=False, device=self.device)

    def __call__(self, im: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Detect faces and keypoints in the given image.

        Parameters:
        im (np.ndarray): The input image.
        maxface (bool): Whether to detect the maximum face. Default is True.

        Returns:
        Tuple[List[np.ndarray], np.ndarray]: A tuple containing the bounding boxes and keypoints.
        """
        try:
            # Ensure the input image has the correct shape
            if im.ndim == 3:
                im = np.expand_dims(im, axis=0)
            elif im.ndim != 4 or im.shape[0] != 1:
                raise ValueError("Input image must have shape (1, H, W, C)")
            
            bbox = self.facedet.get_detections_for_batch(np.asarray(im))
            results = inference_topdown(self.dwpose, np.asarray(im)[0])
            results = merge_data_samples(results)
            keypoints = results.pred_instances.keypoints
            face_land_mark= keypoints[0][23:91]
            face_land_mark = face_land_mark.astype(np.int32)

            return face_land_mark, bbox
        
        except Exception as e:
            print(f"Error during face analysis: {e}")
            return np.array([]),[] 
    
def convert_video(org_path: str, dst_path: str, vid_list: List[str]) -> None:

    """
    Convert video files to a specified format and save them to the destination path.

    Parameters:
    org_path (str): The directory containing the original video files.
    dst_path (str): The directory where the converted video files will be saved.
    vid_list (List[str]): A list of video file names to process.

    Returns:
    None
    """

    def convert_single_video(vid_info):
        """转换单个视频文件"""
        idx, vid, codec_type = vid_info
        if vid.endswith('.mp4'):
            org_vid_path = os.path.join(org_path, vid)
            dst_vid_path = os.path.join(dst_path, vid)
                
            if org_vid_path != dst_vid_path:
                if codec_type == "nvenc":
                    video_codec = "h264_nvenc"
                    threads_param = []  # NVENC不需要线程参数
                else:
                    video_codec = "libx264"
                    threads_param = ["-threads", "8"]  # CPU编码使用8线程
                
                cmd = [
                    "ffmpeg", "-hide_banner", "-y", "-i", org_vid_path, 
                    "-r", "25", "-crf", "15", "-c:v", video_codec,
                    "-pix_fmt", "yuv420p"
                ] + threads_param + [dst_vid_path]
                
                subprocess.run(cmd, check=True)
                
                if idx % 100 == 0:
                    print(f"### {idx} 个视频已转换，使用 {codec_type} 编码器 ###")
    
    # 检查是否有NVIDIA硬件编码器
    has_nvenc = has_nvidia_encoder()
    
    # 准备任务列表
    nvenc_tasks = []
    cpu_tasks = []
    
    for idx, vid in enumerate(vid_list):
        if vid.endswith('.mp4'):
            org_vid_path = os.path.join(org_path, vid)
            dst_vid_path = os.path.join(dst_path, vid)
            
            if org_vid_path != dst_vid_path:
                if has_nvenc:
                    # 如果有NVENC，将任务分配给NVENC和CPU
                    if idx % 2 == 0:  # 偶数索引使用NVENC
                        nvenc_tasks.append((idx, vid, "nvenc"))
                    else:  # 奇数索引使用CPU
                        cpu_tasks.append((idx, vid, "cpu"))
                else:
                    # 只有CPU编码
                    cpu_tasks.append((idx, vid, "cpu"))
    
    print(f"开始视频转换: NVENC任务 {len(nvenc_tasks)} 个, CPU任务 {len(cpu_tasks)} 个")
    
    # 确定CPU并行数量
    cpu_cores = psutil.cpu_count(logical=True)  # 逻辑核心数
    initial_cpu_usage = get_cpu_usage()
    
    # 根据CPU使用率动态调整并行数
    if initial_cpu_usage < 30:  # CPU使用率低于30%
        max_cpu_workers = min(cpu_cores // 8, len(cpu_tasks))  # 每个任务8线程
    elif initial_cpu_usage < 60:  # CPU使用率30-60%
        max_cpu_workers = min(cpu_cores // 12, len(cpu_tasks))
    else:  # CPU使用率高于60%
        max_cpu_workers = 1
    
    max_cpu_workers = max(1, max_cpu_workers)  # 至少1个worker
    
    print(f"使用 {max_cpu_workers} 个CPU编码进程 (每个使用8线程)")
    
    if has_nvenc:
        print("使用NVIDIA硬件编码器并行处理")
        
        # 使用线程池同时处理NVENC和CPU任务
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as nvenc_executor, \
             concurrent.futures.ThreadPoolExecutor(max_workers=max_cpu_workers) as cpu_executor:
            
            # 提交NVENC任务
            nvenc_futures = [nvenc_executor.submit(convert_single_video, task) for task in nvenc_tasks]
            
            # 提交CPU任务
            cpu_futures = [cpu_executor.submit(convert_single_video, task) for task in cpu_tasks]
            
            # 等待所有任务完成
            all_futures = nvenc_futures + cpu_futures
            for future in concurrent.futures.as_completed(all_futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"视频转换过程中出现错误: {e}")
    else:
        print("仅使用CPU编码器")
        # 只使用CPU编码
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_cpu_workers) as executor:
            futures = [executor.submit(convert_single_video, task) for task in cpu_tasks]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"视频转换过程中出现错误: {e}")
    
    print("所有视频转换完成")

def segment_video(org_path: str, dst_path: str, vid_list: List[str], segment_duration: int = 30) -> None:
    """
    Segment video files into smaller clips of specified duration.

    Parameters:
    org_path (str): The directory containing the original video files.
    dst_path (str): The directory where the segmented video files will be saved.
    vid_list (List[str]): A list of video file names to process.
    segment_duration (int): The duration of each segment in seconds. Default is 30 seconds.

    Returns:
    None
    """
    # 过滤出mp4文件
    mp4_files = [vid for vid in vid_list if vid.endswith('.mp4')]
    
    if not mp4_files:
        print("没有找到需要分割的MP4文件")
        return
    
    # 确定CPU并行数量
    cpu_cores = psutil.cpu_count(logical=True)  # 逻辑核心数
    initial_cpu_usage = get_cpu_usage()
    
    # 根据CPU使用率动态调整并行数
    if initial_cpu_usage < 30:  # CPU使用率低于30%
        max_workers = min(cpu_cores // 8, len(mp4_files))  # 每个任务8线程
    elif initial_cpu_usage < 60:  # CPU使用率30-60%
        max_workers = min(cpu_cores // 12, len(mp4_files))
    else:  # CPU使用率高于60%
        max_workers = 1
    
    max_workers = max(1, max_workers)  # 至少1个worker
    
    print(f"使用 {max_workers} 个CPU进程并行分割视频 (每个使用8线程)")
    
    def segment_single_video(vid):
        """分割单个视频文件"""
        input_file = os.path.join(org_path, vid)
        original_filename = os.path.basename(input_file)

        command = [
            'ffmpeg', '-threads', '8', '-i', input_file, '-c', 'copy', '-map', '0',
            '-segment_time', str(segment_duration), '-f', 'segment',
            '-reset_timestamps', '1',
            os.path.join(dst_path, f'clip%03d_{original_filename}')
        ]

        try:
            subprocess.run(command, check=True)
            print(f"成功分割视频: {vid}")
        except subprocess.CalledProcessError as e:
            print(f"分割视频 {vid} 时出现错误: {e}")
    
    # 使用线程池并行处理视频分割
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(segment_single_video, vid) for vid in mp4_files]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"视频分割过程中出现错误: {e}")
    
    print("所有视频分割完成")

def extract_audio(org_path: str, dst_path: str, vid_list: List[str]) -> None:
    """
    Extract audio from video files and save as WAV format.

    Parameters:
    org_path (str): The directory containing the original video files.
    dst_path (str): The directory where the extracted audio files will be saved.
    vid_list (List[str]): A list of video file names to process.

    Returns:
    None
    """
    # 过滤出MP4文件
    mp4_files = [vid for vid in vid_list if vid.endswith('.mp4')]
    
    if not mp4_files:
        print("没有找到需要提取音频的MP4文件")
        return
    
    # 确定CPU并行数量
    cpu_cores = psutil.cpu_count(logical=True)  # 逻辑核心数
    initial_cpu_usage = get_cpu_usage()
    
    # 根据CPU使用率动态调整并行数
    if initial_cpu_usage < 30:  # CPU使用率低于30%
        max_workers = min(cpu_cores // 8, len(mp4_files))  # 每个任务8线程
    elif initial_cpu_usage < 60:  # CPU使用率30-60%
        max_workers = min(cpu_cores // 12, len(mp4_files))
    else:  # CPU使用率高于60%
        max_workers = 1
    
    max_workers = max(1, max_workers)  # 至少1个worker
    
    print(f"使用 {max_workers} 个CPU进程并行提取音频 (每个使用8线程)")
    
    def extract_single_audio(vid):
        """提取单个视频文件的音频"""
        video_path = os.path.join(org_path, vid)
        audio_output_path = os.path.join(dst_path, os.path.splitext(vid)[0] + ".wav")
        
        command = [
            'ffmpeg', '-threads', '8', '-hide_banner', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-f', 'wav',
            '-ar', '16000', '-ac', '1', audio_output_path,
        ]
        
        try:
            subprocess.run(command, check=True)
            print(f"音频已保存到: {audio_output_path}")
        except subprocess.CalledProcessError as e:
            print(f"从视频 {vid} 提取音频时出现错误: {e}")
    
    # 使用线程池并行处理音频提取
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(extract_single_audio, vid) for vid in mp4_files]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"音频提取过程中出现错误: {e}")
    
    print("所有音频提取完成")

def split_data(video_files: List[str], val_list_hdtf: List[str]) -> (List[str], List[str]):
    """
    Split video files into training and validation sets based on val_list_hdtf.

    Parameters:
    video_files (List[str]): A list of video file names.
    val_list_hdtf (List[str]): A list of validation file identifiers.

    Returns:
    (List[str], List[str]): A tuple containing the training and validation file lists.
    """
    val_files = [f for f in video_files if any(val_id in f for val_id in val_list_hdtf)]
    train_files = [f for f in video_files if f not in val_files]
    return train_files, val_files

def save_list_to_file(file_path: str, data_list: List[str]) -> None:
    """
    Save a list of strings to a file, each string on a new line.

    Parameters:
    file_path (str): The path to the file where the list will be saved.
    data_list (List[str]): The list of strings to save.

    Returns:
    None
    """
    with open(file_path, 'w') as file:
        for item in data_list:
            file.write(f"{item}\n")

def generate_train_list(cfg):
    train_file_path = cfg.video_clip_file_list_train
    val_file_path = cfg.video_clip_file_list_val
    val_list_hdtf = cfg.val_list_hdtf

    meta_list = os.listdir(cfg.meta_root)

    sorted_meta_list = sorted(meta_list)
    train_files, val_files = split_data(meta_list, val_list_hdtf)

    save_list_to_file(train_file_path, train_files)
    save_list_to_file(val_file_path, val_files)

    print(val_list_hdtf)    

def analyze_video(org_path: str, dst_path: str, vid_list: List[str]) -> None:
    """
    Convert video files to a specified format and save them to the destination path.

    Parameters:
    org_path (str): The directory containing the original video files.
    dst_path (str): The directory where the meta json will be saved.
    vid_list (List[str]): A list of video file names to process.

    Returns:
    None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_file = './musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
    checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'
    
    # 检查GPU显存，如果大于16GB则创建四个analyze_face实例
    use_multiple_analyzers = False
    num_analyzers = 1
    if device == "cuda":
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # 转换为GB
        if gpu_memory > 16:
            use_multiple_analyzers = True
            num_analyzers = 4
            print(f"检测到GPU显存 {gpu_memory:.1f}GB > 16GB，启用{num_analyzers}个analyze_face实例优化")
    
    # 创建多个analyze_face实例
    analyzers = []
    if use_multiple_analyzers:
        for i in range(num_analyzers):
            analyzers.append(AnalyzeFace(device, config_file, checkpoint_file))
    else:
        analyzers.append(AnalyzeFace(device, config_file, checkpoint_file))

    def process_single_video(vid, thread_id=None):
        """处理单个视频文件"""
        
        # 根据线程ID选择对应的analyze_face实例
        if use_multiple_analyzers:
            current_analyze_face = analyzers[thread_id % num_analyzers]
        else:
            current_analyze_face = analyzers[0]
        
        if vid.endswith('.mp4'):
            vid_path = os.path.join(org_path, vid)
            wav_path = vid_path.replace(".mp4",".wav")
            vid_meta = os.path.join(dst_path, os.path.splitext(vid)[0] + ".json")
            if os.path.exists(vid_meta):
                return f"跳过已存在的文件: {vid}"
            print('process video {}'.format(vid))

            total_bbox_list = []
            total_pts_list = []
            isvalid = True

            # process
            try:
                cap = decord.VideoReader(vid_path, fault_tol=1)
            except Exception as e:
                return f"读取视频失败 {vid}: {e}"

            total_frames = len(cap)
            for frame_idx in range(total_frames):
                frame = cap[frame_idx]
                if frame_idx==0:
                    video_height,video_width,_ = frame.shape
                frame_bgr = cv2.cvtColor(frame.asnumpy(), cv2.COLOR_BGR2RGB)
                pts_list, bbox_list = current_analyze_face(frame_bgr)

                if len(bbox_list)>0 and None not in bbox_list:
                    bbox = bbox_list[0]
                else:
                    isvalid = False
                    bbox = []
                    print(f"set isvalid to False as broken img in {frame_idx} of {vid}")
                    break

                #print(pts_list)
                if len(pts_list)>0 and pts_list is not None:
                    pts = pts_list.tolist()
                else:
                    isvalid = False
                    pts = []
                    break

                if frame_idx==0:
                    x1,y1,x2,y2 = bbox 
                    face_height, face_width = y2-y1,x2-x1

                total_pts_list.append(pts)
                total_bbox_list.append(bbox)

            meta_data = {
                    "mp4_path": vid_path,
                     "wav_path": wav_path,
                     "video_size": [video_height, video_width],
                     "face_size": [face_height, face_width],
                     "frames": total_frames,
                     "face_list": total_bbox_list,
                     "landmark_list": total_pts_list,
                     "isvalid":isvalid,
            }        
            with open(vid_meta, 'w') as f:
                json.dump(meta_data, f, indent=4)
            
            return f"成功处理视频: {vid}"
        else:
            return f"跳过非MP4文件: {vid}"

    # 获取CPU核心数并计算线程数
    cpu_count = psutil.cpu_count(logical=False)  # 物理核心数
    max_workers = max(1, cpu_count // 24)  # CPU总数//24，至少为1
    
    print(f"检测到 {cpu_count} 个CPU核心，使用 {max_workers} 个线程进行并行处理")
    
    # 使用线程池并行处理视频
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor: # 线程是 ThreadPoolExecutor
        # 提交所有任务，并为每个任务分配线程ID
        future_to_vid = {}
        for i, vid in enumerate(vid_list):
            future = executor.submit(process_single_video, vid, i)
            future_to_vid[future] = vid
        
        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(future_to_vid), 
                          total=len(vid_list), desc="处理视频"):
            vid = future_to_vid[future]
            try:
                result = future.result()
                if "成功处理" in result:
                    print(result)
            except Exception as e:
                print(f"处理视频 {vid} 时出现异常: {e}")


def main(cfg):
    # Ensure all necessary directories exist
    os.makedirs(cfg.video_root_25fps, exist_ok=True)
    os.makedirs(cfg.video_audio_clip_root, exist_ok=True)
    os.makedirs(cfg.meta_root, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.video_file_list), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.video_clip_file_list_train), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.video_clip_file_list_val), exist_ok=True)

    vid_list = os.listdir(cfg.video_root_raw)
    sorted_vid_list = sorted(vid_list)
 
    # Save video file list
    with open(cfg.video_file_list, 'w', encoding='utf-8') as file:
        for vid in sorted_vid_list:
            file.write(vid + '\n')

    # 1. Convert videos to 25 FPS
    convert_video(cfg.video_root_raw, cfg.video_root_25fps, sorted_vid_list)
    
    # 2. Segment videos into 30-second clips
    segment_video(cfg.video_root_25fps, cfg.video_audio_clip_root, vid_list, segment_duration=cfg.clip_len_second)
    
    # # 3. Extract audio
    clip_vid_list = os.listdir(cfg.video_audio_clip_root)
    extract_audio(cfg.video_audio_clip_root, cfg.video_audio_clip_root, clip_vid_list)
    
    # 4. Generate video metadata
    analyze_video(cfg.video_audio_clip_root, cfg.meta_root, clip_vid_list)
    
    # 5. Generate training and validation set lists
    generate_train_list(cfg)
    print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/training/preprocess.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    main(config)
    
