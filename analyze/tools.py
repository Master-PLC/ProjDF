import os
import pickle
import random
import re
import shutil
import time
from itertools import product

import cupy as cp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
import tarfile

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from matplotlib.backends.backend_pdf import PdfPages
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cp.random.seed(seed)


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return {"result": result, "time": end_time - start_time}
    return wrapper


def load_yaml_fast(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=Loader)


def load_npy(path):
    return np.load(path)


def load_pkl(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def load_yaml_as_df(path):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    df = pd.json_normalize(data)
    return df


def exist_metric(exp_dir):
    try:
        res_dir = os.path.join(exp_dir, 'results')
        setting = os.listdir(res_dir)[0]
        setting_dir = os.path.join(res_dir, setting)
        metric_path = os.path.join(setting_dir, 'metrics.npy')
        metric_yaml_path = os.path.join(setting_dir, 'metrics.yaml')
        if os.path.exists(metric_path) or os.path.exists(metric_yaml_path):
            return True, setting_dir
        else:
            return False, setting_dir
    except Exception as e:
        return False, None


def exist_pred(exp_dir):
    try:
        
        res_dir = os.path.join(exp_dir, 'results')
        setting = os.listdir(res_dir)[0]
        setting_dir = os.path.join(res_dir, setting)
        metric_path = os.path.join(setting_dir, 'pred.npy')
        if os.path.exists(metric_path):
            return True, setting_dir
        else:
            return False, None
    except Exception as e:
        return False, None


def exist_stf_metric(exp_dir):
    try:
        res_dir = os.path.join(exp_dir, 'results')
        metric_dir = os.path.join(res_dir, "m4_results")
        metric_path = os.path.join(metric_dir, 'metrics.pkl')
        if os.path.exists(metric_path):
            settings = os.listdir(res_dir)
            setting = [s for s in settings if 'Hourly' in s][0]
            setting_dir = os.path.join(res_dir, setting)
            return True, metric_dir, setting_dir
        else:
            return False, None, None
    except Exception as e:
        return False, None, None


def inverse_stf_metrics(metrics, names):
    new_metrics = {}
    for key, value in metrics.items():
        if key not in names:
            continue
        for k, v in value.items():
            if k in new_metrics:
                new_metrics[k][key] = v
            else:
                new_metrics[k] = {key: v}
    return new_metrics


def keep_split(exp, special_words=[]):
    pattern = '|'.join(map(re.escape, special_words)) + '|_'
    parts = re.findall(f'({pattern})|([^_]+)', exp)
    result = [part[0] or part[1] for part in parts if any(part) and (part[0] or part[1]) != '_']
    output = []
    for part in result:
        try: part = eval(part)
        except: pass
        output.append(part)
    return output


def is_full_group(x):
    if str(x['data_id'].iloc[0]).startswith('PEMS'):
        return set(x['pred_len']) == {12, 24, 36, 48}
    else:
        return set(x['pred_len']) == {96, 192, 336, 720}


def load_metric_from_log(log_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
    for line in lines[::-1]:
        if line.strip() == '':
            continue

        if 'mse' in line and 'mae' in line:
            parts = line.strip().split(', ')
            metrics = {}
            for part in parts:
                key_value = part.split(':')
                if len(key_value) == 2:
                    key = key_value[0].strip()
                    try:
                        value = float(key_value[1].strip())
                    except ValueError:
                        value = key_value[1].strip()
                    metrics[key] = value
            return metrics
    return None


def restore_folder(target_folder, remove_archive=False):
    # 1. 构造压缩包的完整路径
    archive_path = os.path.join(target_folder, "archive.tar.xz")
    
    # 2. 检查文件是否存在
    if not os.path.exists(archive_path):
        print(f"错误: 在该路径下找不到 archive.tar.xz -> {target_folder}")
        return

    try:
        print(f"正在解压: {target_folder} ...")
        
        # 3. 打开压缩包 ('r:xz' 表示读取 xz 压缩格式)
        with tarfile.open(archive_path, "r:xz") as tar:
            # 4. 解压到指定目录 (path 参数相当于 tar 命令的 -C)
            # 警告：在 Python 3.12+ 中，出于安全考虑，可能需要添加 filter='data'
            # tar.extractall(path=target_folder, filter='data') 
            tar.extractall(path=target_folder)
            
        print("✅ 解压成功！")
        
        # 5. (可选) 解压后删除压缩包
        if remove_archive:
            os.remove(archive_path)
            print("已删除压缩包文件")
        
    except Exception as e:
        print(f"❌ 解压出错: {str(e)}")


def read_tensorboard_file(path, load_image=False):
    # 1. 设置 size_guidance
    # 'images': 0 表示加载该 tag 下所有的图片数据，不进行采样或截断
    if load_image:
        ea = EventAccumulator(path, size_guidance={'images': 0})
    else:
        ea = EventAccumulator(path)
    ea.Reload()

    data_dict = {}
    
    # --- 读取 Scalars (Loss, Accuracy 等) ---
    # 先判断是否存在 scalars，防止报错
    if 'scalars' in ea.Tags():
        for tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            data_dict[tag] = [
                {"step": e.step, "value": e.value, "time": e.wall_time, "type": "scalar"} 
                for e in events
            ]

    # --- 读取 Images (包括 add_figure 加入的内容) ---
    if 'images' in ea.Tags() and load_image:
        for tag in ea.Tags()['images']:
            events = ea.Images(tag)
            data_dict[tag] = []
            
            for e in events:
                # e 包含: wall_time, step, encoded_image_string, width, height
                data_dict[tag].append({
                    "step": e.step,
                    "value": e.encoded_image_string,   # 这里存的是 PIL.Image 对象
                    "time": e.wall_time,
                    "width": e.width,
                    "height": e.height,
                    "type": "bytes"
                })
        
    return data_dict


def tensorboard_smoothing(values, weight=0.6):
    """
    严格模拟 TensorBoard 前端处理逻辑 (包含偏差修正)
    """
    last = 0
    smoothed = []
    num_accum = 0
    
    for point in values:
        last = last * weight + point * (1 - weight)
        num_accum += 1
        
        # 偏差修正: 
        # 在初期，由于 last 从 0 开始，会导致平滑值偏小。
        # 需要除以 (1 - weight^steps) 来修正。
        debias_weight = 1
        if weight != 1.0:
            debias_weight = 1.0 - (weight ** num_accum)
            
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)

    return smoothed
