import torch.utils.data as data
import numpy as np
import os, sys
import torch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import data_transforms
from .io import IO
import random
import json
from .build import DATASETS
from utils.logger import *
import open3d as o3d

@DATASETS.register_module()
class PCN(data.Dataset):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.skeleton_points_path = config.get('SKELETON_POINTS_PATH', None)  # Optional skeleton path
        self.category_file = config.CATEGORY_FILE_PATH
        self.n_points = config.N_POINTS
        self.n_complete_points = config.get('N_COMPLETE_POINTS', 16384)
        self.skeleton_points = config.get('SKELETON_POINTS', 512)
        self.subset = config.subset
        self.cars = config.CARS

        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            if config.CARS:
                self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']

        self.n_renderings = 8 if self.subset == 'train' else 1
        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {'n_points': self.n_points},
                'objects': ['partial']
            }, {
                'callback': 'RandomSamplePoints',
                'parameters': {'n_points': self.n_complete_points},
                'objects': ['gt']
            }, {
                'callback': 'RandomSamplePoints',
                'parameters': {'n_points': self.skeleton_points},
                'objects': ['skeleton']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial', 'gt', 'skeleton']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt', 'skeleton']
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {'n_points': self.n_points},
                'objects': ['partial']
            }, {
                'callback': 'RandomSamplePoints',
                'parameters': {'n_points': self.n_complete_points},
                'objects': ['gt']
            }, {
                'callback': 'RandomSamplePoints',
                'parameters': {'n_points': self.skeleton_points},
                'objects': ['skeleton']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt', 'skeleton']
            }])

    def _get_file_list(self, subset, n_renderings=1):
        file_list = []
        for dc in self.dataset_categories:
            print_log(f'Collecting files of Taxonomy [ID={dc["taxonomy_id"]}, Name={dc["taxonomy_name"]}]', logger='PCNDATASET')
            samples = dc[subset]
            for s in samples:
                item = {
                    'taxonomy_id': dc['taxonomy_id'],
                    'model_id': s,
                    'partial_path': [
                        self.partial_points_path % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                    'gt_path': self.complete_points_path % (subset, dc['taxonomy_id'], s)
                }
                if self.skeleton_points_path:
                    item['skeleton_path'] = self.skeleton_points_path % (subset, dc['taxonomy_id'], s)
                file_list.append(item)
        print_log(f'Complete collecting files of the dataset. Total files: {len(file_list)}', logger='PCNDATASET')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset == 'train' else 0

        for ri in ['partial', 'gt', 'skeleton']:
            file_path = sample[f'{ri}_path']
            if ri == 'skeleton' and not self.skeleton_points_path:
                data[ri] = None
                continue
            if type(file_path) == list:
                file_path = file_path[rand_idx]
            try:
                data[ri] = IO.get(file_path).astype(np.float32)
            except Exception as e:
                print_log(f"Error reading file {file_path}: {str(e)}", logger='PCNDATASET')
                return self.__getitem__((idx + 1) % len(self.file_list))

        centroid = np.mean(data['partial'], axis=0)
        m = np.max(np.abs(data['partial'] - centroid))
        for ri in ['partial', 'gt', 'skeleton']:
            if data[ri] is not None:
                data[ri] = (data[ri] - centroid) / m

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], {
            'input': data['partial'],
            'gt': data['gt'],
            'skeleton': data['skeleton']
        }

    def __len__(self):
        return len(self.file_list)

@DATASETS.register_module()
class PCNv2(PCN):
    def __init__(self, config):
        super().__init__(config)

@DATASETS.register_module()
class TreeDataset(data.Dataset):
    def __init__(self, config, subset):
        self.subset = subset  # train, val, or test
        self.data_root = config.DATA_ROOT
        self.npoints = config.N_POINTS
        self.complete_points = config.N_COMPLETE_POINTS
        self.skeleton_points = config.SKELETON_POINTS
        self.data_list = self._get_file_list()
        self.max_retries = 5  # 最大重试次数
        
    def _get_file_list(self):
        data_list = []
        json_path = os.path.join(self.data_root, 'PCN.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"[TreeDataset] PCN.json 未找到：{json_path}")
        
        with open(json_path, 'r') as f:
            data_json = json.load(f)
        
        for data in data_json:
            tax_id = data['taxonomy_id']
            if self.subset not in data:
                raise ValueError(f"[TreeDataset] 子集 {self.subset} 在 PCN.json 中未定义 for taxonomy_id {tax_id}")
            
            for model_id in data.get(self.subset, []):
                partial_path = os.path.join(self.subset, 'partial', tax_id, model_id, '00.pcd')
                complete_path = os.path.join(self.subset, 'complete', tax_id, f'{model_id}.pcd')
                skeleton_path = os.path.join(self.subset, 'skeleton', tax_id, f'{model_id}_skeleton.ply')
                
                # 验证文件存在
                full_partial_path = os.path.join(self.data_root, partial_path)
                full_complete_path = os.path.join(self.data_root, complete_path)
                full_skeleton_path = os.path.join(self.data_root, skeleton_path)
                
                if not (os.path.exists(full_partial_path) and os.path.exists(full_complete_path) and os.path.exists(full_skeleton_path)):
                    print(f"[TreeDataset] 警告：跳过 {self.subset}/{tax_id}/{model_id}，文件缺失："
                          f"partial={full_partial_path} ({os.path.exists(full_partial_path)}), "
                          f"complete={full_complete_path} ({os.path.exists(full_complete_path)}), "
                          f"skeleton={full_skeleton_path} ({os.path.exists(full_skeleton_path)})")
                    continue
                
                # 验证点云非空
                try:
                    partial_pc = np.asarray(o3d.io.read_point_cloud(full_partial_path).points, dtype=np.float32)
                    complete_pc = np.asarray(o3d.io.read_point_cloud(full_complete_path).points, dtype=np.float32)
                    skeleton_pc = np.asarray(o3d.io.read_point_cloud(full_skeleton_path).points, dtype=np.float32)
                    
                    if partial_pc.shape[0] == 0 or complete_pc.shape[0] == 0 or skeleton_pc.shape[0] == 0:
                        print(f"[TreeDataset] 警告：{self.subset}/{tax_id}/{model_id} 中点云为空："
                              f"partial={partial_pc.shape[0]} 点, complete={complete_pc.shape[0]} 点, skeleton={skeleton_pc.shape[0]} 点")
                        continue
                    data_list.append((partial_path, complete_path, skeleton_path))
                except Exception as e:
                    print(f"[TreeDataset] 警告：读取 {self.subset}/{tax_id}/{model_id} 失败：{str(e)}")
                    continue
        
        if not data_list:
            raise ValueError(f"[TreeDataset] 未找到 {self.subset} 的有效文件")
        print(f"[TreeDataset] 完成 {self.subset} 文件收集，共 {len(data_list)} 个文件")
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx, retries=0):
        if retries >= self.max_retries:
            print(f"[TreeDataset] 错误：超过最大重试次数 ({self.max_retries})，索引 {idx}，返回空数据")
            return {
                'input': torch.zeros((self.npoints, 3)).float(),
                'gt': torch.zeros((self.complete_points, 3)).float(),
                'skeleton': torch.zeros((self.skeleton_points, 3)).float()
            }
        
        partial_path, complete_path, skeleton_path = self.data_list[idx]
        try:
            partial_pc = np.asarray(o3d.io.read_point_cloud(os.path.join(self.data_root, partial_path)).points, dtype=np.float32)
            complete_pc = np.asarray(o3d.io.read_point_cloud(os.path.join(self.data_root, complete_path)).points, dtype=np.float32)
            skeleton_pc = np.asarray(o3d.io.read_point_cloud(os.path.join(self.data_root, skeleton_path)).points, dtype=np.float32)
        except Exception as e:
            print(f"[TreeDataset] 读取 {partial_path} 失败：{str(e)}")
            return self.__getitem__(random.randint(0, len(self.data_list)-1), retries + 1)

        # 检查空点云
        if partial_pc.shape[0] == 0 or complete_pc.shape[0] == 0 or skeleton_pc.shape[0] == 0:
            print(f"[TreeDataset] 警告：{partial_path} 中点云为空："
                  f"partial={partial_pc.shape[0]} 点, complete={complete_pc.shape[0]} 点, skeleton={skeleton_pc.shape[0]} 点")
            return self.__getitem__(random.randint(0, len(self.data_list)-1), retries + 1)
        
        # 归一化点云
        try:
            partial_pc = self._normalize_pc(partial_pc)
            complete_pc = self._normalize_pc(complete_pc)
            skeleton_pc = self._normalize_pc(skeleton_pc)
        except ValueError as e:
            print(f"[TreeDataset] 警告：{partial_path} 归一化失败：{str(e)}")
            return self.__getitem__(random.randint(0, len(self.data_list)-1), retries + 1)
        
        # 采样点
        try:
            partial_pc = self._sample_points(partial_pc, self.npoints)
            complete_pc = self._sample_points(complete_pc, self.complete_points)
            skeleton_pc = self._sample_points(skeleton_pc, self.skeleton_points)
        except Exception as e:
            print(f"[TreeDataset] 警告：{partial_path} 采样失败：{str(e)}")
            return self.__getitem__(random.randint(0, len(self.data_list)-1), retries + 1)
        
        return {
            'input': torch.from_numpy(partial_pc).float(),
            'gt': torch.from_numpy(complete_pc).float(),
            'skeleton': torch.from_numpy(skeleton_pc).float()
        }
    
    def _normalize_pc(self, pc):
        if pc.shape[0] == 0:
            raise ValueError("无法归一化空点云")
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        if m == 0:
            raise ValueError("无法归一化最大距离为零的点云")
        pc = pc / m
        return pc
    
    def _sample_points(self, pc, n_points):
        if pc.shape[0] > n_points:
            indices = np.random.choice(pc.shape[0], n_points, replace=False)
            return pc[indices]
        elif pc.shape[0] < n_points:
            indices = np.random.choice(pc.shape[0], n_points, replace=True)
            return pc[indices]
        return pc