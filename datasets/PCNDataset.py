import torch.utils.data as data
import numpy as np
import os
import torch
import open3d as o3d
import json
import random
import transforms3d  # 需要安装: pip install transforms3d
from .build import DATASETS

@DATASETS.register_module()
class TreeDataset(data.Dataset):
    def __init__(self, config, subset=None, generate_views=False):
        self.subset = subset or config.subset
        self.data_root = config.DATA_ROOT
        self.npoints = config.N_POINTS
        self.complete_points = config.N_COMPLETE_POINTS
        self.skeleton_points = config.SKELETON_POINTS
        self.data_list = self._get_file_list()
        self.max_retries = 5
        self.apply_rotation = (self.subset == 'train')  # 仅训练时应用变换
        if generate_views:
            self.generate_fixed_views()
    
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
                partial_path = os.path.join(self.subset, 'partial', tax_id, model_id, '00.pcd')  # 只读取00.pcd
                complete_path = os.path.join(self.subset, 'complete', tax_id, f'{model_id}.pcd')
                skeleton_path = os.path.join(self.subset, 'skeleton', tax_id, f'{model_id}_skeleton.ply')
                
                full_partial_path = os.path.join(self.data_root, partial_path)
                full_complete_path = os.path.join(self.data_root, complete_path)
                full_skeleton_path = os.path.join(self.data_root, skeleton_path)
                
                if not (os.path.exists(full_partial_path) and os.path.exists(full_complete_path) and os.path.exists(full_skeleton_path)):
                    print(f"[TreeDataset] 警告：跳过 {self.subset}/{tax_id}/{model_id}，文件缺失")
                    continue
                
                try:
                    partial_pc = np.asarray(o3d.io.read_point_cloud(full_partial_path).points, dtype=np.float32)
                    complete_pc = np.asarray(o3d.io.read_point_cloud(full_complete_path).points, dtype=np.float32)
                    skeleton_pc = np.asarray(o3d.io.read_point_cloud(full_skeleton_path).points, dtype=np.float32)
                    if partial_pc.shape[0] == 0 or complete_pc.shape[0] == 0 or skeleton_pc.shape[0] == 0:
                        print(f"[TreeDataset] 警告：{self.subset}/{tax_id}/{model_id} 中点云为空")
                        continue
                    data_list.append((partial_path, complete_path, skeleton_path, tax_id, model_id))
                except Exception as e:
                    print(f"[TreeDataset] 警告：读取 {self.subset}/{tax_id}/{model_id} 失败：{str(e)}")
                    continue
        
        if not data_list:
            raise ValueError(f"[TreeDataset] 未找到 {self.subset} 的有效文件")
        print(f"[TreeDataset] 完成 {self.subset} 文件收集，共 {len(data_list)} 个文件")
        return data_list
    
    def generate_fixed_views(self):
        print("[TreeDataset] 开始生成7个固定视角点云...")
        viewpoints = [
            (0, 0, 0),  # 0° (原始)
            (np.pi / 2, 0, 0),  # x轴90°
            (np.pi, 0, 0),  # x轴180°
            (3 * np.pi / 2, 0, 0),  # x轴270°
            (0, np.pi / 2, 0),  # y轴90°
            (0, np.pi, 0),  # y轴180°
            (0, 3 * np.pi / 2, 0),  # y轴270°
            (0, 0, np.pi / 2)  # z轴90°
        ]
        
        for item in self.data_list:
            tax_id = item[3]
            model_id = item[4]
            partial_dir = os.path.join(self.data_root, self.subset, 'partial', tax_id, model_id)
            partial_base_path = os.path.join(self.data_root, self.subset, 'partial', tax_id, model_id, '00.pcd')
            
            if not os.path.exists(partial_base_path):
                print(f"警告：跳过 {partial_base_path}，文件不存在")
                continue
            
            pcd = o3d.io.read_point_cloud(partial_base_path)
            points = np.asarray(pcd.points)
            
            for i, angles in enumerate(viewpoints):
                rotation_matrix = transforms3d.euler.euler2mat(*angles)
                rotated_points = np.dot(points, rotation_matrix.T)
                rotated_pcd = o3d.geometry.PointCloud()
                rotated_pcd.points = o3d.utility.Vector3dVector(rotated_points)
                
                output_path = os.path.join(partial_dir, f'{i:02d}.pcd')
                o3d.io.write_point_cloud(output_path, rotated_pcd)
                print(f"生成 {output_path}")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx, retries=0):
        if retries >= self.max_retries:
            print(f"[TreeDataset] 错误：超过最大重试次数 ({self.max_retries})，索引 {idx}，返回空数据")
            return (
                '',  # taxonomy_id
                '',  # model_id
                {
                    'input': torch.zeros((self.npoints, 3)).float(),
                    'gt': torch.zeros((self.complete_points, 3)).float(),
                    'skeleton': torch.zeros((self.skeleton_points, 3)).float()
                }
            )
        
        partial_path, complete_path, skeleton_path, taxonomy_id, model_id = self.data_list[idx]
        # print(f"[TreeDataset] 加载索引 {idx}: partial={partial_path}, complete={complete_path}, skeleton={skeleton_path}")
        
        try:
            partial_pc = np.asarray(o3d.io.read_point_cloud(os.path.join(self.data_root, partial_path)).points, dtype=np.float32)
            complete_pc = np.asarray(o3d.io.read_point_cloud(os.path.join(self.data_root, complete_path)).points, dtype=np.float32)
            skeleton_pc = np.asarray(o3d.io.read_point_cloud(os.path.join(self.data_root, skeleton_path)).points, dtype=np.float32)
        except Exception as e:
            print(f"[TreeDataset] 读取失败：{partial_path}, {complete_path}, {skeleton_path}，错误：{str(e)}")
            return self.__getitem__(random.randint(0, len(self.data_list)-1), retries + 1)
        
        if partial_pc.shape[0] == 0 or complete_pc.shape[0] == 0 or skeleton_pc.shape[0] == 0:
            print(f"[TreeDataset] 警告：{partial_path} 中点云为空")
            return self.__getitem__(random.randint(0, len(self.data_list)-1), retries + 1)
        
        # 应用相同的随机旋转变换（仅训练时）
        if self.apply_rotation:
            angle = np.random.uniform(-np.pi/6, np.pi/6, 3)  # 随机角度[-30°, 30°]
            rotation_matrix = transforms3d.euler.euler2mat(*angle)
            partial_pc = np.dot(partial_pc, rotation_matrix.T)
            skeleton_pc = np.dot(skeleton_pc, rotation_matrix.T)
            # 注意：complete_pc (gt)不变换，以保持真实参考，但如果需要，可以同样变换
        
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
        
        return (
            taxonomy_id,
            model_id,
            {
                'input': torch.from_numpy(partial_pc).float(),
                'gt': torch.from_numpy(complete_pc).float(),
                'skeleton': torch.from_numpy(skeleton_pc).float()
            }
        )
    
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