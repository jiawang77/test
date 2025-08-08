import os
import numpy as np
import torch
from datasets.data_transforms import Compose
from datasets.io import IO
import cv2
from tools import builder
from utils.config import cfg_from_yaml_file
from utils import misc

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_config', help='yaml config file')
    parser.add_argument('ckpts', help='ckpt path')
    parser.add_argument('--pc_root', default='PCN/test/partial', help='Pc root')
    parser.add_argument('--skeleton_pc_root', default='PCN/test/skeleton', help='Skeleton point cloud root')
    parser.add_argument('--out_pc_root', default='', help='Output point cloud root')
    parser.add_argument('--device', default='cuda:0', help='Device for testing')
    parser.add_argument('--save_vis_img', action='store_true')
    return parser.parse_args()

def inference_single(model, pc_path, args, config, root=None):
    if root is not None:
        pc_file = os.path.join(root, pc_path)
        # pc_path = <category>/<model_id>/<view_id>.pcd
        category, model_id, _ = pc_path.split('/')
        skeleton_file = os.path.join(args.skeleton_pc_root, category, f'{model_id}_skeleton.ply')
    else:
        pc_file = pc_path
        skeleton_file = args.skeleton_pc

    pc_ndarray = IO.get(pc_file).astype(np.float32)
    skeleton_ndarray = IO.get(skeleton_file).astype(np.float32)

    # 归一化
    centroid = np.mean(pc_ndarray, axis=0)
    pc_ndarray = pc_ndarray - centroid
    m = np.max(np.abs(pc_ndarray))
    pc_ndarray = pc_ndarray / m
    skeleton_ndarray = (skeleton_ndarray - centroid) / m

    transform = Compose([{
        'callback': 'UpSamplePoints',
        'parameters': {'n_points': 2048},
        'objects': ['input']
    }, {
        'callback': 'ToTensor',
        'objects': ['input']
    }])
    skeleton_transform = Compose([{
        'callback': 'ToTensor',
        'objects': ['input']
    }])

    pc_ndarray_normalized = transform({'input': pc_ndarray})
    skeleton_ndarray_normalized = skeleton_transform({'input': skeleton_ndarray})

    model.eval()
    with torch.no_grad():
        ret = model(pc_ndarray_normalized['input'].unsqueeze(0).to(args.device),
                    skeleton_ndarray_normalized['input'].unsqueeze(0).to(args.device))
    dense_points = ret[-1].squeeze(0).detach().cpu().numpy()
    dense_points = dense_points * m + centroid

    if args.out_pc_root != '':
        target_path = os.path.join(args.out_pc_root, os.path.dirname(pc_path))
        os.makedirs(target_path, exist_ok=True)
        np.save(os.path.join(target_path, 'fine.npy'), dense_points)
        np.save(os.path.join(target_path, 'skeleton.npy'), skeleton_ndarray)
        if args.save_vis_img:
            input_img = misc.get_ptcloud_img(pc_ndarray)
            dense_img = misc.get_ptcloud_img(dense_points)
            skeleton_img = misc.get_ptcloud_img(skeleton_ndarray)
            cv2.imwrite(os.path.join(target_path, 'input.jpg'), input_img)
            cv2.imwrite(os.path.join(target_path, 'fine.jpg'), dense_img)
            cv2.imwrite(os.path.join(target_path, 'skeleton.jpg'), skeleton_img)

def main():
    args = get_args()
    config = cfg_from_yaml_file(args.model_config)
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts)
    if args.use_gpu:
        base_model.to(args.device)
    base_model.eval()

    if args.pc_root != '':
        for category in os.listdir(args.pc_root):
            category_path = os.path.join(args.pc_root, category)
            if not os.path.isdir(category_path):
                continue
            for model_id in os.listdir(category_path):
                model_path = os.path.join(category_path, model_id)
                for pc_file in os.listdir(model_path):
                    if pc_file.endswith('.pcd'):
                        inference_single(base_model, os.path.join(category, model_id, pc_file), args, config, root=args.pc_root)
    else:
        inference_single(base_model, args.ckpts, args, config)

if __name__ == '__main__':
    main()