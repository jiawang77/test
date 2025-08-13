import torch
from torch import nn
from pointnet2_ops import pointnet2_utils
from extensions.chamfer_dist import ChamferDistanceL1
from .Transformer import PCTransformer
from .build import MODELS
from .AdaPoinTr import Fold
from torch_geometric.nn import GraphConv
from torch_geometric.nn import knn_graph

def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num)
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return sub_pc

class GraphSkeletonEncoder(nn.Module):
    def __init__(self, skeleton_dim):
        super().__init__()
        self.conv1 = GraphConv(3, 64)
        self.relu = nn.ReLU()
        self.conv2 = GraphConv(64, skeleton_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x

class PoinTr(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.trans_dim
        self.knn_layer = config.knn_layer
        self.num_pred = config.num_pred
        self.num_query = config.num_query
        self.fold_step = int(pow(self.num_pred // self.num_query, 0.5) + 0.5)
        self.base_model = PCTransformer(in_chans=3, embed_dim=self.trans_dim, depth=[6, 8], drop_rate=0., num_query=self.num_query, knn_layer=self.knn_layer)
        self.foldingnet = Fold(self.trans_dim, step=self.fold_step, hidden_dim=256)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()

    def get_loss(self, ret, gt, epoch=0):
        coarse_points, dense_points = ret
        sparse_loss = self.loss_func(coarse_points, gt)
        dense_loss = self.loss_func(dense_points, gt)
        return sparse_loss, dense_loss

    def forward(self, xyz):
        q, coarse_point_cloud = self.base_model(xyz)  # B M C and B M 3
        B, M, C = q.shape
        global_feature = self.increase_dim(q.transpose(1, 2)).transpose(1, 2)  # B M 1024
        global_feature = torch.max(global_feature, dim=1)[0]  # B 1024
        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse_point_cloud], dim=-1)  # B M 1027 + C
        rebuild_feature = self.reduce_map(rebuild_feature.reshape(B * M, -1))  # BM C
        relative_xyz = self.foldingnet(rebuild_feature).reshape(B, M, 3, -1)  # B M 3 S
        rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2, 3).reshape(B, -1, 3)  # B N 3
        inp_sparse = fps(xyz, self.num_query)
        coarse_point_cloud = torch.cat([coarse_point_cloud, inp_sparse], dim=1).contiguous()
        rebuild_points = torch.cat([rebuild_points, xyz], dim=1).contiguous()
        ret = (coarse_point_cloud, rebuild_points)
        return ret

@MODELS.register_module()
class TreePoinTr(PoinTr):
    def __init__(self, config):
        super().__init__(config)
        self.skeleton_dim = config.skeleton_dim
        self.skeleton_encoder = GraphSkeletonEncoder(self.skeleton_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=self.trans_dim, num_heads=8)
        self.skeleton_projection = nn.Linear(self.skeleton_dim, self.trans_dim)

    def forward(self, point_cloud, skeleton_points=None):
        # print(f"模型输入: point_cloud={point_cloud.shape}, skeleton_points={skeleton_points.shape if skeleton_points is not None else None}")
        q, coarse_point_cloud = self.base_model(point_cloud)  # B M C and B M 3
        B, M, C = q.shape

        if skeleton_points is not None:
            skeleton_points_flat = skeleton_points.view(B * 512, 3)  # [B*512, 3]
            edge_index = knn_graph(skeleton_points_flat, k=8, batch=torch.arange(B).repeat_interleave(512).to(skeleton_points.device))  # [2, E]
            skeleton_proxies = self.skeleton_encoder(skeleton_points_flat, edge_index)  # [B*512, skeleton_dim]
            skeleton_proxies = skeleton_proxies.view(B, 512, self.skeleton_dim)  # [B, 512, skeleton_dim]
            skeleton_proxies = torch.max(skeleton_proxies, dim=1)[0]  # [B, skeleton_dim]
            skeleton_proxies = self.skeleton_projection(skeleton_proxies)  # [B, trans_dim]
            skeleton_proxies = skeleton_proxies.unsqueeze(1).expand(-1, M, -1)  # [B, M, trans_dim]
            q = q.permute(1, 0, 2)  # [M, B, trans_dim]
            skeleton_proxies = skeleton_proxies.permute(1, 0, 2)  # [M, B, trans_dim]
            fused_features, _ = self.cross_attention(q, skeleton_proxies, skeleton_proxies)  # [M, B, trans_dim]
            q = fused_features.permute(1, 0, 2)  # [B, M, trans_dim]

        global_feature = self.increase_dim(q.transpose(1, 2)).transpose(1, 2)  # B M 1024
        global_feature = torch.max(global_feature, dim=1)[0]  # B 1024
        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse_point_cloud], dim=-1)  # B M 1027 + C
        rebuild_feature = self.reduce_map(rebuild_feature.reshape(B * M, -1))  # BM C
        relative_xyz = self.foldingnet(rebuild_feature).reshape(B, M, 3, -1)  # B M 3 S
        rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2, 3).reshape(B, -1, 3)  # B N 3
        inp_sparse = fps(point_cloud, self.num_query)
        coarse_point_cloud = torch.cat([coarse_point_cloud, inp_sparse], dim=1).contiguous()
        rebuild_points = torch.cat([rebuild_points, point_cloud], dim=1).contiguous()
        ret = (coarse_point_cloud, rebuild_points)
        return ret

    def get_loss(self, ret, gt, epoch, skeleton_points=None):
        coarse_points, dense_points = ret
        sparse_loss = self.loss_func(coarse_points, gt)
        dense_loss = self.loss_func(dense_points, gt)
        skeleton_loss = torch.tensor(0.0, device=dense_points.device)
        if skeleton_points is not None:
            skeleton_loss = torch.mean(torch.min(torch.cdist(dense_points, skeleton_points), dim=-1)[0])
        return sparse_loss, dense_loss + 0.1 * skeleton_loss