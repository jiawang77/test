import open3d as o3d
pcd = o3d.io.read_point_cloud("/home/lhz/PoinTr-master/data/PCN/test/skeleton/0000000/fc36d8e8521a433d98b3955602ef75dd_skeleton.ply")
print(f"Skeleton 点数: {len(pcd.points)}")  # 应接近512