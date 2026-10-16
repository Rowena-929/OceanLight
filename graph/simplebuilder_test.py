"""基于区域生长的简单自适应mesh构建器（支持HDBSCAN优化）

这个模块提供了一个基于区域生长的自适应mesh-grid构建器，它：
1. 对96维特征进行标准化
2. 使用PCA降维到1维
3. 使用Sobel算子计算每个grid的梯度
4. 使用科学方法自动确定区域生长的阈值
5. 从梯度最小的grid开始作为种子元素进行区域生长
6. 遍历上下左右四个邻居，如果特征差异在阈值内则合并到同一mesh区域
7. 使用HDBSCAN对过大的mesh进行进一步划分
8. 基于mesh区域的地理相邻性构建mesh之间的连接关系
9. 与现有的processor、decoder、denoiser兼容

注意：区域生长算法本身就能保证mesh区域的连通性，因此不需要额外的连通性检测。
"""

'''1/12度训练推理实验'''
"""先给高分辨率点找对应的mesh，没找到对应的mesh迭代继续找连通的已分配的，剩下4个孤立点手动分配最近的"""

import numpy as np
import torch
from torch_geometric.data import Data, HeteroData
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import scipy.ndimage as ndimage
from scipy.spatial.distance import cdist
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from scipy.spatial import Delaunay # 新增导入
import hdbscan # 新增导入
from typing import Union, Tuple # 新增导入
import pickle # 新增导入，用于保存和加载
from graph.model_utils import lat_lon_deg_to_spherical, spherical_to_cartesian

# 空间特征配置，与原始GraphBuilder保持一致
_spatial_features_kwargs = dict(
    add_node_positions=False,
    add_node_latitude=True,
    add_node_longitude=True,
    add_relative_positions=True,
    relative_longitude_local_coordinates=True,
    relative_latitude_local_coordinates=True,
)

# 半径查询因子，与原始GraphBuilder保持一致
radius_query_fraction_edge_length = 0.6
mesh2grid_edge_normalization_factor = None

mask = np.load('./data/land_sea_mask1.4.npy')[0, 4]
mask_flat = mask.flatten()
# fine_mask_flat = np.load('../data/land_sea_mask.npy')[0, 4, ::3, ::3].flatten()
fine_mask_flat = (np.load('./data/union_ocean_mask.npy')[::3,::3]*np.load('./data/total0.25mask.npy')[0][4]).flatten()
mercator = xr.open_dataset('./data/mercatorglorys12v1_gl12_mean_1993_2016_01.nc')

class SimpleAdaptiveMeshBuilder:
    """
    基于区域生长的简单自适应mesh-grid构建器（支持HDBSCAN优化）

    属性:
        g2m_graph (pyg.data.HeteroData): 连接grid节点到mesh节点的异构图
        mesh_graph (pyg.data.Data): 连接mesh节点的无向图
        m2g_graph (pyg.data.HeteroData): 连接mesh节点到grid节点的异构图
        khop_mesh_graph (pyg.data.Data): mesh图的k-hop邻居增强版本
        grid_nodes_dim (int): grid节点特征维度
        mesh_nodes_dim (int): mesh节点特征维度
        mesh_edges_dim (int): mesh边特征维度
        g2m_edges_dim (int): grid到mesh边特征维度
        m2g_edges_dim (int): mesh到grid边特征维度
    """

    def __new__(cls, path=None, *args, **kwargs):
        """如果提供path参数，则从文件加载而不是创建新实例"""
        if path is not None:
            return cls.load(path)
        return super().__new__(cls)

    def __init__(
            self,
            path=None,
            grid_lon: np.ndarray = None,
            grid_lat: np.ndarray = None,
            grid_features: np.ndarray = None,
            fine_grid_lon=None, # 新增参数
            fine_grid_lat=None,  # 新增参数
            fine_grid_features: np.ndarray = None,
            pca_dim: int = 1,
            threshold_method: str = "kmeans",
            gradient_threshold: float = 0.1,
            min_mesh_size: int = 5,
            num_hops: int = 0,
            device: torch.device = torch.device("cpu"),
            khop_device: torch.device = torch.device("cpu"),
            add_edge_features_to_khop: bool = True,
            # 移除不再需要的KNN相关参数
            # knn_k: int = 3,
            # radius_query_fraction: float = 0.6,
            # 新增HDBSCAN相关参数，与adaptive文件保持一致
            enable_hdbscan_refinement: bool = True,
            large_mesh_threshold: float = 0.2,
            hdbscan_min_cluster_size_factor: float = 0.05,
            hdbscan_min_samples_factor: float = 0.01,
            # 新增与adaptive文件一致的HDBSCAN参数
            cluster_selection_epsilon: float = 0.1,
            cluster_selection_method: str = 'leaf',
            alpha: float = 1.0,
    ):
        """初始化SimpleAdaptiveMeshBuilder对象。

        Args:
            path (str, optional): 如果提供，从此路径加载已保存的builder
            grid_lon (np.ndarray): 经度数组
            grid_lat (np.ndarray): 纬度数组
            grid_features (np.ndarray): grid特征数据，形状为(num_grid, 94)
            fine_grid_lon (np.ndarray): 新的1/12度经度数组 # 新增说明
            fine_grid_lat (np.ndarray): 新的1/12度纬度数组  # 新增说明
            pca_dim (int): PCA降维后的维度，默认1
            threshold_method (str): 阈值确定方法，可选：
                - "otsu": Otsu方法（基于直方图双峰）
                - "percentile": 百分位数方法
                - "kmeans": K-means聚类方法
                - "gmm": 高斯混合模型方法
                - "manual": 手动设定阈值
            gradient_threshold (float): 手动设定的梯度阈值，仅在threshold_method="manual"时使用
            min_mesh_size (int): 最小mesh区域大小，默认5
            num_hops (int): k-hop邻居的跳数，默认0
            device (torch.device): 图构建的设备，默认cpu
            khop_device (torch.device): k-hop计算的设备，默认cpu
            add_edge_features_to_khop (bool): 是否为k-hop图添加边特征，默认True
            # 移除knn_k和radius_query_fraction的说明
            enable_hdbscan_refinement (bool): 是否启用HDBSCAN优化，默认True
            large_mesh_threshold (float): 过大mesh的阈值比例，默认0.2（20%）
            hdbscan_min_cluster_size_factor (float): HDBSCAN最小聚类大小因子，默认0.05
            hdbscan_min_samples_factor (float): HDBSCAN最小样本数因子，默认0.01
            cluster_selection_epsilon (float): HDBSCAN聚类选择epsilon，默认0.1
            cluster_selection_method (str): HDBSCAN聚类选择方法，默认'leaf'
            alpha (float): HDBSCAN alpha参数，默认1.0
        """
        # 如果是从文件加载的，path已经被处理过了，不需要再次初始化
        if path is not None:
            return

        self._spatial_features_kwargs = _spatial_features_kwargs
        self.add_edge_features_to_khop = add_edge_features_to_khop
        self.device = device
        self.khop_device = khop_device
        # 移除不再需要的KNN相关属性
        # self.knn_k = knn_k
        # self.radius_query_fraction = radius_query_fraction
        self.num_hops = num_hops
        self.threshold_method = threshold_method
        self.gradient_threshold = gradient_threshold
        self.min_mesh_size = min_mesh_size
        self.pca_dim = pca_dim  # 保存pca_dim参数

        # 新增HDBSCAN相关属性
        self.enable_hdbscan_refinement = enable_hdbscan_refinement
        self.large_mesh_threshold = large_mesh_threshold
        self.hdbscan_min_cluster_size_factor = hdbscan_min_cluster_size_factor
        self.hdbscan_min_samples_factor = hdbscan_min_samples_factor

        # 新增与adaptive文件一致的HDBSCAN参数
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = cluster_selection_method
        self.alpha = alpha

        # 新增属性：用于收集附近没有海洋区域mesh的高分辨率网格点
        self._unassigned_fine_points = []

        # 初始化网格属性
        self._init_grid_properties(grid_lat, grid_lon)
        self._fine_grid_lon = fine_grid_lon # 保存高分辨率经纬度网格
        self._fine_grid_lat = fine_grid_lat # 保存高分辨率经纬度网格

        # 构建自适应mesh（区域生长算法本身保证连通性）
        self._build_adaptive_mesh(grid_features, pca_dim)

        # 初始化mesh属性
        # self._init_mesh_properties()

        self._grid_features_raw = fine_grid_features
        self._compute_fine_grid_to_mesh_connections()

        # 在所有高分辨率网格点都分配完毕后，更新mesh_labels并初始化mesh属性
        self._init_mesh_properties()

        # 构建图结构
        self._init_grid2mesh_graph()
        self._init_mesh_graph()
        self._init_mesh2grid_graph()
        self._init_khop_mesh_graph()

    def _determine_threshold_scientifically(self, feature_differences: np.ndarray) -> float:
        """使用科学方法自动确定阈值

        Args:
            feature_differences: 所有相邻grid点之间的特征差异

        Returns:
            float: 自动确定的阈值
        """
        print(f"使用 {self.threshold_method} 方法自动确定阈值...")

        if self.threshold_method == "manual":
            if self.gradient_threshold is None:
                raise ValueError("使用manual方法时必须提供gradient_threshold参数")
            print(f"使用手动设定阈值: {self.gradient_threshold}")
            return self.gradient_threshold

        elif self.threshold_method == "otsu":
            # Otsu方法：基于直方图双峰自动确定阈值
            threshold = self._otsu_threshold(feature_differences)
            print(f"Otsu方法确定阈值: {threshold:.4f}")
            return threshold

        elif self.threshold_method == "percentile":
            # 百分位数方法：使用75%分位数作为阈值
            threshold = np.percentile(feature_differences, 87)
            print(f"百分位数方法确定阈值: {threshold:.4f}")
            return threshold

        elif self.threshold_method == "kmeans":
            # K-means聚类方法：将特征差异分为两类
            threshold = self._kmeans_threshold(feature_differences)
            print(f"K-means方法确定阈值: {threshold:.4f}")
            return threshold

        elif self.threshold_method == "gmm":
            # 高斯混合模型方法
            threshold = self._gmm_threshold(feature_differences)
            print(f"GMM方法确定阈值: {threshold:.4f}")
            return threshold

        else:
            raise ValueError(f"不支持的阈值确定方法: {self.threshold_method}")

    def _otsu_threshold(self, feature_differences: np.ndarray) -> float:
        """Otsu方法确定阈值"""
        # 计算直方图
        hist, bin_edges = np.histogram(feature_differences, bins=60)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 计算累积分布
        total_pixels = np.sum(hist)
        cumulative_sum = np.cumsum(hist)
        cumulative_mean = np.cumsum(hist * bin_centers)

        # 计算全局均值
        global_mean = cumulative_mean[-1] / total_pixels

        # 计算类间方差
        max_variance = 0
        optimal_threshold = bin_centers[0]

        for i in range(len(hist)):
            if cumulative_sum[i] == 0 or cumulative_sum[i] == total_pixels:
                continue

            # 计算两个类的概率
            p1 = cumulative_sum[i] / total_pixels
            p2 = 1 - p1

            # 计算两个类的均值
            if cumulative_sum[i] > 0:
                mean1 = cumulative_mean[i] / cumulative_sum[i]
            else:
                mean1 = 0

            if (total_pixels - cumulative_sum[i]) > 0:
                mean2 = (cumulative_mean[-1] - cumulative_mean[i]) / (total_pixels - cumulative_sum[i])
            else:
                mean2 = 0

            # 计算类间方差
            variance = p1 * p2 * (mean1 - mean2) ** 2

            if variance > max_variance:
                max_variance = variance
                optimal_threshold = bin_centers[i]

        return optimal_threshold

    def _kmeans_threshold(self, feature_differences: np.ndarray) -> float:
        """K-means聚类方法确定阈值"""
        # 将特征差异重塑为2D数组
        X = feature_differences.reshape(-1, 1)

        # 使用K-means分为两类
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(X)

        # 获取两个聚类中心
        centers = kmeans.cluster_centers_.flatten()

        # 阈值是两个聚类中心的平均值
        threshold = np.mean(centers)

        return threshold

    def _gmm_threshold(self, feature_differences: np.ndarray) -> float:
        """高斯混合模型方法确定阈值"""
        # 将特征差异重塑为2D数组
        X = feature_differences.reshape(-1, 1)

        # 使用高斯混合模型分为两个组件
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)

        # 获取两个组件的均值
        means = gmm.means_.flatten()

        # 阈值是两个组件均值的加权平均
        weights = gmm.weights_
        threshold = np.sum(means * weights)

        return threshold

    def _init_grid_properties(self, grid_lat: np.ndarray, grid_lon: np.ndarray):
        """初始化网格属性"""
        self._grid_lat = grid_lat.astype(np.float32)
        self._grid_lon = grid_lon.astype(np.float32)
        self._num_lat = len(grid_lat)
        self._num_lon = len(grid_lon)
        self._num_grid_nodes = self._num_lat * self._num_lon

        # 创建网格坐标
        lon2d, lat2d = np.meshgrid(grid_lon, grid_lat)
        self._grid_nodes_lon = lon2d.flatten()
        self._grid_nodes_lat = lat2d.flatten()

    def _build_adaptive_mesh(self, grid_features: np.ndarray, pca_dim: int):
        """构建自适应mesh节点（区域生长算法保证连通性）"""
        print("开始构建基于区域生长的自适应mesh...")

        # 1. 数据预处理
        print(f"原始特征维度: {grid_features.shape}")
        assert grid_features.shape[
                   0] == self._num_grid_nodes, f"特征数量 {grid_features.shape[0]} 与网格节点数量 {self._num_grid_nodes} 不匹配"

        # 保存原始特征数据为类属性，供后续使用
        self._grid_features_raw = grid_features

        # 去掉标准化步骤，直接使用原始特征
        features_raw = grid_features
        print("跳过标准化步骤，直接使用原始特征")

        # 2. PCA降维到1维（在原始特征上）
        print(f"对94维原始特征进行PCA降维到{pca_dim}维...")
        pca = PCA(n_components=pca_dim, random_state=42)
        features_pca = pca.fit_transform(features_raw)
        print(f"PCA降维后特征维度: {features_pca.shape}")

        # 3. 计算梯度（在原始特征上）
        print("使用Sobel算子在原始特征上计算梯度...")
        # 将1维特征重塑为2D网格
        features_2d = features_pca.reshape(self._num_lat, self._num_lon)

        # 使用Sobel算子计算梯度
        grad_x = ndimage.sobel(features_2d, axis=1)  # 经度方向梯度
        grad_y = ndimage.sobel(features_2d, axis=0)  # 纬度方向梯度
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # 将梯度展平
        gradient_magnitude_flat = gradient_magnitude.flatten()

        # 4. 自动确定阈值（在原始特征上）
        print("自动确定区域生长阈值...")
        feature_differences = self._compute_feature_differences(features_raw)
        self.gradient_threshold = self._determine_threshold_scientifically(feature_differences)

        # 5. 区域生长算法（在原始特征上，保证连通性）
        print("开始区域生长算法...")
        mesh_labels = self._region_growing_algorithm(features_raw, gradient_magnitude_flat)

        # 6. 检查并进一步划分过大的mesh
        if self.enable_hdbscan_refinement:
            print("检查mesh大小分布...")
            mesh_labels = self._refine_large_meshes(features_raw, mesh_labels)
        else:
            print("跳过HDBSCAN优化...")

        # 7. 构建mesh节点
        self.mesh_labels = mesh_labels
        unique_labels = np.unique(mesh_labels)
        self.n_mesh = len(unique_labels)

        # mesh节点位置（聚类内grid点的经纬度均值）
        self.mesh_nodes = np.zeros((self.n_mesh, 2))
        for i, label in enumerate(unique_labels):
            cluster_indices = np.where(self.mesh_labels == label)[0]
            if len(cluster_indices) > 0:
                # 使用聚类内grid点的经纬度均值作为mesh节点位置
                self.mesh_nodes[i, 0] = self._grid_nodes_lat[cluster_indices].mean()  # 纬度
                self.mesh_nodes[i, 1] = self._grid_nodes_lon[cluster_indices].mean()  # 经度

        print(f"构建完成: {self.n_mesh} 个mesh节点（区域生长算法保证连通性）")

        # 保存PCA用于后续使用（不再需要scaler）
        # self.pca = pca
        self.gradient_magnitude = gradient_magnitude_flat

    def _compute_feature_differences(self, features_raw: np.ndarray) -> np.ndarray:
        """计算所有相邻grid点之间的特征差异，用于自动确定阈值"""
        print("计算相邻grid点间的特征差异...")
        feature_differences = []

        for i in range(self._num_grid_nodes):
            neighbors = self._get_neighbors(i)
            for neighbor_idx in neighbors:
                if neighbor_idx is not None:
                    # 检查当前点和邻居点是否都是陆地区域
                    current_is_land = mask_flat[i] == 0
                    neighbor_is_land = mask_flat[neighbor_idx] == 0

                    # 如果两个点都是陆地区域，则跳过不计算差异
                    if current_is_land and neighbor_is_land:
                        continue

                    feature_diff = np.linalg.norm(
                        features_raw[i] - features_raw[neighbor_idx]
                    )
                    feature_differences.append(feature_diff)

        return np.array(feature_differences)

    def _region_growing_algorithm(self, features_raw: np.ndarray, gradient_magnitude: np.ndarray):
        """区域生长算法（保证连通性，跳过陆地部分，与种子点比较特征差异，使用8连通邻居）"""
        num_grid = features_raw.shape[0]
        mesh_labels = np.full(num_grid, -1)  # -1表示未分配
        current_mesh_id = 0

        # 创建访问标记
        visited = np.zeros(num_grid, dtype=bool)

        # 将陆地部分标记为已访问，避免处理
        land_indices = np.where(mask_flat == 0)[0]
        visited[land_indices] = True
        mesh_labels[land_indices] = -2  # -2表示陆地，不参与mesh构建

        # 按梯度大小排序（从小到大），但只考虑海洋部分
        ocean_indices = np.where(mask_flat == 1)[0]
        sorted_indices = ocean_indices[np.argsort(gradient_magnitude[ocean_indices])]

        for start_idx in sorted_indices:
            if visited[start_idx]:
                continue

            # 开始新的区域生长，记录种子点特征
            seed_features = features_raw[start_idx]
            queue = [start_idx]
            mesh_labels[start_idx] = current_mesh_id
            visited[start_idx] = True

            while queue:
                current_idx = queue.pop(0)

                # 获取当前grid的8连通邻居（上下左右+四个对角方向）
                neighbors = self._get_neighbors(current_idx)

                for neighbor_idx in neighbors:
                    if neighbor_idx is None or visited[neighbor_idx]:
                        continue

                    # 计算与种子点的特征差异（而不是与相邻点比较）
                    feature_diff = np.linalg.norm(
                        seed_features - features_raw[neighbor_idx]
                    )

                    # 如果特征差异在阈值内，加入当前mesh区域
                    if feature_diff <= self.gradient_threshold:
                        mesh_labels[neighbor_idx] = current_mesh_id
                        visited[neighbor_idx] = True
                        queue.append(neighbor_idx)
            current_mesh_id += 1

            # 检查mesh区域大小，如果太小则重新分配
            # mesh_size = np.sum(mesh_labels == current_mesh_id)
            # if mesh_size < self.min_mesh_size:
            #     # 将小区域重新标记为未分配
            #     mesh_labels[mesh_labels == current_mesh_id] = -1
            #     visited[mesh_labels == current_mesh_id] = False
            # else:
            #     current_mesh_id += 1

        # 处理未分配的海洋grid点（分配给最近的mesh）
        unassigned_ocean_indices = np.where((mesh_labels == -1) & (mask_flat == 1))[0]
        if len(unassigned_ocean_indices) > 0:
            print(f"处理 {len(unassigned_ocean_indices)} 个未分配的海洋grid点...")
            for idx in unassigned_ocean_indices:
                nearest_mesh = self._find_nearest_mesh(idx, mesh_labels, features_raw)
                mesh_labels[idx] = nearest_mesh

        return mesh_labels

    def _refine_large_meshes(self, features_raw: np.ndarray, mesh_labels: np.ndarray) -> np.ndarray:
        """使用HDBSCAN对过大的mesh进行进一步划分"""
        print("检查并进一步划分过大的mesh...")

        # 统计每个mesh的大小
        unique_labels = np.unique(mesh_labels[mesh_labels >= 0])
        mesh_sizes = {}
        for label in unique_labels:
            mesh_sizes[label] = np.sum(mesh_labels == label)

        # 计算总海洋grid数量
        total_ocean_grids = sum(mesh_sizes.values())

        # 找出过大的mesh（占据超过阈值的grid节点）
        large_meshes = []

        for label, size in mesh_sizes.items():
            percentage = size / total_ocean_grids
            if percentage > self.large_mesh_threshold:
                large_meshes.append((label, size, percentage))
                print(f"发现过大mesh {label}: {size} 个grid节点 ({percentage:.2%})")

        if not large_meshes:
            print(f"没有发现过大的mesh（阈值: {self.large_mesh_threshold:.0%}），无需进一步划分")
            return mesh_labels

        # 对每个过大的mesh使用HDBSCAN进行进一步划分
        refined_labels = mesh_labels.copy()
        next_mesh_id = max(unique_labels) + 1

        for original_label, size, percentage in large_meshes:
            print(f"使用HDBSCAN进一步划分mesh {original_label}...")

            # 获取该mesh的所有grid点
            mesh_indices = np.where(mesh_labels == original_label)[0]
            mesh_features = features_raw[mesh_indices]

            # 使用HDBSCAN进行聚类
            try:
                from hdbscan import HDBSCAN

                # 配置HDBSCAN参数，与adaptive文件保持一致
                min_cluster_size = max(5, int(size * self.hdbscan_min_cluster_size_factor))
                min_samples = max(3, int(size * self.hdbscan_min_samples_factor))

                hdbscan_params = {
                    'min_cluster_size': min_cluster_size,
                    'min_samples': min_samples,
                    'cluster_selection_epsilon': 0.1,
                    'metric': 'euclidean',
                    'cluster_selection_method': 'leaf',  # 使用leaf方法提高连续性
                    'alpha': 1.0,  # 增加alpha值提高空间连续性
                }

                print(f"HDBSCAN参数: {hdbscan_params}")

                # 执行HDBSCAN聚类
                clusterer = HDBSCAN(**hdbscan_params)
                hdbscan_labels = clusterer.fit_predict(mesh_features)

                # 统计聚类结果
                unique_hdbscan_labels = np.unique(hdbscan_labels)
                num_clusters = len(unique_hdbscan_labels[unique_hdbscan_labels >= 0])
                noise_points = np.sum(hdbscan_labels == -1)

                print(f"HDBSCAN聚类结果: {num_clusters} 个聚类, {noise_points} 个噪声点")

                # 处理噪声点（分配给最近的聚类）
                if noise_points > 0:
                    print("处理噪声点...")
                    noise_indices = np.where(hdbscan_labels == -1)[0]
                    valid_cluster_indices = np.where(hdbscan_labels != -1)[0]

                    if len(valid_cluster_indices) > 0:
                        # 使用欧几里得距离计算最近的聚类
                        noise_features = mesh_features[noise_indices]
                        valid_features = mesh_features[valid_cluster_indices]

                        distances = cdist(noise_features, valid_features, metric='euclidean')
                        nearest_indices = np.argmin(distances, axis=1)
                        nearest_cluster = hdbscan_labels[valid_cluster_indices[nearest_indices]]

                        hdbscan_labels[noise_indices] = nearest_cluster
                        print(f"噪声点已重新分配给最近的聚类")

                # 重新分配标签
                for i, hdbscan_label in enumerate(hdbscan_labels):
                    grid_idx = mesh_indices[i]

                    if hdbscan_label == 0:
                        # 第一个聚类保持原标签
                        continue
                    else:
                        # 其他聚类分配新标签
                        new_label = next_mesh_id + hdbscan_label - 1
                        refined_labels[grid_idx] = new_label

                # 更新下一个可用的mesh ID
                next_mesh_id += num_clusters - 1

                print(f"Mesh {original_label} 已划分为 {num_clusters} 个子区域")

            except ImportError:
                print("警告: 未安装hdbscan，跳过HDBSCAN划分")
                print("请安装: pip install hdbscan")
                continue
            except Exception as e:
                print(f"HDBSCAN聚类失败: {e}")
                continue

        # 强制地理连通性修正：将每个聚类的每个连通分量单独编号为mesh节点
        print("强制地理连通性修正...")
        refined_labels = self._enforce_geographic_continuity(refined_labels)

        # 重新整理标签，确保连续
        final_labels = self._relabel_meshes(refined_labels)

        return final_labels

    def _enforce_geographic_continuity(self, mesh_labels: np.ndarray) -> np.ndarray:
        """强制地理连通性修正：将每个聚类的每个连通分量单独编号为mesh节点

        Args:
            mesh_labels: 当前的mesh标签数组

        Returns:
            np.ndarray: 修正后的mesh标签数组
        """
        print("强制地理连通性修正...")

        # 创建网格形状的标签矩阵
        grid_shape = (self._num_lat, self._num_lon)
        labels_matrix = mesh_labels.reshape(grid_shape)

        # 创建新的标签数组
        new_labels = np.full_like(mesh_labels, -1)
        new_mesh_id = 0

        # 获取所有唯一的mesh标签（排除陆地和未分配）
        unique_labels = np.unique(mesh_labels[mesh_labels >= 0])

        # 遍历每个聚类
        for cluster_id in unique_labels:
            # 创建该聚类的二值掩码
            mask = (labels_matrix == cluster_id)

            # 计算连通分量（使用8连通）
            labeled_array, num_features = ndimage.label(mask, structure=np.ones((3, 3)))

            # print(f"  聚类 {cluster_id} 包含 {num_features} 个连通分量")

            # 为每个连通分量分配新的mesh ID
            for comp_id in range(1, num_features + 1):
                comp_mask = (labeled_array == comp_id)
                new_labels[comp_mask.flatten()] = new_mesh_id
                new_mesh_id += 1

        print(f"连通性修正完成: {new_mesh_id} 个连通mesh")

        return new_labels

    def _relabel_meshes(self, mesh_labels: np.ndarray) -> np.ndarray:
        """重新整理mesh标签，确保标签连续"""
        print("重新整理mesh标签...")

        # 获取所有有效的mesh标签（排除陆地和未分配）
        valid_labels = mesh_labels[mesh_labels >= 0]
        unique_labels = np.unique(valid_labels)

        # 创建标签映射
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}

        # 应用映射
        relabeled = mesh_labels.copy()
        for old_label, new_label in label_mapping.items():
            relabeled[mesh_labels == old_label] = new_label

        print(f"标签重新整理完成: {len(unique_labels)} 个mesh")

        return relabeled

    def _get_neighbors(self, idx: int):
        """获取grid点的8连通邻居（上下左右+四个对角方向，只考虑海洋部分）"""
        row = idx // self._num_lon
        col = idx % self._num_lon

        neighbors = []

        # 上
        if row > 0:
            up_idx = idx - self._num_lon
            if mask_flat[up_idx] == 1:  # 只考虑海洋部分
                neighbors.append(up_idx)
            else:
                neighbors.append(None)
        else:
            neighbors.append(None)

        # 下
        if row < self._num_lat - 1:
            down_idx = idx + self._num_lon
            if mask_flat[down_idx] == 1:  # 只考虑海洋部分
                neighbors.append(down_idx)
            else:
                neighbors.append(None)
        else:
            neighbors.append(None)

        # 左
        if col > 0:
            left_idx = idx - 1
            if mask_flat[left_idx] == 1:  # 只考虑海洋部分
                neighbors.append(left_idx)
            else:
                neighbors.append(None)
        else:
            neighbors.append(None)

        # 右
        if col < self._num_lon - 1:
            right_idx = idx + 1
            if mask_flat[right_idx] == 1:  # 只考虑海洋部分
                neighbors.append(right_idx)
            else:
                neighbors.append(None)
        else:
            neighbors.append(None)

        # 左上对角
        if row > 0 and col > 0:
            upleft_idx = idx - self._num_lon - 1
            if mask_flat[upleft_idx] == 1:  # 只考虑海洋部分
                neighbors.append(upleft_idx)
            else:
                neighbors.append(None)
        else:
            neighbors.append(None)

        # 右上对角
        if row > 0 and col < self._num_lon - 1:
            upright_idx = idx - self._num_lon + 1
            if mask_flat[upright_idx] == 1:  # 只考虑海洋部分
                neighbors.append(upright_idx)
            else:
                neighbors.append(None)
        else:
            neighbors.append(None)

        # 左下对角
        if row < self._num_lat - 1 and col > 0:
            downleft_idx = idx + self._num_lon - 1
            if mask_flat[downleft_idx] == 1:  # 只考虑海洋部分
                neighbors.append(downleft_idx)
            else:
                neighbors.append(None)
        else:
            neighbors.append(None)

        # 右下对角
        if row < self._num_lat - 1 and col < self._num_lon - 1:
            downright_idx = idx + self._num_lon + 1
            if mask_flat[downright_idx] == 1:  # 只考虑海洋部分
                neighbors.append(downright_idx)
            else:
                neighbors.append(None)
        else:
            neighbors.append(None)

        return neighbors

    def _find_nearest_mesh(self, idx: int, mesh_labels: np.ndarray, features_raw: np.ndarray):
        """找到最近的mesh区域"""
        current_features = features_raw[idx]

        # 找到所有已分配的mesh
        assigned_meshes = np.unique(mesh_labels[mesh_labels >= 0])

        min_distance = float('inf')
        nearest_mesh = assigned_meshes[0] if len(assigned_meshes) > 0 else 0

        for mesh_id in assigned_meshes:
            mesh_indices = np.where(mesh_labels == mesh_id)[0]
            mesh_center_features = features_raw[mesh_indices].mean(axis=0)

            distance = np.linalg.norm(current_features - mesh_center_features)
            if distance < min_distance:
                min_distance = distance
                nearest_mesh = mesh_id

        return nearest_mesh

    def _find_nearest_assigned_coarse_grid_idx(self, fine_lat: float, fine_lon: float) -> Union[int, None]:
        """
        在所有已分配mesh的原始grid点中，找到距离给定高分辨率点最近的grid点索引。
        """
        # 找到所有已分配mesh的原始grid点索引
        assigned_coarse_grid_indices = np.where(self.mesh_labels >= 0)[0]

        if len(assigned_coarse_grid_indices) == 0:
            return None # 没有已分配的mesh区域

        # 获取这些已分配grid点的经纬度坐标
        assigned_coarse_grid_lats = self._grid_nodes_lat[assigned_coarse_grid_indices]
        assigned_coarse_grid_lons = self._grid_nodes_lon[assigned_coarse_grid_indices]

        # 组合成 (N, 2) 形状的坐标数组
        assigned_coarse_coords = np.column_stack([assigned_coarse_grid_lats, assigned_coarse_grid_lons])
        fine_point_coords = np.array([[fine_lat, fine_lon]])

        # 计算距离
        distances = cdist(fine_point_coords, assigned_coarse_coords, metric='euclidean')

        # 找到距离最近的已分配原始grid点的索引
        nearest_idx_in_assigned_list = np.argmin(distances)
        nearest_original_grid_idx = assigned_coarse_grid_indices[nearest_idx_in_assigned_list]

        return nearest_original_grid_idx

    def continue_fine(self, fine_grid_idx: int, fine_lat: float, fine_lon: float):
        # 获取当前高分辨率网格点的八个邻居
        neighbors = self._get_fine_grid_neighbors(fine_grid_idx)

        # 检查邻居中是否有已分配mesh的网格点
        for neighbor_fine_grid_idx in neighbors:
            if neighbor_fine_grid_idx is not None and self.fine_grid_mesh_labels[neighbor_fine_grid_idx] >= 0:
                # 如果邻居已分配mesh，则将当前点归为该mesh
                mesh_id = self.fine_grid_mesh_labels[neighbor_fine_grid_idx]
                self.fine_grid_mesh_labels[fine_grid_idx] = mesh_id
                return mesh_id
        # 如果所有邻居都没有分配mesh，返回-1表示未找到
        return -1

    def _find_mesh_for_fine_grid_point(self, fine_grid_idx: int, fine_lat: float, fine_lon: float) -> Union[int, None]:
        """
        找到高分辨率网格点所属的mesh区域ID。
        通过查找距离高分辨率点最近的原始grid点（局部搜索），并返回该原始grid点所属的mesh ID。
        """
        # 1. 找到高分辨率点在原始1.4度网格中的最近索引
        lat_insert_idx = np.searchsorted(self._grid_lat, fine_lat)
        lat_idx_low = np.clip(lat_insert_idx - 1, 0, self._num_lat - 2)
        lat_idx_high = lat_idx_low + 1
        # 找到fine_lon在_grid_lon中的插入点，并计算低/高经度索引
        lon_insert_idx = np.searchsorted(self._grid_lon, fine_lon)
        lon_idx_low = np.clip(lon_insert_idx - 1, 0, self._num_lon - 2)
        lon_idx_high = lon_idx_low + 1

        # 2. 获取该网格点及其周围的邻居（中心点 + 上下左右）
        candidate_indices = []
        candidate_indices.append(lat_idx_low * self._num_lon + lon_idx_low)
        candidate_indices.append(lat_idx_low * self._num_lon + lon_idx_high)
        candidate_indices.append(lat_idx_high * self._num_lon + lon_idx_low)
        candidate_indices.append(lat_idx_high * self._num_lon + lon_idx_high)

        # 过滤掉陆地点的索引，只保留海洋点
        valid_candidate_indices = [idx for idx in candidate_indices if self.mesh_labels[idx] >= 0]

        if not valid_candidate_indices:
           # 如果全局也没有已分配的mesh，则加入未分配列表
            self._unassigned_fine_points.append((fine_grid_idx, fine_lat, fine_lon))
            return -1  # 附近没有海洋区域的mesh

        # 3. 计算高分辨率点到这些候选原始grid点的距离
        fine_point_coords = np.array([[fine_lat, fine_lon]])
        candidate_grid_coords = np.column_stack([
            self._grid_nodes_lat[valid_candidate_indices],
            self._grid_nodes_lon[valid_candidate_indices]
        ])
        distances = cdist(fine_point_coords, candidate_grid_coords, metric='euclidean')

        # 4. 找到距离最近的原始grid点的索引
        nearest_candidate_idx_in_valid_list = np.argmin(distances)
        nearest_original_grid_idx = valid_candidate_indices[nearest_candidate_idx_in_valid_list]

        # 5. 返回该最近原始网格点所属的mesh ID
        mesh_id = self.mesh_labels[nearest_original_grid_idx]
        return mesh_id

    def _init_mesh_properties(self):
        """初始化mesh属性"""
        # mesh节点的经纬度
        self._mesh_nodes_lat = self.mesh_nodes[:, 0]
        self._mesh_nodes_lon = self.mesh_nodes[:, 1]

        # 计算mesh节点之间的距离矩阵
        mesh_coords = np.column_stack([self._mesh_nodes_lat, self._mesh_nodes_lon])
        self._mesh_distances = cdist(mesh_coords, mesh_coords, metric='euclidean')

        # 计算mesh节点的空间特征（与graph_builder保持一致）
        # 特征维度: [纬度余弦, 经度余弦, 经度正弦]
        self._mesh_features = np.zeros((self.n_mesh, 3))
        # 纬度余弦: cos(latitude)，纬度范围 [-90, 90]
        self._mesh_features[:, 0] = np.cos(np.deg2rad(self._mesh_nodes_lat))
        # 经度余弦和经度正弦: 经度范围 [0, 360]
        self._mesh_features[:, 1] = np.cos(np.deg2rad(self._mesh_nodes_lon))
        self._mesh_features[:, 2] = np.sin(np.deg2rad(self._mesh_nodes_lon))

    def _compute_fine_grid_to_mesh_connections(self):
        """
        计算高分辨率网格点到mesh区域的对应关系，即填充fine_grid_mesh_labels。
        Args:
            fine_grid_nodes_lon (np.ndarray): 高分辨率网格点的经度数组。
            fine_grid_nodes_lat (np.ndarray): 高分辨率网格点的纬度数组。
            fine_mask_flat (np.ndarray): 高分辨率网格的平坦化掩码。
        Returns:
            np.ndarray: fine_grid_mesh_labels数组，表示每个高分辨率网格点所属的mesh ID。
        """
        fine_grid_lon_2d, fine_grid_lat_2d = np.meshgrid(self._fine_grid_lon, self._fine_grid_lat)
        self.fine_grid_nodes_lon = fine_grid_lon_2d.flatten()
        self.fine_grid_nodes_lat = fine_grid_lat_2d.flatten()
        num_fine_grid_nodes = len(self.fine_grid_nodes_lon)
        self.fine_grid_mesh_labels = np.full(num_fine_grid_nodes, -1, dtype=int)  # 初始化mesh_labels为-1

        # 遍历每个高分辨率网格点
        for fine_grid_idx in range(num_fine_grid_nodes):
            fine_lon = self.fine_grid_nodes_lon[fine_grid_idx]
            fine_lat = self.fine_grid_nodes_lat[fine_grid_idx]

            if fine_mask_flat[fine_grid_idx] == 0:
                self.fine_grid_mesh_labels[fine_grid_idx] = -2 # 陆地标记为-2
                continue
            # 找到该高分辨率网格点所属的mesh区域
            mesh_id = self._find_mesh_for_fine_grid_point(fine_grid_idx, fine_lat, fine_lon)
            self.fine_grid_mesh_labels[fine_grid_idx] = mesh_id  # 存储mesh_id
        self.mesh_labels = self.fine_grid_mesh_labels
        print('继续划分')
        #继续给未分配点找有没有连通的
        while len(self._unassigned_fine_points) > 0:
            print(len(self._unassigned_fine_points))
            _unassigned_fine_points = []
            for fine_grid_idx,fine_lat,fine_lon in self._unassigned_fine_points:
                if fine_mask_flat[fine_grid_idx] == 0:
                    self.fine_grid_mesh_labels[fine_grid_idx] = -2  # 陆地标记为-2
                    continue
                # 找到该高分辨率网格点所属的mesh区域
                mesh_id = self.continue_fine(fine_grid_idx, fine_lat, fine_lon)
                self.fine_grid_mesh_labels[fine_grid_idx] = mesh_id  # 存储mesh_id
                if mesh_id < 0:
                    _unassigned_fine_points.append((fine_grid_idx, fine_lat, fine_lon))
            self._unassigned_fine_points = _unassigned_fine_points
            #继续分配这些奇怪的点
            if len(self._unassigned_fine_points) == 42:
                for fine_grid_idx, fine_lat, fine_lon in self._unassigned_fine_points:
                    nearest_mesh_id = self._calculate_nearest_assigned_mesh(fine_grid_idx, fine_lat, fine_lon)
                    self.fine_grid_mesh_labels[fine_grid_idx] = nearest_mesh_id
                break


        # self._group_unassigned_fine_points_by_connectivity()
        self.mesh_labels = self.fine_grid_mesh_labels
        self._init_grid_properties(self._fine_grid_lat, self._fine_grid_lon)
        unique_labels = np.unique(self.mesh_labels[self.mesh_labels >= 0])
        # unique_labels = np.unique(self.mesh_labels)
        self.n_mesh = len(unique_labels)
        self.mesh_nodes = np.zeros((self.n_mesh, 2))
        for i, label in enumerate(unique_labels):
            cluster_indices = np.where(self.mesh_labels == label)[0]
            if len(cluster_indices) > 0:
                # 使用聚类内grid点的经纬度均值作为mesh节点位置
                self.mesh_nodes[i, 0] = self._grid_nodes_lat[cluster_indices].mean()  # 纬度
                self.mesh_nodes[i, 1] = self._grid_nodes_lon[cluster_indices].mean()  # 经度

        print(f"高分辨率mesh构建完成: {self.n_mesh} 个mesh节点")

    def _calculate_nearest_assigned_mesh(self, fine_grid_idx, fine_lat, fine_lon):
        """
        为未分配的高分辨率网格点找到最近的已分配mesh。
        """
        assigned_indices = np.where(self.fine_grid_mesh_labels >= 0)[0]

        assigned_lons = self.fine_grid_nodes_lon[assigned_indices]
        assigned_lats = self.fine_grid_nodes_lat[assigned_indices]

        # Calculate distances (using Haversine or simple Euclidean for approximation)
        # For simplicity, using Euclidean distance on lat/lon, but Haversine is more accurate for geographical points
        distances = np.sqrt((assigned_lons - fine_lon) ** 2 + (assigned_lats - fine_lat) ** 2)
        nearest_assigned_idx_in_subset = np.argmin(distances)
        nearest_assigned_grid_idx = assigned_indices[nearest_assigned_idx_in_subset]

        return self.fine_grid_mesh_labels[nearest_assigned_grid_idx]


    def _group_unassigned_fine_points_by_connectivity(self):
        """
        将所有未分配的高分辨率网格点（在self._unassigned_fine_points中）根据连通性进行分组，
        并为每个连通分量分配新的mesh ID。
        """
        print(f"处理 {len(self._unassigned_fine_points)} 个未分配的高分辨率网格点...")

        if not self._unassigned_fine_points:
            print("没有未分配的高分辨率网格点，跳过连通性分组。")
            return

        # 将未分配点转换为一个集合，方便快速查找和删除已处理点
        unassigned_set = set(item[0] for item in self._unassigned_fine_points)  # 存储fine_grid_idx
        current_max_mesh_id = np.max(self.fine_grid_mesh_labels[self.fine_grid_mesh_labels >= 0]) if len(self.fine_grid_mesh_labels[self.fine_grid_mesh_labels >= 0]) > 0 else -1
        next_new_mesh_id = current_max_mesh_id + 1

        while unassigned_set:
            # 选取一个未处理的种子点
            seed_fine_grid_idx = unassigned_set.pop()
            current_cluster = []
            queue = [seed_fine_grid_idx]
            
            # 分配新的mesh ID给该种子点
            self.fine_grid_mesh_labels[seed_fine_grid_idx] = next_new_mesh_id
            current_cluster.append(seed_fine_grid_idx)

            while queue:
                current_fine_grid_idx = queue.pop(0)

                # 获取当前高分辨率点的8连通邻居
                fine_neighbors = self._get_fine_grid_neighbors(current_fine_grid_idx)

                for neighbor_fine_grid_idx in fine_neighbors:
                    if neighbor_fine_grid_idx is None or neighbor_fine_grid_idx not in unassigned_set:
                        continue

                    # 如果邻居点也是未分配的，则将其加入当前mesh区域
                    self.fine_grid_mesh_labels[neighbor_fine_grid_idx] = next_new_mesh_id
                    unassigned_set.remove(neighbor_fine_grid_idx)
                    queue.append(neighbor_fine_grid_idx)
                    current_cluster.append(neighbor_fine_grid_idx)
            
            if current_cluster:
                print(f"创建新mesh {next_new_mesh_id}，包含 {len(current_cluster)} 个高分辨率网格点")
            next_new_mesh_id += 1
        
        # 清空未分配列表
        self._unassigned_fine_points = []
        print("未分配的高分辨率网格点处理完成。")

    def _init_grid2mesh_graph(self):
        """初始化grid到mesh的图 - 基于高分辨率网格建立连接"""
        print("构建基于高分辨率网格的grid到mesh连接...")

        # 获取高分辨率网格的经纬度数据
        fine_grid_lon_2d, fine_grid_lat_2d = np.meshgrid(self._fine_grid_lon, self._fine_grid_lat)
        fine_grid_nodes_lon = fine_grid_lon_2d.flatten()
        fine_grid_nodes_lat = fine_grid_lat_2d.flatten()
        # num_fine_grid_nodes = len(fine_grid_nodes_lon) # 这一行不再需要，因为函数内部会处理

        # 调用新的函数来计算fine_grid_mesh_labels

        senders = []  # fine_grid节点索引
        receivers = []  # mesh节点索引
        edge_features = []  # 边特征

        # 根据fine_grid_mesh_labels重新构建senders, receivers和edge_features
        for fine_grid_idx in range(len(fine_grid_nodes_lon)):
            mesh_id = self.fine_grid_mesh_labels[fine_grid_idx]
            if mesh_id >= 0:  # 排除陆地 (-1) 和未分配 (-2)
                senders.append(fine_grid_idx)
                receivers.append(mesh_id)

                # 计算边特征：使用fine_grid点到mesh中心的距离作为权重
                fine_lon = fine_grid_nodes_lon[fine_grid_idx]
                fine_lat = fine_grid_nodes_lat[fine_grid_idx]
                mesh_lat = self._mesh_nodes_lat[mesh_id]
                mesh_lon = self._mesh_nodes_lon[mesh_id]

                # 计算欧几里得距离
                distance = np.sqrt((fine_lat - mesh_lat) ** 2 + (fine_lon - mesh_lon) ** 2)

                # 使用距离的倒数作为权重（距离越近权重越大）
                edge_weight = 1.0 / (distance + 1e-8)
                # 先转换为球坐标
                fine_phi, fine_theta = lat_lon_deg_to_spherical(fine_lat, fine_lon)
                mesh_phi, mesh_theta = lat_lon_deg_to_spherical(mesh_lat, mesh_lon)

                # 再转换为笛卡尔坐标
                fine_x, fine_y, fine_z = spherical_to_cartesian(fine_phi, fine_theta)
                mesh_x, mesh_y, mesh_z = spherical_to_cartesian(mesh_phi, mesh_theta)

                # 计算相对位置
                rel_x = fine_x - mesh_x
                rel_y = fine_y - mesh_y
                rel_z = fine_z - mesh_z
                edge_features.append([edge_weight, rel_x, rel_y, rel_z])  # 添加边类型标记：0表示所属mesh连接

        # 检查是否有任何连接被创建
        if not senders:
            print("警告: 没有高分辨率网格点连接到任何mesh区域。Grid到Mesh图将为空。")
            # 设置空的图结构，避免后续错误
            self.g2m_graph = HeteroData()
            self.g2m_graph["grid_nodes"].x = torch.empty(0, 3, dtype=torch.float32, device=self.device)  # 空节点特征
            self.g2m_graph["mesh_nodes"].x = torch.empty(0, 3, dtype=torch.float32, device=self.device)  # 空节点特征
            self.g2m_graph["grid_nodes", "to", "mesh_nodes"].edge_index = torch.empty(2, 0, dtype=torch.long,
                                                                                      device=self.device)  # 空边索引
            self.g2m_graph["grid_nodes", "to", "mesh_nodes"].edge_attr = torch.empty(0, 4, dtype=torch.float32,
                                                                                     device=self.device)  # 空边特征
            self._grid_nodes_dim = 3  # 纬度余弦、经度余弦、经度正弦
            self._mesh_nodes_dim = 3  # 纬度余弦、经度余弦、经度正弦
            self._g2m_edges_dim = 4  # 边权重 + 相对位置向量
            return

        # 转换为PyTorch张量
        senders = torch.tensor(senders, dtype=torch.long, device=self.device)
        receivers = torch.tensor(receivers, dtype=torch.long, device=self.device)
        edge_features = torch.tensor(edge_features, dtype=torch.float32, device=self.device)
        self.mesh_labels = self.fine_grid_mesh_labels

        # 创建异构图
        self.g2m_graph = HeteroData()
        self.g2m_graph["grid_nodes"].x = torch.tensor(
            np.column_stack([
                np.cos(np.radians(fine_grid_nodes_lat)), # 使用高分辨率网格的纬度
                np.cos(np.radians(fine_grid_nodes_lon)), # 使用高分辨率网格的经度
                np.sin(np.radians(fine_grid_nodes_lon))  # 使用高分辨率网格的经度
            ]), dtype=torch.float32, device=self.device
        )
        self.g2m_graph["mesh_nodes"].x = torch.tensor(
            np.column_stack([
                np.cos(np.radians(self._mesh_nodes_lat)),
                np.cos(np.radians(self._mesh_nodes_lon)),
                np.sin(np.radians(self._mesh_nodes_lon))
            ]), dtype=torch.float32, device=self.device
        )
        self.g2m_graph["grid_nodes", "to", "mesh_nodes"].edge_index = torch.stack([senders, receivers])
        self.g2m_graph["grid_nodes", "to", "mesh_nodes"].edge_attr = edge_features

        # 设置节点特征维度（作为实例变量而不是属性）
        self._grid_nodes_dim = 3  # 纬度余弦、经度余弦、经度正弦
        self._mesh_nodes_dim = 3  # 纬度余弦、经度余弦、经度正弦
        self._g2m_edges_dim = 4  # 边权重 + 边类型

        # 统计边类型
        belonging_edges = sum(1 for edge_feat in edge_features if len(edge_feat) == 2 and edge_feat[1] == 0.0)
        # nearest_edges = sum(1 for edge_feat in edge_features if len(edge_feat) == 2 and edge_feat[1] == 1.0) # 移除此行，因为现在只有所属连接

        print(f"Grid到Mesh连接完成:")
        print(f"  - 基于区域生长的所属mesh连接: {belonging_edges} 条边")
        # print(f"  - 基于距离的最近mesh连接: {nearest_edges} 条边") # 移除此行
        print(f"  - 总连接数: {len(senders)} 条边")
        print(f"  - 连接的grid点数量: {len(torch.unique(senders))}")
        print(f"  - 连接的mesh节点数量: {len(torch.unique(receivers))}")
        print(f"  - 边特征维度: {self._g2m_edges_dim} (权重 + 边类型)")

    def _init_mesh_graph(self):
        """初始化mesh图 - 基于地理相邻性构建连接"""
        print("构建基于地理相邻性的mesh连接...")

        # 基于地理相邻性构建mesh连接
        # 检查每个mesh区域是否与其相邻，如果相邻则建立连接

        # 将mesh_labels重塑为2D网格
        mesh_labels_2d = self.mesh_labels.reshape(self._num_lat, self._num_lon)

        # 存储相邻关系的集合，避免重复
        adjacent_pairs = set()

        # 遍历所有网格点，检查每个点的邻居是否属于不同mesh
        for i in range(self._num_grid_nodes):
            if self.mesh_labels[i] < 0:  # 跳过陆地和未分配点
                continue

            current_mesh = self.mesh_labels[i]
            neighbors = self._get_fine_grid_neighbors(i)

            for neighbor_idx in neighbors:
                if neighbor_idx is None or self.mesh_labels[neighbor_idx] < 0:
                    continue

                neighbor_mesh = self.mesh_labels[neighbor_idx]
                if current_mesh != neighbor_mesh:
                    # 两个不同的mesh相邻，建立连接
                    edge = tuple([current_mesh, neighbor_mesh])
                    adjacent_pairs.add(edge)

        # 将相邻对转换为边列表
        edges = list(adjacent_pairs)
        senders = [edge[0] for edge in edges]
        receivers = [edge[1] for edge in edges]

        # 计算边特征 (使用与graph_builder.py兼容的格式)
        edge_features = []
        for i in range(len(senders)):
            sender_lat = self._mesh_nodes_lat[senders[i]]
            sender_lon = self._mesh_nodes_lon[senders[i]]
            receiver_lat = self._mesh_nodes_lat[receivers[i]]
            receiver_lon = self._mesh_nodes_lon[receivers[i]]

            # 将经纬度转换为球坐标
            s_x, s_y, s_z = self._latlon_to_cartesian(sender_lat, sender_lon)
            r_x, r_y, r_z = self._latlon_to_cartesian(receiver_lat, receiver_lon)

            # 计算相对位置特征
            rel_x = r_x - s_x
            rel_y = r_y - s_y
            rel_z = r_z - s_z

            # 计算欧几里得距离作为归一化因子 (假设单位球)
            distance_norm_factor = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2) + 1e-8

            # 添加节点纬度、经度和相对位置特征
            # 注意：这里的特征维度与graph_builder中的_get_graph_spatial_features的输出保持一致
            # add_node_latitude=True, add_node_longitude=True, add_relative_positions=True
            feature = [
                np.cos(np.radians(sender_lat)),
                np.sin(np.radians(sender_lat)),
                np.cos(np.radians(sender_lon)),
                np.sin(np.radians(sender_lon)),
                rel_x / distance_norm_factor,
                rel_y / distance_norm_factor,
                rel_z / distance_norm_factor,
            ]
            edge_features.append(feature)

        edge_features = np.array(edge_features)

        # 转换为PyTorch张量
        edge_index = torch.tensor(np.stack([senders, receivers]), dtype=torch.long, device=self.device)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32, device=self.device)

        # 创建mesh图
        self.mesh_graph = Data(
            x=torch.tensor(self._mesh_features, dtype=torch.float32, device=self.device),
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        # self.mesh_edges_dim = edge_features.shape[1] # 更新边特征维度，删除此行

        print(f"基于地理相邻性的Mesh图: {edge_index.shape[1]} 条边")

    def _latlon_to_cartesian(self, lat, lon, radius=1.0):
        """将经纬度转换为笛卡尔坐标 (x, y, z) - 辅助函数"""
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        x = radius * np.cos(lat_rad) * np.cos(lon_rad)
        y = radius * np.cos(lat_rad) * np.sin(lon_rad)
        z = radius * np.sin(lat_rad)
        return x, y, z

    def _init_mesh2grid_graph(self):
        """初始化mesh到grid的图 - 基于区域生长结果建立连接，并添加最近的mesh连接"""
        print("构建基于高分辨率网格的mesh到grid连接...")

        # 获取高分辨率网格的经纬度数据
        fine_grid_lon_2d, fine_grid_lat_2d = np.meshgrid(self._fine_grid_lon, self._fine_grid_lat)
        fine_grid_nodes_lon = fine_grid_lon_2d.flatten()
        fine_grid_nodes_lat = fine_grid_lat_2d.flatten()
        num_fine_grid_nodes = len(fine_grid_nodes_lon)

        senders = []  # mesh节点索引
        receivers = []  # grid节点索引
        edge_features = []  # 边特征

        # 对于每个高分辨率网格点，找到其所属的mesh区域
        # 这里需要一个函数来判断点是否在mesh区域内或其左边界上
        # 为了简化，我们暂时先找到最近的mesh，后续再精确判断
        # nearest_mesh_id = self._find_mesh_for_fine_grid_point(fine_lat, fine_lon)

        # if nearest_mesh_id is not None:
        #     senders.append(nearest_mesh_id)
        #     receivers.append(fine_grid_idx)

        #     # 计算边特征：使用mesh中心到grid点的距离作为权重
        #     mesh_lat = self._mesh_nodes_lat[nearest_mesh_id]
        #     mesh_lon = self._mesh_nodes_lon[nearest_mesh_id]

        #     # 计算欧几里得距离
        #     distance = np.sqrt((mesh_lat - fine_lat)**2 + (mesh_lon - fine_lon)**2)

        #     # 使用距离的倒数作为权重（距离越近权重越大）
        #     edge_weight = 1.0 / (distance + 1e-8)
        #     edge_features.append([edge_weight, 0.0])  # 添加边类型标记：0表示所属mesh连接

        # 直接重用g2m_graph的边，并反转方向
        g2m_edge_index = self.g2m_graph["grid_nodes", "to", "mesh_nodes"].edge_index
        g2m_edge_attr = self.g2m_graph["grid_nodes", "to", "mesh_nodes"].edge_attr

        # 反转边方向
        senders = g2m_edge_index[1]  # mesh_id
        receivers = g2m_edge_index[0]  # fine_grid_idx
        edge_features = g2m_edge_attr

        # 转换为PyTorch张量
        # senders = torch.tensor(senders, dtype=torch.long, device=self.device)
        # receivers = torch.tensor(receivers, dtype=torch.long, device=self.device)
        # edge_features = torch.tensor(edge_features, dtype=torch.float32, device=self.device)

        # 创建异构图
        self.m2g_graph = HeteroData()
        self.m2g_graph["mesh_nodes"].x = torch.tensor(
            np.column_stack([
                np.cos(np.radians(self._mesh_nodes_lat)),
                np.cos(np.radians(self._mesh_nodes_lon)),
                np.sin(np.radians(self._mesh_nodes_lon))
            ]), dtype=torch.float32, device=self.device
        )
        self.m2g_graph["grid_nodes"].x = torch.tensor(
            np.column_stack([
                np.cos(np.radians(fine_grid_nodes_lat)), # 使用高分辨率网格的纬度
                np.cos(np.radians(fine_grid_nodes_lon)), # 使用高分辨率网格的经度
                np.sin(np.radians(fine_grid_nodes_lon))  # 使用高分辨率网格的经度
            ]), dtype=torch.float32, device=self.device
        )
        self.m2g_graph["mesh_nodes", "to", "grid_nodes"].edge_index = torch.stack([senders, receivers])
        self.m2g_graph["mesh_nodes", "to", "grid_nodes"].edge_attr = edge_features

        # 更新边特征维度（现在包含权重和边类型）
        self._m2g_edges_dim = self._g2m_edges_dim

        # 统计边类型
        belonging_edges = sum(1 for edge_feat in edge_features if len(edge_feat) == 2 and edge_feat[1] == 0.0)
        # nearest_edges = sum(1 for edge_feat in edge_features if len(edge_feat) == 2 and edge_feat[1] == 1.0) # 移除此行，因为现在只有所属连接

        print(f"Mesh到Grid连接完成:")
        print(f"  - 基于区域生长的所属mesh连接: {belonging_edges} 条边")
        # print(f"  - 基于距离的最近mesh连接: {nearest_edges} 条边") # 移除此行
        print(f"  - 总连接数: {len(senders)} 条边")
        print(f"  - 连接的mesh节点数量: {len(torch.unique(senders))}")
        print(f"  - 连接的grid点数量: {len(torch.unique(receivers))}") # 这里的数量应该基于新的高分辨率网格点
        print(f"  - 边特征维度: {self._m2g_edges_dim} (权重 + 边类型)")

    def _init_khop_mesh_graph(self):
        """初始化k-hop mesh图"""
        print("构建k-hop mesh图...")

        if self.num_hops == 0:
            self.khop_mesh_graph = self.mesh_graph
            return

        # 使用现有的mesh图构建k-hop图
        edge_index = self.mesh_graph.edge_index
        # edge_attr = self.mesh_graph.edge_attr # 原始边特征可能不再适用

        # 构建稀疏邻接矩阵以计算k-hop连接
        adj = torch.sparse_coo_tensor(
            edge_index,
            values=torch.ones_like(edge_index[0], dtype=torch.float32),
            size=(self.n_mesh, self.n_mesh),
        ).to(self.khop_device)

        adj_k = adj.coalesce()
        # 这里应该从 num_hops=1 开始计算，因为 num_hops=0 已经处理了
        for _ in range(self.num_hops - 1): # 第一次迭代是 1-hop，所以 -1
            adj_k = adj_k + torch.sparse.mm(adj_k, adj)
            adj_k = adj_k.coalesce()

            # 移除自环
            new_indices = adj_k.indices()
            mask = new_indices[0] != new_indices[1]
            new_indices = new_indices[:, mask]
            new_values = torch.ones_like(adj_k.values()[mask])

            adj_k = torch.sparse_coo_tensor(
                indices=new_indices, values=new_values, size=adj_k.shape
            ).coalesce()

            # 清理内存
            del mask, new_indices, new_values
            import gc
            gc.collect()

        # 构建k-hop图
        khop_edges = adj_k.indices().to(self.device)

        # 构建k-hop边特征 (与graph_builder.py兼容)
        if self.add_edge_features_to_khop:
            khop_edge_attr = []
            senders_cpu = khop_edges[0].cpu().numpy()
            receivers_cpu = khop_edges[1].cpu().numpy()

            for i in range(len(senders_cpu)):
                sender_lat = self._mesh_nodes_lat[senders_cpu[i]]
                sender_lon = self._mesh_nodes_lon[senders_cpu[i]]
                receiver_lat = self._mesh_nodes_lat[receivers_cpu[i]]
                receiver_lon = self._mesh_nodes_lon[receivers_cpu[i]]

                # 将经纬度转换为球坐标
                s_x, s_y, s_z = self._latlon_to_cartesian(sender_lat, sender_lon)
                r_x, r_y, r_z = self._latlon_to_cartesian(receiver_lat, receiver_lon)

                # 计算相对位置特征
                rel_x = r_x - s_x
                rel_y = r_y - s_y
                rel_z = r_z - s_z

                # 计算欧几里得距离作为归一化因子 (假设单位球)
                distance_norm_factor = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2) + 1e-8

                # 添加节点纬度、经度和相对位置特征 (与_init_mesh_graph一致)
                feature = [
                    np.cos(np.radians(sender_lat)),
                    np.sin(np.radians(sender_lat)),
                    np.cos(np.radians(sender_lon)),
                    np.sin(np.radians(sender_lon)),
                    rel_x / distance_norm_factor,
                    rel_y / distance_norm_factor,
                    rel_z / distance_norm_factor,
                ]
                khop_edge_attr.append(feature)

            khop_edge_attr = torch.tensor(khop_edge_attr, dtype=torch.float32, device=self.device)
        else:
            khop_edge_attr = None

        self.khop_mesh_graph = Data(
            x=self.mesh_graph.x,
            edge_index=khop_edges,
            edge_attr=khop_edge_attr
        )

        del adj_k, adj
        import gc
        gc.collect()

        print(f"K-hop mesh图: {khop_edges.shape[1]} 条边")

    @property
    def grid_nodes_dim(self):
        """Grid节点特征维度"""
        return self._grid_nodes_dim

    @property
    def mesh_nodes_dim(self):
        """Mesh节点特征维度"""
        return self._mesh_nodes_dim

    @property
    def mesh_edges_dim(self):
        """Mesh边特征维度"""
        return self.mesh_graph.edge_attr.shape[-1] if self.mesh_graph.edge_attr is not None else None

    @property
    def g2m_edges_dim(self):
        """Grid到mesh边特征维度"""
        return self._g2m_edges_dim

    @property
    def m2g_edges_dim(self):
        """Mesh到grid边特征维度"""
        return self._m2g_edges_dim

    def save(self, filepath: str):
        """保存SimpleAdaptiveMeshBuilder到文件

        Args:
            filepath (str): 保存文件路径
        """
        # 收集需要保存的所有属性
        save_data = {
            # 基本配置参数
            'pca_dim': self.pca_dim,
            'threshold_method': self.threshold_method,
            'gradient_threshold': self.gradient_threshold,
            'min_mesh_size': self.min_mesh_size,
            'num_hops': self.num_hops,
            'device': self.device,
            'khop_device': self.khop_device,
            'add_edge_features_to_khop': self.add_edge_features_to_khop,
            'enable_hdbscan_refinement': self.enable_hdbscan_refinement,
            'large_mesh_threshold': self.large_mesh_threshold,
            'hdbscan_min_cluster_size_factor': self.hdbscan_min_cluster_size_factor,
            'hdbscan_min_samples_factor': self.hdbscan_min_samples_factor,
            'cluster_selection_epsilon': self.cluster_selection_epsilon,
            'cluster_selection_method': self.cluster_selection_method,
            'alpha': self.alpha,

            # 网格属性
            '_grid_lat': self._grid_lat,
            '_grid_lon': self._grid_lon,
            '_num_lat': self._num_lat,
            '_num_lon': self._num_lon,
            '_num_grid_nodes': self._num_grid_nodes,
            '_grid_nodes_lon': self._grid_nodes_lon,
            '_grid_nodes_lat': self._grid_nodes_lat,

            # 高分辨率网格属性
            '_fine_grid_lon': self._fine_grid_lon,
            '_fine_grid_lat': self._fine_grid_lat,
            'fine_grid_nodes_lon': self.fine_grid_nodes_lon,
            'fine_grid_nodes_lat': self.fine_grid_nodes_lat,

            # mesh属性
            'mesh_labels': self.mesh_labels,
            'n_mesh': self.n_mesh,
            'mesh_nodes': self.mesh_nodes,
            '_mesh_nodes_lat': self._mesh_nodes_lat,
            '_mesh_nodes_lon': self._mesh_nodes_lon,
            '_mesh_features': self._mesh_features,
            '_mesh_distances': self._mesh_distances,

            # 特征数据
            '_grid_features_raw': self._grid_features_raw,

            # 图结构
            'g2m_graph': self.g2m_graph,
            'mesh_graph': self.mesh_graph,
            'm2g_graph': self.m2g_graph,
            'khop_mesh_graph': self.khop_mesh_graph,

            # 维度信息
            '_grid_nodes_dim': self._grid_nodes_dim,
            '_mesh_nodes_dim': self._mesh_nodes_dim,
            '_g2m_edges_dim': self._g2m_edges_dim,
            '_m2g_edges_dim': self._m2g_edges_dim,

            # 其他属性
            '_unassigned_fine_points': self._unassigned_fine_points,
        }

        # 保存到文件
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"SimpleAdaptiveMeshBuilder已保存到: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'SimpleAdaptiveMeshBuilder':
        """从文件加载SimpleAdaptiveMeshBuilder

        Args:
            filepath (str): 保存文件路径

        Returns:
            SimpleAdaptiveMeshBuilder: 加载的类实例
        """
        # 从文件加载数据
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        # 创建一个空的实例
        instance = cls.__new__(cls)

        # 恢复所有属性
        for key, value in save_data.items():
            setattr(instance, key, value)

        print(f"SimpleAdaptiveMeshBuilder已从{filepath}加载")
        return instance

    def _get_fine_grid_neighbors(self, fine_grid_idx: int) -> list[Union[int, None]]:
        """
        获取高分辨率grid点的8连通邻居（上下左右+四个对角方向，只考虑海洋部分）。
        """
        # 获取高分辨率网格的维度
        num_fine_lat = len(self._fine_grid_lat)
        num_fine_lon = len(self._fine_grid_lon)

        row = fine_grid_idx // num_fine_lon
        col = fine_grid_idx % num_fine_lon

        neighbors = []

        # 定义8个方向的偏移量 (dr, dc)
        deltas = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # 上下左右
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # 四个对角
        ]

        for dr, dc in deltas:
            n_row, n_col = row + dr, col + dc

            if 0 <= n_row < num_fine_lat and 0 <= n_col < num_fine_lon:
                neighbor_fine_grid_idx = n_row * num_fine_lon + n_col
                # 检查是否为海洋点
                if fine_mask_flat[neighbor_fine_grid_idx] == 1: # 假设fine_mask_flat是全局可访问的
                    neighbors.append(neighbor_fine_grid_idx)
                else:
                    neighbors.append(None) # 陆地或无效点
            else:
                neighbors.append(None) # 超出边界

        return [n for n in neighbors if n is not None]


def create_simple_adaptive_mesh_builder(
        path: str = None,
        grid_lon: np.ndarray = None,
        grid_lat: np.ndarray = None,
        grid_features: np.ndarray = None,
        fine_grid_lon: np.ndarray = None,  # 新增参数
        fine_grid_lat: np.ndarray = None,   # 新增参数
        fine_grid_features: np.ndarray = None,
        pca_dim: int = 1,
        threshold_method: str = "kmeans",
        gradient_threshold: float = None,
        min_mesh_size: int = 5,
        num_hops: int = 6,
        device: torch.device = torch.device("cpu"),
        # 新增HDBSCAN相关参数
        enable_hdbscan_refinement: bool = True,
        large_mesh_threshold: float = 0.2,
        hdbscan_min_cluster_size_factor: float = 0.05,
        hdbscan_min_samples_factor: float = 0.01,
        # 新增与adaptive文件一致的HDBSCAN参数
        cluster_selection_epsilon: float = 0.1,
        cluster_selection_method: str = 'leaf',
        alpha: float = 1.0,
        **kwargs
) -> SimpleAdaptiveMeshBuilder:
    """创建基于区域生长的简单自适应mesh-grid构建器的便捷函数

    如果提供path参数，则从文件加载已保存的builder，否则创建新的builder
    """
    if path is not None:
        return SimpleAdaptiveMeshBuilder(path=path)

    # 创建新的builder
    grid_features = grid_features.squeeze(0)  # shape: (94, 121, 256)

    # 变换为 (121*256, 94)
    grid_features = np.transpose(grid_features, (1, 2, 0))  # shape: (121, 256, 94)
    grid_features = grid_features.reshape(-1, 94)  # shape: (30976, 94)

    fine_grid_features = fine_grid_features.squeeze(0)
    fine_grid_features = np.transpose(fine_grid_features, (1, 2, 0))
    fine_grid_features = fine_grid_features.reshape(-1, 94)

    return SimpleAdaptiveMeshBuilder(
        path=None,
        grid_lon=grid_lon,
        grid_lat=grid_lat,
        grid_features=grid_features,
        fine_grid_lon=fine_grid_lon,  # 传递给构造函数
        fine_grid_lat=fine_grid_lat,   # 传递给构造函数
        fine_grid_features=fine_grid_features,
        pca_dim=pca_dim,
        threshold_method=threshold_method,
        gradient_threshold=gradient_threshold,
        min_mesh_size=min_mesh_size,
        num_hops=num_hops,
        device=device,
        enable_hdbscan_refinement=enable_hdbscan_refinement,
        large_mesh_threshold=large_mesh_threshold,
        hdbscan_min_cluster_size_factor=hdbscan_min_cluster_size_factor,
        hdbscan_min_samples_factor=hdbscan_min_samples_factor,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        alpha=alpha,
        **kwargs
    )


def visualize_simple_mesh_results(builder, grid_lat, grid_lon, lat2,  lon2):
    """可视化基于区域生长的简单mesh构建结果（支持HDBSCAN优化）"""
    lon2d, lat2d = np.meshgrid(grid_lon, grid_lat)
    labels_2d = builder.mesh_labels.reshape((len(grid_lat), len(grid_lon)))

    # 创建用于可视化的标签数组，将陆地部分设为NaN，海洋部分重新映射
    viz_labels_2d = labels_2d.astype(np.float16)

    # 将陆地部分（-1）设为NaN，这样在可视化中会显示为透明
    viz_labels_2d[labels_2d == -2] = np.nan

    # 重新映射海洋部分的mesh标签，使其从0开始连续
    ocean_labels = labels_2d[labels_2d >= 0]
    unique_ocean_labels = np.unique(ocean_labels)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_ocean_labels)}

    # 应用映射
    for old_label, new_label in label_mapping.items():
        viz_labels_2d[labels_2d == old_label] = new_label

    # 统计每个mesh对应的grid节点数量
    print("\n=== Mesh统计信息（HDBSCAN优化后） ===")
    print(f"总共有 {len(unique_ocean_labels)} 个唯一的mesh区域")

    # 统计每个mesh的grid节点数量
    mesh_grid_counts = {}
    for old_label in unique_ocean_labels:
        new_label = label_mapping[old_label]
        grid_count = np.sum(labels_2d == old_label)
        mesh_grid_counts[new_label] = grid_count

    # 按grid节点数量排序并打印
    sorted_mesh_counts = sorted(mesh_grid_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\n每个mesh对应的grid节点数量（按数量降序排列）:")
    print(f"{'Mesh ID':<8} {'Grid节点数':<12} {'占比(%)':<8}")
    print("-" * 30)

    total_ocean_grids = sum(mesh_grid_counts.values())
    # for mesh_id, grid_count in sorted_mesh_counts:
    #     percentage = (grid_count / total_ocean_grids) * 100
    #     print(f"{mesh_id:<8} {grid_count:<12} {percentage:<8.2f}")

    # 统计信息摘要
    grid_counts_array = np.array(list(mesh_grid_counts.values()))
    print(f"\n统计摘要:")
    print(f"最小mesh大小: {grid_counts_array.min()}")
    print(f"最大mesh大小: {grid_counts_array.max()}")
    print(f"平均mesh大小: {grid_counts_array.mean():.2f}")
    print(f"中位数mesh大小: {np.median(grid_counts_array):.2f}")
    print(f"标准差: {grid_counts_array.std():.2f}")

    # 检查是否还有过大的mesh
    large_mesh_threshold = getattr(builder, 'large_mesh_threshold', 0.2)
    large_meshes = []
    for mesh_id, grid_count in sorted_mesh_counts:
        percentage = grid_count / total_ocean_grids
        if percentage > large_mesh_threshold:
            large_meshes.append((mesh_id, grid_count, percentage))

    if large_meshes:
        print(f"\n警告: 仍有 {len(large_meshes)} 个过大的mesh:")
        for mesh_id, grid_count, percentage in large_meshes:
            print(f"  Mesh {mesh_id}: {grid_count} 个grid节点 ({percentage:.2%})")
    else:
        print(f"\n成功: 所有mesh大小都在合理范围内（<{large_mesh_threshold:.0%}）")

    # 动态选择颜色映射，确保有足够的颜色
    num_unique_meshes = len(unique_ocean_labels)

    if num_unique_meshes <= 20:
        base_cmap = plt.get_cmap('tab20', num_unique_meshes)
    elif num_unique_meshes <= 40: # 增加一个tab20b或者tab20c来提供更多离散颜色
        base_cmap = plt.get_cmap('tab20b', num_unique_meshes) # 或者 'tab20c'
    else:
        # 对于更多的离散颜色需求，直接使用连续颜色映射并指定颜色数量
        base_cmap = plt.get_cmap('gist_ncar', num_unique_meshes)

    # 为避免相邻ID颜色过于相似，打乱颜色顺序（固定随机种子保证可复现）
    rng = np.random.default_rng(2025)
    shuffled_indices = rng.permutation(num_unique_meshes)
    base_colors = base_cmap(np.linspace(0, 1, num_unique_meshes, endpoint=False))
    shuffled_colors = base_colors[shuffled_indices]
    cmap = ListedColormap(shuffled_colors)

    # 创建图形
    fig = plt.figure(figsize=(20, 12)) # 恢复图形大小

    # 子图1：聚类结果
    # ax1 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree()) # 恢复为2行2列，第1个
    # ax1.set_global()
    # ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
    # ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
    #
    # # 使用处理后的标签进行可视化
    # print(lon2d.shape, lat2d.shape, viz_labels_2d.shape)
    # im1 = ax1.pcolormesh(lon2d, lat2d, viz_labels_2d, cmap=cmap, shading='auto',
    #                      transform=ccrs.PlateCarree())
    # plt.colorbar(im1, ax=ax1, label='Mesh ID')
    # ax1.set_title(f'Combined Mesh Clustering (Ocean Only, {num_unique_meshes} meshes)', fontsize=14) # 更新标题
    # ax1.set_xlabel('Longitude')
    # ax1.set_ylabel('Latitude')

    # 子图2：梯度图（只显示海洋部分）
    ax2 = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree()) # 恢复为2行2列，第2个
    ax2.set_global()
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.5)

    gradient_2d = builder.gradient_magnitude.reshape((121, 256))
    # 将陆地部分的梯度设为NaN
    gradient_viz = gradient_2d.astype(float)  # 转换为float类型以支持NaN
    # gradient_viz[labels_2d == -2] = np.nan
    gradient_viz[mask==0] = np.nan

    im2 = ax2.pcolormesh(lon2, lat2, gradient_viz, cmap='viridis', shading='auto',
                         transform=ccrs.PlateCarree())
    plt.colorbar(im2, ax=ax2, label='Gradient Magnitude')
    ax2.set_title('Gradient Magnitude (Ocean Only)', fontsize=14)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')

    # 子图3：mesh节点和Delaunay三角剖分边 (新)
    # ax3 = plt.subplot(2, 2, 3, projection=ccrs.PlateCarree()) # 恢复为2行2列，第3个
    # ax3.set_global()
    # ax3.add_feature(cfeature.COASTLINE, linewidth=0.5)
    # ax3.add_feature(cfeature.BORDERS, linewidth=0.5)
    #
    # # 绘制聚类结果作为背景（只显示海洋部分）
    # im3 = ax3.pcolormesh(lon2d, lat2d, viz_labels_2d, cmap=cmap, shading='auto', alpha=0.3,
    #                      transform=ccrs.PlateCarree())
    #
    # # 获取mesh节点位置
    # mesh_lats = builder._mesh_nodes_lat
    # mesh_lons = builder._mesh_nodes_lon
    #
    # # 绘制mesh节点
    # ax3.scatter(mesh_lons, mesh_lats, c='red', s=5, marker='o', label='Mesh Nodes',
    #             transform=ccrs.PlateCarree(), zorder=5)
    #
    # # 获取mesh图的边（现在是Delaunay三角剖分边）
    # mesh_graph = builder.mesh_graph
    # edge_index = mesh_graph.edge_index.cpu().numpy()
    #
    # # 绘制Delaunay三角剖分的边
    # for i in range(edge_index.shape[1]):
    #     sender_idx = edge_index[0, i]
    #     receiver_idx = edge_index[1, i]
    #
    #     sender_lon = mesh_lons[sender_idx]
    #     sender_lat = mesh_lats[sender_idx]
    #     receiver_lon = mesh_lons[receiver_idx]
    #     receiver_lat = mesh_lats[receiver_idx]
    #
    #     ax3.plot([sender_lon, receiver_lon], [sender_lat, receiver_lat],
    #              color='#87CEEB', linewidth=0.6, alpha=0.5, zorder=3, transform=ccrs.PlateCarree())
    #
    # ax3.set_title(f'Edges on Mesh Nodes (Ocean Only, {num_unique_meshes} meshes)', fontsize=14) # 更新标题
    # ax3.set_xlabel('Longitude')
    # ax3.set_ylabel('Latitude')
    # ax3.legend(loc='upper right')

    # # 子图4：mesh2grid连接可视化 (新)
    # ax4 = plt.subplot(2, 2, 4, projection=ccrs.PlateCarree()) # 恢复为2行2列，第4个
    # ax4.set_global()
    # ax4.add_feature(cfeature.COASTLINE, linewidth=0.5)
    # ax4.add_feature(cfeature.BORDERS, linewidth=0.5)
    #
    # # 绘制网格作为背景
    # ax4.pcolormesh(lon2d, lat2d, viz_labels_2d, cmap=cmap, shading='auto', alpha=0.3, transform=ccrs.PlateCarree())
    #
    # # 绘制mesh节点
    # ax4.scatter(mesh_lons, mesh_lats, c='red', s=30, marker='o', label='Mesh Nodes', transform=ccrs.PlateCarree(), zorder=5)
    #
    # # 绘制mesh到grid的连接
    # m2g_graph = builder.m2g_graph
    # m2g_edge_index = m2g_graph["mesh_nodes", "to", "grid_nodes"].edge_index.cpu().numpy()
    #
    # # 随机选择一部分边进行绘制，避免过于密集
    # num_m2g_edges_to_plot = min(10000, m2g_edge_index.shape[1]) # 最多绘制10000条边
    # if m2g_edge_index.shape[1] > 0:
    #     plot_indices = np.random.choice(m2g_edge_index.shape[1], num_m2g_edges_to_plot, replace=False)
    #
    #     for i in plot_indices:
    #         sender_mesh_idx = m2g_edge_index[0, i]
    #         receiver_grid_idx = m2g_edge_index[1, i]
    #
    #         mesh_lon = mesh_lons[sender_mesh_idx]
    #         mesh_lat = mesh_lats[sender_mesh_idx]
    #         grid_lon_val = builder._fine_grid_lon[receiver_grid_idx] # 使用builder的fine_grid_lon
    #         grid_lat_val = builder._fine_grid_lat[receiver_grid_idx] # 使用builder的fine_grid_lat
    #
    #         ax4.plot([mesh_lon, grid_lon_val], [mesh_lat, grid_lat_val],
    #                  'g--', linewidth=0.5, alpha=0.3, zorder=2, transform=ccrs.PlateCarree())
    #
    # ax4.set_title('Mesh to Grid Connections (Sampled)', fontsize=14)
    # ax4.set_xlabel('Longitude')
    # ax4.set_ylabel('Latitude')
    # ax4.legend(loc='upper right')


    plt.tight_layout()
    plt.savefig('simple_mesh_results.png', dpi=300, bbox_inches='tight')
    plt.close()


# 使用示例和测试保存/加载功能
if __name__ == "__main__":
    mask = np.load('../data/land_sea_mask1.4.npy')[0, 4]
    mask_flat = mask.flatten()
    # fine_mask_flat = np.load('../data/union_ocean_mask.npy')[::3,::3].flatten()
    fine_mask_flat = (np.load('../data/union_ocean_mask.npy')[::3, ::3] * np.load('../data/total0.25mask.npy')[0][
        4]).flatten()
    grid_features = np.load('../data/climatology/mean1.4.npy')
    grid_features = np.nan_to_num(grid_features, nan=0)
    fine_grid_features = np.load('../data/climatology/mean.npy')[:,:,::3,::3]
    fine_grid_features = np.nan_to_num(fine_grid_features, nan=0)

    # 创建网格坐标
    oceandata = xr.open_dataset('../data/19930101.nc')
    num_lat, num_lon = 121, 256
    grid_lon = oceandata['lon'].values.astype(np.float32)
    grid_lat = oceandata['lat'].values

    mercator = xr.open_dataset('../data/mercatorglorys12v1_gl12_mean_1993_2016_01.nc')
    lon1 = mercator['longitude'].values[::3]
    lat1 = mercator['latitude'].values[::3]

    # 测试保存和加载功能
    save_path = './simple_adaptive_mesh_builder2.pkl'

    # 首先创建并保存builder
    print("创建基于区域生长的简单自适应mesh构建器...")
    builder = create_simple_adaptive_mesh_builder(
        grid_lon=grid_lon,
        grid_lat=grid_lat,
        grid_features=grid_features,
        fine_grid_lon=lon1,
        fine_grid_lat=lat1,
        fine_grid_features=fine_grid_features,
        pca_dim=1,
        threshold_method="percentile",
        gradient_threshold=0.1,
        min_mesh_size=5,
        num_hops=2,
        device=torch.device("cpu"),
        enable_hdbscan_refinement=True,
        large_mesh_threshold=0.15,
        hdbscan_min_cluster_size_factor=0.03,
        hdbscan_min_samples_factor=0.01,
    )

    # print("保存builder到文件...")
    # builder.save(save_path)
    #
    # print("从文件加载builder...")
    # loaded_builder = create_simple_adaptive_mesh_builder(path=save_path)
    #
    # # 验证加载的builder是否正确
    # print("验证加载结果...")
    # print(f"原始builder mesh数量: {builder.n_mesh}")
    # print(f"加载builder mesh数量: {loaded_builder.n_mesh}")
    # print(f"mesh数量匹配: {builder.n_mesh == loaded_builder.n_mesh}")
    #
    # print(f"原始builder grid节点维度: {builder.grid_nodes_dim}")
    # print(f"加载builder grid节点维度: {loaded_builder.grid_nodes_dim}")
    # print(f"grid节点维度匹配: {builder.grid_nodes_dim == loaded_builder.grid_nodes_dim}")

    # 可视化结果
    print("可视化结果...")
    visualize_simple_mesh_results(builder, lat1, lon1, grid_lat, grid_lon)

    print("测试完成！")