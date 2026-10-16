"""Build the three graphs for GenCast.

The following code is a port of several components from GraphCast's original graph generation
(https://github.com/google-deepmind/graphcast) to PyG and PyTorch. The graphs are:
- g2m: grid to mesh.
- mesh: icosphere refinement.
- m2g: mesh to grid.
- khop: k-hop neighbours mesh.
"""

import gc

import numpy as np
import torch
from torch_geometric.data import Data, HeteroData

# from torch_geometric.transforms import TwoHop
from graph import grid_mesh_connectivity, icosahedral_mesh, model_utils

# Some configs from graphcast:
_spatial_features_kwargs = dict(
    add_node_positions=False,
    add_node_latitude=True,
    add_node_longitude=True,
    add_relative_positions=True,
    relative_longitude_local_coordinates=True,
    relative_latitude_local_coordinates=True,
)

# radius_query_fraction_edge_length: Scalar that will be multiplied by the
#   length of the longest edge of the finest mesh to define the radius of
#   connectivity to use in the Grid2Mesh graph. Reasonable values are
#   between 0.6 and 1. 0.6 reduces the number of grid points feeding into
#   multiple mesh nodes and therefore reduces edge count and memory use, but
#   1 gives better predictions.
# mesh2grid_edge_normalization_factor: Allows explicitly controlling edge
#   normalization for mesh2grid edges. If None, defaults to max edge length.
#   This supports using pre-trained model weights with a different graph
#   structure to what it was trained on.

radius_query_fraction_edge_length = 0.6
mesh2grid_edge_normalization_factor = None


def _get_max_edge_distance(mesh):
    senders, receivers = icosahedral_mesh.faces_to_edges(mesh.faces)
    edge_distances = np.linalg.norm(mesh.vertices[senders] - mesh.vertices[receivers], axis=-1)
    return edge_distances.max()


class GraphBuilder:
    """
    Class for building GenCast's graphs.

    Attributes:
        g2m_graph (pyg.data.HeteroData): heterogeneous directed graph connecting the grid nodes
            to the mesh nodes.
        mesh_graph (pyg.data.Data): undirected graph connecting the mesh nodes.
        m2g_graph (pyg.data.HeteroData): heterogeneous directed graph connecting the mesh nodes
            to the grid nodes.
        khop_mesh_graph (pyg.data.Data): augmented version of mesh_graph in which every node is
            connected to its num_hops neighbours.
        grid_nodes_dim (int): dimension of the grid nodes features.
        mesh_nodes_dim (int): dimension of the mesh nodes features.
        mesh_edges_dim (int): dimension of the mesh edges features.
        g2m_edges_dim (int): dimension of the "grid to mesh" edges features.
        m2g_edges_dim (int): dimension of the "mesh to grid" edges features.
    """

    def __init__(
        self,
        grid_lon: np.ndarray,
        grid_lat: np.ndarray,
        splits: int = 5,
        num_hops: int = 0,
        device: torch.device = torch.device("cpu"),
        khop_device: torch.device = torch.device("cpu"),
        add_edge_features_to_khop=True,
    ):
        """Initialize the GraphBuilder object.

        Args:
            grid_lon (np.ndarray): 1D np.ndarray containing the list of longitudes.
            grid_lat: (np.ndarray) 1D np.ndarray containing the list of latitudes.
            splits (int): number of times to split the icosphere to build the mesh. Defaults to 5.
            num_hops (int): if num_hops=k then khop_mesh_graph will be the k-neighbours version of
                the mesh. Defaults to 0.
            device: the device to which the final graph will be moved.
            khop_device: the device that will compute the k-hop mesh graph. Note that while setting
                this to gpu may result in faster computations, it may also cause some memory leaks
                in the current implementation. Defaults to cpu.
            add_edge_features_to_khop (bool): if true compute edge features for the k-hop neighbours
                graph. Defaults to False.
        """

        self._spatial_features_kwargs = _spatial_features_kwargs
        self.add_edge_features_to_khop = add_edge_features_to_khop
        self.device = device
        self.khop_device = khop_device

        # Specification of the mesh.
        _icosahedral_refinements = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
            splits
        )
        self._mesh = _icosahedral_refinements[-1]
        # 合并所有层级的 mesh，包括原始 icosahedron
        self._mesh_total = self._merge_meshes(_icosahedral_refinements)

        # Obtain the query radius in absolute units for the unit-sphere for the
        # grid2mesh model, by rescaling the `radius_query_fraction_edge_length`.
        self._query_radius = _get_max_edge_distance(self._mesh) * radius_query_fraction_edge_length
        self._mesh2grid_edge_normalization_factor = mesh2grid_edge_normalization_factor

        self.grid_nodes_dim = None
        self.mesh_nodes_dim = None
        self.mesh_edges_dim = None
        self.g2m_edges_dim = None
        self.m2g_edges_dim = None

        # A "_init_mesh_properties":
        # This one could be initialized at init but we delay it for consistency too.
        self._num_mesh_nodes = None  # num_mesh_nodes
        self._mesh_nodes_lat = None  # [num_mesh_nodes]
        self._mesh_nodes_lon = None  # [num_mesh_nodes]

        # A "_init_grid_properties":
        self._grid_lat = None  # [num_lat_points]
        self._grid_lon = None  # [num_lon_points]
        self._num_grid_nodes = None  # num_lat_points * num_lon_points
        self._grid_nodes_lat = None  # [num_grid_nodes]
        self._grid_nodes_lon = None  # [num_grid_nodes]

        self._init_grid_properties(grid_lat, grid_lon)
        self._init_mesh_properties()
        self.g2m_graph = self._init_grid2mesh_graph()
        self.mesh_graph = self._init_mesh_graph()
        self.m2g_graph = self._init_mesh2grid_graph()

        self.num_hops = num_hops
        self.khop_mesh_graph = self._init_khop_mesh_graph()

    @staticmethod
    def _merge_meshes(meshes: list) -> icosahedral_mesh.TriangularMesh:
        """将多个层级的 TriangularMesh 合并成一个完整的 mesh。

        Args:
            meshes: 从低分辨率到高分辨率的 TriangularMesh 列表

        Returns:
            包含所有层级的顶点和面的 TriangularMesh
        """
        if not meshes:
            raise ValueError("meshes list cannot be empty")

        if len(meshes) == 1:
            return meshes[0]

        all_vertices = []
        all_faces = []

        for mesh in meshes:
            all_vertices.append(mesh.vertices)
            # 调整面索引的偏移量
            # faces_with_offset = mesh.faces + vertex_offset
            all_faces.append(mesh.faces)
            # vertex_offset += mesh.vertices.shape[0]

        # 合并所有顶点和面
        merged_vertices = np.vstack(all_vertices)
        merged_faces = np.vstack(all_faces)

        return icosahedral_mesh.TriangularMesh(
            vertices=merged_vertices,
            faces=merged_faces
        )


    def _init_grid_properties(self, grid_lat: np.ndarray, grid_lon: np.ndarray):
        """Inits static properties that have to do with grid nodes."""
        self._grid_lat = grid_lat.astype(np.float32)
        self._grid_lon = grid_lon.astype(np.float32)
        # Initialized the counters.
        self._num_grid_nodes = grid_lat.shape[0] * grid_lon.shape[0]

        # Initialize lat and lon for the grid.
        grid_nodes_lon, grid_nodes_lat = np.meshgrid(grid_lon, grid_lat)
        self._grid_nodes_lon = grid_nodes_lon.reshape([-1]).astype(np.float32)
        self._grid_nodes_lat = grid_nodes_lat.reshape([-1]).astype(np.float32)

    def _init_mesh_properties(self):
        """Inits static properties that have to do with mesh nodes."""
        '''mesh包括vertices和faces两个部分,vertices的shape: [num_nodes, 3],3代表xyz球坐标，faces的shape:[num_faces, 3]，
        3代表每个三角面片由3个节点索引组成，分别代表组成该三角形的三个顶点在 vertices 数组中的下标'''
        self._num_mesh_nodes = self._mesh.vertices.shape[0]
        mesh_phi, mesh_theta = model_utils.cartesian_to_spherical(
            self._mesh.vertices[:, 0], self._mesh.vertices[:, 1], self._mesh.vertices[:, 2]
        )  # (2562,) (2562,) _mesh.faces.shape(5120, 3)
        (
            mesh_nodes_lat,
            mesh_nodes_lon,
        ) = model_utils.spherical_to_lat_lon(phi=mesh_phi, theta=mesh_theta)
        # Convert to f32 to ensure the lat/lon features aren't in f64.
        self._mesh_nodes_lat = mesh_nodes_lat.astype(np.float32)
        self._mesh_nodes_lon = mesh_nodes_lon.astype(np.float32)

        # For mesh_total: compute lat/lon for all levels 0-6
        self._num_mesh_total_nodes = self._mesh_total.vertices.shape[0]
        mesh_total_phi, mesh_total_theta = model_utils.cartesian_to_spherical(
            self._mesh_total.vertices[:, 0],
            self._mesh_total.vertices[:, 1],
            self._mesh_total.vertices[:, 2]
        )
        (
            mesh_total_nodes_lat,
            mesh_total_nodes_lon,
        ) = model_utils.spherical_to_lat_lon(phi=mesh_total_phi, theta=mesh_total_theta)
        self._mesh_total_nodes_lat = mesh_total_nodes_lat.astype(np.float32)
        self._mesh_total_nodes_lon = mesh_total_nodes_lon.astype(np.float32)

    def _init_grid2mesh_graph(self):
        """Build Grid2Mesh graph."""

        # Create some edges according to distance between mesh and grid nodes.
        assert self._grid_lat is not None and self._grid_lon is not None
        (grid_indices, mesh_indices) = grid_mesh_connectivity.radius_query_indices(
            grid_latitude=self._grid_lat,
            grid_longitude=self._grid_lon,
            mesh=self._mesh,
            radius=self._query_radius,
        )

        # Edges sending info from grid to mesh.
        senders = grid_indices # (51745,)
        receivers = mesh_indices # (51745,)

        # Precompute structural node and edge features according to config options.
        # Structural features are those that depend on the fixed values of the
        # latitude and longitudes of the nodes.
        (senders_node_features, receivers_node_features, edge_features) = (
            model_utils.get_bipartite_graph_spatial_features(
                senders_node_lat=self._grid_nodes_lat,
                senders_node_lon=self._grid_nodes_lon,
                receivers_node_lat=self._mesh_nodes_lat,
                receivers_node_lon=self._mesh_nodes_lon,
                senders=senders,
                receivers=receivers,
                edge_normalization_factor=None,
                **self._spatial_features_kwargs,
            )
        )
        #senders_node_features (128*256, 3) 一个节点有纬度的余弦、经度的余弦、经度的正弦三个特征
        #receivers_node_features (2562, 3) 2562 个 mesh 节点，每个节点有 3 维空间特征（纬度余弦、经度余弦、经度正弦）
        #edge_features.shape (51745, 4) grid到mesh的边

        self.grid_nodes_dim = senders_node_features.shape[1]
        self.mesh_nodes_dim = receivers_node_features.shape[1]
        self.g2m_edges_dim = edge_features.shape[1]

        g2m_graph = HeteroData()
        g2m_graph["grid_nodes"].x = torch.tensor(
            senders_node_features, dtype=torch.float32, device=self.device
        )  # TODO: generate graph with torch or np?
        g2m_graph["mesh_nodes"].x = torch.tensor(
            receivers_node_features, dtype=torch.float32, device=self.device
        )
        g2m_graph["grid_nodes", "to", "mesh_nodes"].edge_index = torch.tensor(
            np.stack([senders, receivers]), dtype=torch.long, device=self.device
        )
        g2m_graph["grid_nodes", "to", "mesh_nodes"].edge_attr = torch.tensor(
            edge_features, dtype=torch.float32, device=self.device
        )

        return g2m_graph

    def _init_mesh_graph(self):
        """Build Mesh graph with all levels 0-6."""
        # Use mesh_total to include edges from all refinement levels
        senders, receivers = icosahedral_mesh.faces_to_edges(self._mesh_total.faces)

        # Precompute structural node and edge features for mesh_total
        assert self._mesh_total_nodes_lat is not None and self._mesh_total_nodes_lon is not None
        node_features, edge_features = model_utils.get_graph_spatial_features(
            node_lat=self._mesh_total_nodes_lat,
            node_lon=self._mesh_total_nodes_lon,
            senders=senders,
            receivers=receivers,
            **self._spatial_features_kwargs,
        )  # node_features：为每个 mesh_total 节点生成空间特征（如纬度余弦、经度余弦、经度正弦）
        # edge_features：为每条 mesh 边生成空间特征（如归一化距离、归一化相对 x/y/z 坐标）
        self.mesh_edges_dim = edge_features.shape[1]

        mesh_graph = Data(
            x=torch.tensor(node_features, dtype=torch.float32, device=self.device),
            edge_attr=torch.tensor(edge_features, dtype=torch.float32, device=self.device),
            edge_index=torch.tensor(
                np.stack([senders, receivers]), dtype=torch.long, device=self.device
            ),
        )

        return mesh_graph

    def _init_mesh2grid_graph(self):
        """Build Mesh2Grid graph."""

        # Create some edges according to how the grid nodes are contained by
        # mesh triangles.
        (grid_indices, mesh_indices) = grid_mesh_connectivity.in_mesh_triangle_indices(
            grid_latitude=self._grid_lat, grid_longitude=self._grid_lon, mesh=self._mesh
        )

        # Edges sending info from mesh to grid.
        senders = mesh_indices
        receivers = grid_indices

        # Precompute structural node and edge features according to config options.
        assert self._mesh_nodes_lat is not None and self._mesh_nodes_lon is not None
        (senders_node_features, receivers_node_features, edge_features) = (
            model_utils.get_bipartite_graph_spatial_features(
                senders_node_lat=self._mesh_nodes_lat,
                senders_node_lon=self._mesh_nodes_lon,
                receivers_node_lat=self._grid_nodes_lat,
                receivers_node_lon=self._grid_nodes_lon,
                senders=senders,
                receivers=receivers,
                edge_normalization_factor=self._mesh2grid_edge_normalization_factor,
                **self._spatial_features_kwargs,
            )
        )

        self.m2g_edges_dim = edge_features.shape[1]

        m2g_graph = HeteroData()
        m2g_graph["mesh_nodes"].x = torch.tensor(
            senders_node_features, dtype=torch.float32, device=self.device
        )
        m2g_graph["grid_nodes"].x = torch.tensor(
            receivers_node_features, dtype=torch.float32, device=self.device
        )
        m2g_graph["mesh_nodes", "to", "grid_nodes"].edge_index = torch.tensor(
            np.stack([senders, receivers]), dtype=torch.long, device=self.device
        )
        m2g_graph["mesh_nodes", "to", "grid_nodes"].edge_attr = torch.tensor(
            edge_features, dtype=torch.float32, device=self.device
        )

        return m2g_graph

    def _init_khop_mesh_graph(self):
        """Build k-hop Mesh graph.

        This implementation constructs the sparse adjacency matrix associated with the mesh graph
        and computes its powers in a sparse manner.
        """
        return self.mesh_graph


