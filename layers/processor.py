"""Processor layer.

The processor:
- performs 16 unshared message-passing steps on the multi-mesh using interaction networks.
- adds residual connections after each step for both nodes and edges.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from torch.utils.checkpoint import checkpoint

from layers.modules import ConditionalLayerNorm


class ProcessorLayer(nn.Module):
    """Single processor layer (Interaction Network) with edge and node updates."""

    def __init__(
            self,
            latent_dim: int,
            hidden_dims: list,
            activation_layer: nn.Module = nn.ReLU,
            use_layer_norm: bool = True,
            noise_emb_dim: int | None = None,
    ):
        """Initialize a single processor layer.

        Args:
            latent_dim (int): dimension of latent features.
            hidden_dims (list[int]): hidden dimensions of internal MLPs.
            activation_layer (torch.nn.Module): activation function.
            use_layer_norm (bool): if true add LayerNorm after each MLP.
            noise_emb_dim (int, optional): dimension of noise embedding for conditioning. Defaults to None.
        """
        super().__init__()

        # Edge update MLP: e' = MLP([e, v_s, v_r])
        # Input: edge_attr (latent_dim) + sender_node (latent_dim) + receiver_node (latent_dim)
        self.edge_mlp = MLP(
            input_dim=3 * latent_dim,
            hidden_dims=hidden_dims,
            activation_layer=activation_layer,
            use_layer_norm=use_layer_norm,
            bias=True,
            activate_final=False,
        )

        # Node update MLP: v' = MLP([v, sum(e')])
        # Input: node (latent_dim) + aggregated_edge (hidden_dims[-1])
        self.node_mlp = MLP(
            input_dim=latent_dim + hidden_dims[-1],
            hidden_dims=hidden_dims,
            activation_layer=activation_layer,
            use_layer_norm=use_layer_norm,
            bias=True,
            activate_final=False,
        )

        # Conditional normalization for nodes
        if noise_emb_dim is not None:
            self.cond_norm = ConditionalLayerNorm(
                conditioning_dim=noise_emb_dim,
                features_dim=hidden_dims[-1],
            )
        else:
            self.cond_norm = None

    def forward(
            self,
            nodes: torch.Tensor,
            edges: torch.Tensor,
            edge_index: torch.Tensor,
            noise_vector: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of a single processor layer.

        Args:
            nodes (torch.Tensor): node features [num_nodes, latent_dim]
            edges (torch.Tensor): edge features [num_edges, latent_dim]
            edge_index (torch.Tensor): edge indices [2, num_edges]
            noise_vector (torch.Tensor, optional): noise vector for conditioning

        Returns:
            tuple[torch.Tensor, torch.Tensor]: updated nodes and edges
        """
        sender_idx, receiver_idx = edge_index

        # 1. Edge update: e' = MLP([e, v_s, v_r])
        sender_nodes = nodes[sender_idx]  # [num_edges, latent_dim]
        receiver_nodes = nodes[receiver_idx]  # [num_edges, latent_dim]

        edge_input = torch.cat([edges, sender_nodes, receiver_nodes], dim=-1)
        edges_updated = self.edge_mlp(edge_input, noise_vector)  # [num_edges, hidden_dims[-1]]

        # 2. Aggregate edges for each receiver node
        # sum(e') for each receiver node
        edge_aggregation = scatter(
            edges_updated,
            receiver_idx,
            dim=0,
            dim_size=nodes.shape[0],
            reduce="sum"
        )  # [num_nodes, hidden_dims[-1]]

        # 3. Node update: v' = MLP([v, sum(e')])
        node_input = torch.cat([nodes, edge_aggregation], dim=-1)
        nodes_updated = self.node_mlp(node_input, noise_vector)  # [num_nodes, hidden_dims[-1]]

        # Apply conditional normalization if enabled
        if self.cond_norm is not None:
            nodes_updated = self.cond_norm(nodes_updated, noise_vector)

        return nodes_updated, edges_updated


class MLP(nn.Module):
    """Simple MLP module as used in other parts of the codebase."""

    def __init__(
            self,
            input_dim: int,
            hidden_dims: list,
            activation_layer: nn.Module = nn.ReLU,
            use_layer_norm: bool = False,
            bias: bool = True,
            activate_final: bool = False,
    ):
        """Initialize MLP module.

        Args:
            input_dim (int): input dimension
            hidden_dims (list): hidden layer dimensions
            activation_layer (nn.Module): activation function
            use_layer_norm (bool): whether to use layer normalization
            bias (bool): whether to use bias in linear layers
            activate_final (bool): whether to apply activation after final layer
            conditioning_dim (int, optional): dimension of conditioning parameter for conditional normalization
        """
        super().__init__()

        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, hidden_dims[0], bias=bias))
        for i in range(len(hidden_dims) - 1):
            self.linears.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=bias))

        self.activation = activation_layer()
        self.activate_final = activate_final

        self.norm_layer = None
        if use_layer_norm:
            self.norm_layer = nn.LayerNorm(hidden_dims[-1])

    def forward(self, x: torch.Tensor, cond_param: torch.Tensor | None = None) -> torch.Tensor:
        """Apply MLP to input."""
        # x = x.float()
        for linear in self.linears[:-1]:
            # x = linear(x)
            # x = self.activation(x)
            x = linear(x)
            x = checkpoint(self.activation, x, use_reentrant=False)

        x = self.linears[-1](x)

        if self.activate_final:
            x = self.activation(x)
            # x = checkpoint(self.activation, x, use_reentrant=False)

        if self.norm_layer is not None:
            if isinstance(self.norm_layer, ConditionalLayerNorm):
                if cond_param is None:
                    raise ValueError("cond_param must be provided when using conditional normalization")
                x = self.norm_layer(x, cond_param)
            else:
                x = self.norm_layer(x)

        return x


class Processor(nn.Module):
    """GenCast's processor with 16 unshared layers."""

    def __init__(
            self,
            hidden_dims: list,
            activation_layer: nn.Module = nn.ReLU,
            use_layer_norm: bool = True,
            num_layers: int = 16,
            noise_emb_dim: int | None = None,
    ):
        """Initialize the Processor.

        Args:
            hidden_dims (list[int]): hidden dimensions of internal MLPs.
            activation_layer (torch.nn.Module): activation function.
            use_layer_norm (bool): if true add LayerNorm after each MLP.
            num_layers (int): number of message-passing layers (default 16).
            noise_emb_dim (int, optional): dimension of noise embedding for conditioning. Defaults to None.
        """
        super().__init__()

        self.latent_dim = hidden_dims[-1]
        self.num_layers = num_layers
        self.noise_emb_dim = noise_emb_dim

        # Create unshared processor layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                ProcessorLayer(
                    latent_dim=self.latent_dim,
                    hidden_dims=hidden_dims,
                    activation_layer=activation_layer,
                    use_layer_norm=use_layer_norm,
                    noise_emb_dim=noise_emb_dim,
                )
            )

        self.transfer = nn.Linear(7, self.latent_dim)

    # def forward(
    #         self,
    #         mesh_nodes: torch.Tensor,
    #         mesh_edges_attr: torch.Tensor,
    #         mesh_edge_index: torch.Tensor,
    #         noise_vector: torch.Tensor | None = None,
    # ) -> torch.Tensor:
    #     """Forward pass through all processor layers.
    #
    #     Args:
    #         mesh_nodes (torch.Tensor): mesh nodes' latent features [num_mesh_nodes, latent_dim].
    #         mesh_edges_attr (torch.Tensor): mesh edges' latent features [num_mesh_edges, latent_dim].
    #         mesh_edge_index (torch.Tensor): mesh edge indices [2, num_mesh_edges].
    #         noise_vector (torch.Tensor, optional): noise vector for conditioning.
    #
    #     Returns:
    #         torch.Tensor: updated mesh nodes' latent features.
    #     """
    #     x = mesh_nodes
    #     e = mesh_edges_attr
    #
    #     # e = self.transfer(e)
    #     e = checkpoint(self.transfer, e, use_reentrant=False)
    #
    #     # Check noise_vector if conditioning is enabled
    #     if self.noise_emb_dim is not None and noise_vector is None:
    #         raise ValueError("noise_vector must be provided when noise_emb_dim is specified")
    #     if self.noise_emb_dim is None and noise_vector is not None:
    #         raise ValueError("noise_emb_dim must be specified in __init__ to use noise_vector")
    #
    #     for layer in self.layers:
    #         # Apply layer to get updates
    #         # nodes_update, edges_update = layer(x, e, mesh_edge_index)
    #         nodes_update, edges_update = checkpoint(layer, x, e, mesh_edge_index, noise_vector, use_reentrant=False)
    #
    #         # Add residual connections (论文公式13)
    #         x = x + nodes_update  # v ← v + v'
    #         e = e + edges_update  # e ← e + e'
    #
    #     return x

    def forward_chunk(self, x, e, edge_index, noise_vector, start, end):
        for layer in self.layers[start:end]:
            nodes_update, edges_update = layer(x, e, edge_index, noise_vector)
            x = x + nodes_update
            e = e + edges_update
        return x, e

    def forward(self, mesh_nodes, mesh_edges_attr, mesh_edge_index, noise_vector=None):
        x, e = mesh_nodes, mesh_edges_attr
        e = checkpoint(self.transfer, e, use_reentrant=False)

        chunk = 2
        for i in range(0, self.num_layers, chunk):
            x, e = checkpoint(
                self.forward_chunk, x, e, mesh_edge_index, noise_vector, i, i + chunk,
                use_reentrant=False
            )
        return x