import torch
import torch.nn as nn
import math

try:
    from mamba_ssm import Mamba2
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not installed. Install with: pip install mamba-ssm")


class PoseMamba(nn.Module):
    """Mamba-based pose estimation network.

    Replaces Transformer with Mamba (Selective State Space Model) for
    efficient sequence modeling with linear complexity.
    """

    def __init__(
        self,
        input_dim=768,
        embedding_dim=128,
        num_layers=2,
        d_state=64,
        d_conv=4,
        expand=2,
        dropout=0.1
    ):
        """Initialize PoseMamba.

        Args:
            input_dim: Input feature dimension (768 for visual+inertial)
            embedding_dim: Hidden dimension for Mamba blocks
            num_layers: Number of Mamba layers to stack
            d_state: SSM state dimension (N in paper)
            d_conv: Local convolution width
            expand: Expansion factor for MLP in Mamba block
            dropout: Dropout probability
        """
        super(PoseMamba, self).__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm package not found. Install with: pip install mamba-ssm")

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Input projection
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, self.embedding_dim),
        )

        # Stack of Mamba blocks
        self.mamba_layers = nn.ModuleList([
            Mamba2(
                d_model=self.embedding_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(num_layers)
        ])

        # Layer norms (applied before each Mamba block)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.embedding_dim)
            for _ in range(num_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output head
        self.fc2 = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.embedding_dim, 6)
        )

    def positional_embedding(self, seq_length, device):
        """Generate sinusoidal positional embeddings."""
        pos = torch.arange(0, seq_length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, device=device).float()
            * -(math.log(10000.0) / self.embedding_dim)
        )
        pos_embedding = torch.zeros(seq_length, self.embedding_dim, device=device)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        return pos_embedding.unsqueeze(0)  # (1, seq_len, embedding_dim)

    def forward(self, batch, gt):
        """Forward pass through Mamba layers.

        Args:
            batch: Tuple of (visual_inertial_features, rot, weight)
            gt: Ground truth poses (not used in forward, kept for compatibility)

        Returns:
            Output poses: (batch_size, seq_len, 6)
        """
        visual_inertial_features, _, _ = batch
        batch_size, seq_length, _ = visual_inertial_features.shape

        # Project input to embedding dimension
        x = self.fc1(visual_inertial_features)  # (B, L, embedding_dim)

        # Add positional encoding
        pos_embedding = self.positional_embedding(seq_length, x.device)
        x = x + pos_embedding

        # Pass through Mamba layers with residual connections
        for layer_norm, mamba_layer in zip(self.layer_norms, self.mamba_layers):
            # Pre-norm architecture (like in Mamba paper)
            residual = x
            x = layer_norm(x)
            x = mamba_layer(x) + residual  # Residual connection
            x = self.dropout(x)

        # Output projection to 6-DOF pose
        output = self.fc2(x)  # (B, L, 6)

        return output


class PoseMambaVisual(nn.Module):
    """Mamba model using only visual features (first 512 dims)."""

    def __init__(
        self,
        input_dim=512,
        embedding_dim=128,
        num_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1
    ):
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm package not found.")

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.fc1 = nn.Linear(input_dim, embedding_dim)

        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=embedding_dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(embedding_dim, 6)
        )

    def positional_embedding(self, seq_length, device):
        pos = torch.arange(0, seq_length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, device=device).float()
            * -(math.log(10000.0) / self.embedding_dim)
        )
        pos_embedding = torch.zeros(seq_length, self.embedding_dim, device=device)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        return pos_embedding.unsqueeze(0)

    def forward(self, batch, gt):
        visual_inertial_features, _, _ = batch
        visual_features = visual_inertial_features[:, :, :512]  # Only visual

        x = self.fc1(visual_features)
        pos_embedding = self.positional_embedding(x.shape[1], x.device)
        x = x + pos_embedding

        for layer_norm, mamba_layer in zip(self.layer_norms, self.mamba_layers):
            residual = x
            x = layer_norm(x)
            x = mamba_layer(x) + residual
            x = self.dropout(x)

        output = self.fc2(x)
        return output


class PoseMambaInertial(nn.Module):
    """Mamba model using only inertial features (last 256 dims)."""

    def __init__(
        self,
        input_dim=256,
        embedding_dim=128,
        num_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1
    ):
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm package not found.")

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.fc1 = nn.Linear(input_dim, embedding_dim)

        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=embedding_dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(num_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(embedding_dim, 6)
        )

    def positional_embedding(self, seq_length, device):
        pos = torch.arange(0, seq_length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, device=device).float()
            * -(math.log(10000.0) / self.embedding_dim)
        )
        pos_embedding = torch.zeros(seq_length, self.embedding_dim, device=device)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        return pos_embedding.unsqueeze(0)

    def forward(self, batch, gt):
        visual_inertial_features, _, _ = batch
        inertial_features = visual_inertial_features[:, :, 512:]  # Only inertial

        x = self.fc1(inertial_features)
        pos_embedding = self.positional_embedding(x.shape[1], x.device)
        x = x + pos_embedding

        for layer_norm, mamba_layer in zip(self.layer_norms, self.mamba_layers):
            residual = x
            x = layer_norm(x)
            x = mamba_layer(x) + residual
            x = self.dropout(x)

        output = self.fc2(x)
        return output


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    seq_len = 11
    input_dim = 768

    # Create dummy input
    dummy_features = torch.randn(batch_size, seq_len, input_dim)
    dummy_rot = torch.randn(batch_size, seq_len)
    dummy_weight = torch.randn(batch_size)
    dummy_gt = torch.randn(batch_size, seq_len, 6)

    batch = (dummy_features, dummy_rot, dummy_weight)

    # Create model
    model = PoseMamba(
        input_dim=768,
        embedding_dim=128,
        num_layers=2,
        d_state=16,
        d_conv=4,
        expand=2
    )

    # Forward pass
    output = model(batch, dummy_gt)
    print(f"Input shape: {dummy_features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✓ PoseMamba model working correctly!")
