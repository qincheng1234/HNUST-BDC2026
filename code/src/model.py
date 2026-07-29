"""Lightweight market-guided multi-scale mixer for cross-sectional ranking."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalMixBlock(nn.Module):
    """Mix trend and residual components without self-attention."""

    def __init__(self, sequence_length, d_model, time_hidden, expansion, dropout):
        super().__init__()
        self.time_norm = nn.LayerNorm(d_model)
        self.trend_mixer = nn.Sequential(
            nn.Linear(sequence_length, time_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(time_hidden, sequence_length),
        )
        self.residual_mixer = nn.Sequential(
            nn.Linear(sequence_length, time_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(time_hidden, sequence_length),
        )
        self.feature_norm = nn.LayerNorm(d_model)
        self.feature_mixer = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        normalized = self.time_norm(inputs).transpose(1, 2)
        trend = F.avg_pool1d(
            normalized,
            kernel_size=5,
            stride=1,
            padding=2,
            count_include_pad=False,
        )
        residual = normalized - trend
        temporal_update = self.trend_mixer(trend) + self.residual_mixer(residual)
        outputs = inputs + self.dropout(temporal_update.transpose(1, 2))
        return outputs + self.dropout(self.feature_mixer(self.feature_norm(outputs)))


class MultiScalePool(nn.Module):
    """Aggregate short, medium and long historical states."""

    def __init__(self, d_model, dropout):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, inputs):
        latest = inputs[:, -1]
        short = inputs[:, -5:].mean(dim=1)
        medium = inputs[:, -20:].mean(dim=1)
        long = inputs.mean(dim=1)
        return self.projection(torch.cat([latest, short, medium, long], dim=-1))


class MarketGuidance(nn.Module):
    """Condition each stock representation on the contemporaneous market state."""

    def __init__(self, d_model, dropout):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.shift = nn.Linear(d_model, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, stock_features):
        market_state = stock_features.mean(dim=1, keepdim=True)
        guided = stock_features * (1.0 + self.gate(market_state)) + self.shift(market_state)
        relative_state = stock_features - market_state
        return self.fusion(torch.cat([guided, relative_state], dim=-1))


class MarketGuidedMixer(nn.Module):
    """A compact ranking model with multi-scale temporal mixing and market gating."""

    model_type = "market_guided_mixer_v1"

    def __init__(self, input_dim, config, num_stocks):
        super().__init__()
        self.config = config
        self.num_stocks = num_stocks
        self.sequence_length = int(config["sequence_length"])
        d_model = int(config["d_model"])
        mixer_layers = int(config.get("mixer_layers", config.get("num_layers", 2)))
        time_hidden = int(config.get("time_mixer_hidden", max(16, self.sequence_length // 2)))
        expansion = int(config.get("mixer_expansion", 2))
        dropout = float(config["dropout"])

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.temporal_blocks = nn.ModuleList(
            [
                TemporalMixBlock(
                    sequence_length=self.sequence_length,
                    d_model=d_model,
                    time_hidden=time_hidden,
                    expansion=expansion,
                    dropout=dropout,
                )
                for _ in range(mixer_layers)
            ]
        )
        self.temporal_pool = MultiScalePool(d_model, dropout)
        self.market_guidance = MarketGuidance(d_model, dropout)
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs):
        batch_size, stock_count, sequence_length, feature_count = inputs.shape
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, received {sequence_length}"
            )

        temporal_inputs = inputs.reshape(batch_size * stock_count, sequence_length, feature_count)
        temporal_features = self.input_projection(temporal_inputs)
        for block in self.temporal_blocks:
            temporal_features = block(temporal_features)

        stock_features = self.temporal_pool(temporal_features).reshape(batch_size, stock_count, -1)
        guided_features = self.market_guidance(stock_features)
        return self.score_head(guided_features).squeeze(-1)
