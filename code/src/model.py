"""Lightweight market-guided multi-scale mixer for cross-sectional ranking."""

import math

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


def _masked_mean(inputs, mask, dim):
    """Return a mean that ignores padded cross-sectional members."""
    if mask is None:
        return inputs.mean(dim=dim)
    weights = mask.to(dtype=inputs.dtype)
    while weights.ndim < inputs.ndim:
        weights = weights.unsqueeze(-1)
    denominator = weights.sum(dim=dim).clamp_min(1.0)
    return (inputs * weights).sum(dim=dim) / denominator


class DynamicFactorMixer(nn.Module):
    """Induce a small set of market factors in linear cross-sectional time."""

    def __init__(self, d_model, factor_count, dropout):
        super().__init__()
        self.factor_queries = nn.Parameter(torch.empty(factor_count, d_model))
        self.stock_key = nn.Linear(d_model, d_model, bias=False)
        self.stock_value = nn.Linear(d_model, d_model, bias=False)
        self.stock_query = nn.Linear(d_model, d_model, bias=False)
        self.factor_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.factor_queries, mean=0.0, std=d_model ** -0.5)

    def forward(self, stock_features, mask=None):
        stock_keys = F.normalize(self.stock_key(stock_features), dim=-1)
        factor_queries = F.normalize(self.factor_queries, dim=-1)
        induction_logits = torch.einsum("bnd,kd->bnk", stock_keys, factor_queries)
        induction_logits = induction_logits / math.sqrt(stock_features.size(-1))
        if mask is not None:
            induction_logits = induction_logits.masked_fill(~mask.bool().unsqueeze(-1), -1e9)
        induction_weights = torch.softmax(induction_logits, dim=1)
        factor_values = torch.einsum(
            "bnk,bnd->bkd",
            induction_weights,
            self.stock_value(stock_features),
        )
        factor_states = self.factor_norm(factor_values + self.factor_queries.unsqueeze(0))

        stock_queries = F.normalize(self.stock_query(stock_features), dim=-1)
        membership_logits = torch.einsum("bnd,bkd->bnk", stock_queries, factor_states)
        membership_logits = membership_logits / math.sqrt(stock_features.size(-1))
        factor_membership = torch.softmax(membership_logits, dim=-1)
        factor_context = torch.einsum("bnk,bkd->bnd", factor_membership, factor_states)
        factor_context = self.context_norm(stock_features + self.dropout(factor_context))
        if mask is not None:
            factor_context = factor_context * mask.unsqueeze(-1).to(factor_context.dtype)
        return factor_context, factor_states


class CausalCrossSectionalFactorMixer(nn.Module):
    """Causal temporal mixer with dynamic market factors and learned task heads."""

    model_type = "causal_factor_mixer_v1"
    supports_cross_sectional_mask = True

    def __init__(self, input_dim, config, num_stocks):
        super().__init__()
        self.num_stocks = num_stocks
        self.sequence_length = int(config["sequence_length"])
        d_model = int(config["d_model"])
        mixer_layers = int(config.get("mixer_layers", 2))
        time_hidden = int(config.get("time_mixer_hidden", max(16, self.sequence_length // 2)))
        expansion = int(config.get("mixer_expansion", 2))
        dropout = float(config["dropout"])
        factor_count = int(config.get("factor_count", 8))
        market_layers = int(config.get("market_mixer_layers", 1))

        self.input_projection = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model))
        self.temporal_blocks = nn.ModuleList(
            [
                TemporalMixBlock(self.sequence_length, d_model, time_hidden, expansion, dropout)
                for _ in range(mixer_layers)
            ]
        )
        self.temporal_pool = MultiScalePool(d_model, dropout)
        self.market_projection = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model))
        self.market_temporal_blocks = nn.ModuleList(
            [
                TemporalMixBlock(self.sequence_length, d_model, time_hidden, expansion, dropout)
                for _ in range(market_layers)
            ]
        )
        self.market_pool = MultiScalePool(d_model, dropout)
        self.factor_mixer = DynamicFactorMixer(d_model, factor_count, dropout)
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        self.fusion_norm = nn.LayerNorm(d_model)
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.horizon_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
        )
        self.downside_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs, mask=None):
        batch_size, stock_count, sequence_length, feature_count = inputs.shape
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, received {sequence_length}"
            )
        if mask is not None and mask.shape != (batch_size, stock_count):
            raise ValueError("Cross-sectional mask shape must match [batch, stock_count]")

        temporal_inputs = inputs.reshape(batch_size * stock_count, sequence_length, feature_count)
        stock_temporal = self.input_projection(temporal_inputs)
        for block in self.temporal_blocks:
            stock_temporal = block(stock_temporal)
        stock_features = self.temporal_pool(stock_temporal).reshape(batch_size, stock_count, -1)

        market_sequence = _masked_mean(inputs, mask, dim=1)
        market_temporal = self.market_projection(market_sequence)
        for block in self.market_temporal_blocks:
            market_temporal = block(market_temporal)
        market_state = self.market_pool(market_temporal).unsqueeze(1).expand(-1, stock_count, -1)

        factor_context, _ = self.factor_mixer(stock_features, mask=mask)
        fusion_inputs = torch.cat([stock_features, factor_context, market_state], dim=-1)
        fusion_weights = torch.softmax(self.fusion_gate(fusion_inputs), dim=-1)
        fused_features = (
            fusion_weights[..., 0:1] * stock_features
            + fusion_weights[..., 1:2] * factor_context
            + fusion_weights[..., 2:3] * market_state
        )
        fused_features = self.fusion_norm(fused_features)
        if mask is not None:
            fused_features = fused_features * mask.unsqueeze(-1).to(fused_features.dtype)

        return {
            "ranking_score": self.score_head(fused_features).squeeze(-1),
            "horizon_return": self.horizon_head(fused_features),
            "downside_risk": F.softplus(self.downside_head(fused_features).squeeze(-1)),
        }


def build_model(input_dim, config, num_stocks):
    """Construct exactly one configured model for training or inference."""
    model_type = config["model_type"]
    if model_type == MarketGuidedMixer.model_type:
        return MarketGuidedMixer(input_dim=input_dim, config=config, num_stocks=num_stocks)
    if model_type == CausalCrossSectionalFactorMixer.model_type:
        return CausalCrossSectionalFactorMixer(input_dim=input_dim, config=config, num_stocks=num_stocks)
    raise ValueError(f"Unsupported model_type: {model_type}")
