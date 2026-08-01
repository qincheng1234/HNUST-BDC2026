"""Lightweight cross-sectional residual factor mixer for CSI 300 stock ranking."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# shared building blocks
# ---------------------------------------------------------------------------

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
            normalized, kernel_size=5, stride=1, padding=2, count_include_pad=False,
        )
        residual = normalized - trend
        temporal_update = self.trend_mixer(trend) + self.residual_mixer(residual)
        outputs = inputs + self.dropout(temporal_update.transpose(1, 2))
        return outputs + self.dropout(self.feature_mixer(self.feature_norm(outputs)))


class LearnableTemporalPool(nn.Module):
    """Learn position-aware attention weights over the temporal dimension.

    Replaces the fixed-window MultiScalePool so the model can shift its
    temporal focus when market regimes change (e.g. giving more weight to
    the most recent 5-10 days during a style rotation).
    """

    def __init__(self, sequence_length, d_model, dropout):
        super().__init__()
        self.position_bias = nn.Parameter(torch.zeros(sequence_length))
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, inputs):
        # inputs: [batch, seq_len, d_model]
        scale = math.sqrt(inputs.size(-1))
        scores = torch.einsum("bld,d->bl", inputs, self.query) / scale
        scores = scores + self.position_bias
        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bld,bl->bd", inputs, weights)
        return self.projection(pooled)


def _masked_mean(inputs, mask, dim):
    """Mean that ignores padded cross-sectional members."""
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
            induction_logits = induction_logits.masked_fill(
                ~mask.bool().unsqueeze(-1), -1e9,
            )
        induction_weights = torch.softmax(induction_logits, dim=1)
        factor_values = torch.einsum(
            "bnk,bnd->bkd", induction_weights, self.stock_value(stock_features),
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


# ---------------------------------------------------------------------------
# main model
# ---------------------------------------------------------------------------

class CrossSectionalResidualFactorMixer(nn.Module):
    """Causal ranking model — cross-sectional residual paths + dynamic factor
    interaction + market-state fusion.

    Replaces the earlier CausalCrossSectionalFactorMixer lineage.  Only the
    ranking head is retained; auxiliary return / downside heads were removed
    after OOF falsification (v1 / v3).
    """

    model_type = "causal_factor_mixer_v2"
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
        self.residual_epsilon = float(config.get("cross_sectional_epsilon", 1e-6))

        # ---- stock path ---------------------------------------------------
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model),
        )
        self.temporal_blocks = nn.ModuleList([
            TemporalMixBlock(self.sequence_length, d_model, time_hidden, expansion, dropout)
            for _ in range(mixer_layers)
        ])
        self.temporal_pool = LearnableTemporalPool(self.sequence_length, d_model, dropout)

        # ---- market path --------------------------------------------------
        self.market_projection = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model),
        )
        self.market_temporal_blocks = nn.ModuleList([
            TemporalMixBlock(self.sequence_length, d_model, time_hidden, expansion, dropout)
            for _ in range(market_layers)
        ])
        self.market_pool = LearnableTemporalPool(self.sequence_length, d_model, dropout)

        # ---- cross-stock interaction --------------------------------------
        self.factor_mixer = DynamicFactorMixer(d_model, factor_count, dropout)

        # ---- fusion -------------------------------------------------------
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3),
        )
        self.fusion_norm = nn.LayerNorm(d_model)

        # ---- ranking head (only head kept after v1/v3 falsification) ------
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # weight init
    # ------------------------------------------------------------------

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # cross-sectional residual standardisation  (the "v2" part)
    # ------------------------------------------------------------------

    def _prepare_stock_inputs(self, inputs, mask):
        market_mean = _masked_mean(inputs, mask, dim=1).unsqueeze(1)
        centered = inputs - market_mean
        market_variance = _masked_mean(centered.square(), mask, dim=1).unsqueeze(1)
        standardized = centered / torch.sqrt(market_variance + self.residual_epsilon)
        if mask is not None:
            standardized = standardized * mask.unsqueeze(-1).unsqueeze(-1).to(
                standardized.dtype,
            )
        return standardized

    # ------------------------------------------------------------------
    # temporal encoding
    # ------------------------------------------------------------------

    def _encode_stock_features(self, inputs, mask):
        batch_size, stock_count, sequence_length, feature_count = inputs.shape
        stock_inputs = self._prepare_stock_inputs(inputs, mask)
        temporal_inputs = stock_inputs.reshape(
            batch_size * stock_count, sequence_length, feature_count,
        )
        stock_temporal = self.input_projection(temporal_inputs)
        for block in self.temporal_blocks:
            stock_temporal = block(stock_temporal)
        return self.temporal_pool(stock_temporal).reshape(batch_size, stock_count, -1)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, inputs, mask=None):
        batch_size, stock_count, sequence_length, feature_count = inputs.shape
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, "
                f"received {sequence_length}",
            )
        if mask is not None and mask.shape != (batch_size, stock_count):
            raise ValueError(
                "Cross-sectional mask shape must match [batch, stock_count]",
            )

        # stock path
        stock_features = self._encode_stock_features(inputs, mask)

        # market path
        market_sequence = _masked_mean(inputs, mask, dim=1)
        market_temporal = self.market_projection(market_sequence)
        for block in self.market_temporal_blocks:
            market_temporal = block(market_temporal)
        market_state = self.market_pool(market_temporal).unsqueeze(1).expand(
            -1, stock_count, -1,
        )

        # factor interaction + gated fusion
        factor_context, _ = self.factor_mixer(stock_features, mask=mask)
        fusion_inputs = torch.cat([stock_features, factor_context, market_state], dim=-1)
        fusion_weights = torch.softmax(self.fusion_gate(fusion_inputs), dim=-1)
        fused = (
            fusion_weights[..., 0:1] * stock_features
            + fusion_weights[..., 1:2] * factor_context
            + fusion_weights[..., 2:3] * market_state
        )
        fused = self.fusion_norm(fused)
        if mask is not None:
            fused = fused * mask.unsqueeze(-1).to(fused.dtype)

        return self.score_head(fused).squeeze(-1)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

def build_model(input_dim, config, num_stocks):
    """Construct a model for training or inference."""
    model_type = config["model_type"]
    if model_type == CrossSectionalResidualFactorMixer.model_type:
        return CrossSectionalResidualFactorMixer(
            input_dim=input_dim, config=config, num_stocks=num_stocks,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")
