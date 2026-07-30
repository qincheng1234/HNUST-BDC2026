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


def _causal_moving_average(inputs, window):
    """Compute a trailing mean using only the current and preceding steps."""
    if window < 1:
        raise ValueError("Causal moving-average window must be positive")
    if inputs.ndim != 3:
        raise ValueError("Expected [batch, time, features] inputs")

    values = inputs.transpose(1, 2)
    padded_values = F.pad(values, (window - 1, 0))
    numerator = F.avg_pool1d(padded_values, kernel_size=window, stride=1) * window

    valid_steps = inputs.new_ones((inputs.size(0), 1, inputs.size(1)))
    padded_steps = F.pad(valid_steps, (window - 1, 0))
    denominator = F.avg_pool1d(padded_steps, kernel_size=window, stride=1) * window
    return (numerator / denominator.clamp_min(1.0)).transpose(1, 2)


class CausalMultiScaleTemporalBlock(nn.Module):
    """Mix causal high-frequency, medium-frequency, and trend components."""

    def __init__(
        self,
        sequence_length,
        d_model,
        time_hidden,
        expansion,
        dropout,
        short_window,
        long_window,
    ):
        super().__init__()
        if not 1 <= short_window < long_window <= sequence_length:
            raise ValueError("Expected 1 <= short_window < long_window <= sequence_length")
        self.sequence_length = sequence_length
        self.short_window = short_window
        self.long_window = long_window
        self.time_norm = nn.LayerNorm(d_model)
        self.component_mixers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(sequence_length, time_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(time_hidden, sequence_length),
                )
                for _ in range(3)
            ]
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
        if inputs.size(1) != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, received {inputs.size(1)}"
            )
        normalized = self.time_norm(inputs)
        short_trend = _causal_moving_average(normalized, self.short_window)
        long_trend = _causal_moving_average(normalized, self.long_window)
        components = (
            normalized - short_trend,
            short_trend - long_trend,
            long_trend,
        )
        temporal_update = sum(
            mixer(component.transpose(1, 2)).transpose(1, 2)
            for mixer, component in zip(self.component_mixers, components)
        )
        outputs = inputs + self.dropout(temporal_update)
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


class CausalMarketTokenPool(nn.Module):
    """Create a small set of trailing market-state tokens at fixed horizons."""

    def __init__(self, d_model, windows, dropout):
        super().__init__()
        if not windows or any(window < 1 for window in windows):
            raise ValueError("Market token windows must be a non-empty positive sequence")
        self.windows = tuple(int(window) for window in windows)
        self.token_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(d_model),
                )
                for _ in self.windows
            ]
        )

    def forward(self, market_sequence):
        tokens = []
        sequence_length = market_sequence.size(1)
        for window, projection in zip(self.windows, self.token_projections):
            state = market_sequence[:, -min(window, sequence_length):].mean(dim=1)
            tokens.append(projection(state))
        return torch.stack(tokens, dim=1)


class MarketTokenGate(nn.Module):
    """Let every stock learn a data-dependent mixture over market-state tokens."""

    def __init__(self, d_model, dropout):
        super().__init__()
        self.stock_query = nn.Linear(d_model, d_model, bias=False)
        self.token_key = nn.Linear(d_model, d_model, bias=False)
        self.token_value = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, stock_features, market_tokens, mask=None):
        scale = math.sqrt(stock_features.size(-1))
        queries = F.normalize(self.stock_query(stock_features), dim=-1)
        keys = F.normalize(self.token_key(market_tokens), dim=-1)
        attention = torch.softmax(torch.einsum("bnd,bkd->bnk", queries, keys) / scale, dim=-1)
        market_context = torch.einsum(
            "bnk,bkd->bnd",
            attention,
            self.token_value(market_tokens),
        )
        gates = self.gate(torch.cat([stock_features, market_context], dim=-1))
        guided_features = self.output_norm(stock_features + gates * market_context)
        if mask is not None:
            valid = mask.unsqueeze(-1).to(guided_features.dtype)
            guided_features = guided_features * valid
            market_context = market_context * valid
        return guided_features, market_context


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

    def _prepare_stock_inputs(self, inputs, mask):
        return inputs

    def _encode_stock_features(self, inputs, mask):
        batch_size, stock_count, sequence_length, feature_count = inputs.shape
        stock_inputs = self._prepare_stock_inputs(inputs, mask)
        temporal_inputs = stock_inputs.reshape(batch_size * stock_count, sequence_length, feature_count)
        stock_temporal = self.input_projection(temporal_inputs)
        for block in self.temporal_blocks:
            stock_temporal = block(stock_temporal)
        return self.temporal_pool(stock_temporal).reshape(batch_size, stock_count, -1)

    def forward(self, inputs, mask=None):
        batch_size, stock_count, sequence_length, feature_count = inputs.shape
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, received {sequence_length}"
            )
        if mask is not None and mask.shape != (batch_size, stock_count):
            raise ValueError("Cross-sectional mask shape must match [batch, stock_count]")

        stock_features = self._encode_stock_features(inputs, mask)

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

        return self._build_outputs(fused_features)

    def _build_outputs(self, fused_features):
        return {
            "ranking_score": self.score_head(fused_features).squeeze(-1),
            "horizon_return": self.horizon_head(fused_features),
            "downside_risk": F.softplus(self.downside_head(fused_features).squeeze(-1)),
        }


class CrossSectionalResidualFactorMixer(CausalCrossSectionalFactorMixer):
    """Separate market-common inputs from standardized idiosyncratic paths."""

    model_type = "causal_factor_mixer_v2"

    def __init__(self, input_dim, config, num_stocks):
        super().__init__(input_dim=input_dim, config=config, num_stocks=num_stocks)
        self.residual_epsilon = float(config.get("cross_sectional_epsilon", 1e-6))

    def _prepare_stock_inputs(self, inputs, mask):
        market_mean = _masked_mean(inputs, mask, dim=1).unsqueeze(1)
        centered = inputs - market_mean
        market_variance = _masked_mean(centered.square(), mask, dim=1).unsqueeze(1)
        standardized = centered / torch.sqrt(market_variance + self.residual_epsilon)
        if mask is not None:
            standardized = standardized * mask.unsqueeze(-1).unsqueeze(-1).to(standardized.dtype)
        return standardized


class CausalMultiscaleMarketFactorMixer(nn.Module):
    """Residual stock paths with causal scales, market tokens, and factor mixing.

    The architecture has no stock-code embeddings or external pretrained
    weights. Its cross-sectional interactions remain linear in stock count
    apart from the fixed, small number of market and factor tokens.
    """

    model_type = "causal_multiscale_market_mixer_v1"
    supports_cross_sectional_mask = True

    def __init__(self, input_dim, config, num_stocks):
        super().__init__()
        self.num_stocks = num_stocks
        self.sequence_length = int(config["sequence_length"])
        self.residual_epsilon = float(config.get("cross_sectional_epsilon", 1e-6))
        d_model = int(config["d_model"])
        mixer_layers = int(config.get("mixer_layers", 2))
        time_hidden = int(config.get("time_mixer_hidden", max(16, self.sequence_length // 2)))
        expansion = int(config.get("mixer_expansion", 2))
        dropout = float(config["dropout"])
        short_window = int(config.get("stock_short_window", 5))
        long_window = int(config.get("stock_long_window", 20))
        factor_count = int(config.get("factor_count", 8))
        market_layers = int(config.get("market_mixer_layers", 1))
        market_windows = tuple(config.get("market_token_windows", (1, 5, 20)))

        self.stock_projection = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model))
        self.stock_temporal_blocks = nn.ModuleList(
            [
                CausalMultiScaleTemporalBlock(
                    self.sequence_length,
                    d_model,
                    time_hidden,
                    expansion,
                    dropout,
                    short_window,
                    long_window,
                )
                for _ in range(mixer_layers)
            ]
        )
        self.stock_pool = MultiScalePool(d_model, dropout)
        self.market_projection = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model))
        self.market_temporal_blocks = nn.ModuleList(
            [
                CausalMultiScaleTemporalBlock(
                    self.sequence_length,
                    d_model,
                    time_hidden,
                    expansion,
                    dropout,
                    short_window,
                    long_window,
                )
                for _ in range(market_layers)
            ]
        )
        self.market_tokens = CausalMarketTokenPool(d_model, market_windows, dropout)
        self.market_gate = MarketTokenGate(d_model, dropout)
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
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _prepare_stock_inputs(self, inputs, mask):
        market_mean = _masked_mean(inputs, mask, dim=1).unsqueeze(1)
        centered = inputs - market_mean
        market_variance = _masked_mean(centered.square(), mask, dim=1).unsqueeze(1)
        standardized = centered / torch.sqrt(market_variance + self.residual_epsilon)
        if mask is not None:
            standardized = standardized * mask.unsqueeze(-1).unsqueeze(-1).to(standardized.dtype)
        return standardized

    def _encode_stock_features(self, inputs, mask):
        batch_size, stock_count, sequence_length, feature_count = inputs.shape
        residual_inputs = self._prepare_stock_inputs(inputs, mask)
        temporal = residual_inputs.reshape(batch_size * stock_count, sequence_length, feature_count)
        temporal = self.stock_projection(temporal)
        for block in self.stock_temporal_blocks:
            temporal = block(temporal)
        return self.stock_pool(temporal).reshape(batch_size, stock_count, -1)

    def _encode_market_tokens(self, inputs, mask):
        market_sequence = _masked_mean(inputs, mask, dim=1)
        temporal = self.market_projection(market_sequence)
        for block in self.market_temporal_blocks:
            temporal = block(temporal)
        return self.market_tokens(temporal)

    def forward(self, inputs, mask=None):
        batch_size, stock_count, sequence_length, _ = inputs.shape
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, received {sequence_length}"
            )
        if mask is not None and mask.shape != (batch_size, stock_count):
            raise ValueError("Cross-sectional mask shape must match [batch, stock_count]")

        stock_features = self._encode_stock_features(inputs, mask)
        market_tokens = self._encode_market_tokens(inputs, mask)
        market_guided, market_context = self.market_gate(stock_features, market_tokens, mask)
        factor_context, _ = self.factor_mixer(market_guided, mask=mask)
        fusion_inputs = torch.cat([stock_features, market_context, factor_context], dim=-1)
        fusion_weights = torch.softmax(self.fusion_gate(fusion_inputs), dim=-1)
        fused_features = (
            fusion_weights[..., 0:1] * stock_features
            + fusion_weights[..., 1:2] * market_context
            + fusion_weights[..., 2:3] * factor_context
        )
        fused_features = self.fusion_norm(fused_features)
        if mask is not None:
            fused_features = fused_features * mask.unsqueeze(-1).to(fused_features.dtype)
        return {"ranking_score": self.score_head(fused_features).squeeze(-1)}


def build_model(input_dim, config, num_stocks):
    """Construct exactly one configured model for training or inference."""
    model_type = config["model_type"]
    if model_type == MarketGuidedMixer.model_type:
        return MarketGuidedMixer(input_dim=input_dim, config=config, num_stocks=num_stocks)
    if model_type == CausalCrossSectionalFactorMixer.model_type:
        return CausalCrossSectionalFactorMixer(input_dim=input_dim, config=config, num_stocks=num_stocks)
    if model_type == CrossSectionalResidualFactorMixer.model_type:
        return CrossSectionalResidualFactorMixer(input_dim=input_dim, config=config, num_stocks=num_stocks)
    if model_type == CausalMultiscaleMarketFactorMixer.model_type:
        return CausalMultiscaleMarketFactorMixer(
            input_dim=input_dim,
            config=config,
            num_stocks=num_stocks,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")
