"""Trading-session-aware splits for leak-free forward-return validation."""

import pandas as pd


def trading_dates_from_frame(frame):
    return pd.DatetimeIndex(sorted(pd.to_datetime(frame["日期"]).dropna().unique()))


def build_walk_forward_folds(
    dates,
    label_horizon,
    embargo_days,
    validation_days,
    min_train_days,
    cv_folds,
    warmup_days,
):
    """Build expanding-window folds whose train labels end before validation."""
    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
    if label_horizon <= 0 or embargo_days < label_horizon:
        raise ValueError("embargo_days must be at least label_horizon")
    if validation_days <= 0 or min_train_days <= 0 or cv_folds <= 0:
        raise ValueError("validation_days, min_train_days, and cv_folds must be positive")

    latest_validation_end = len(sessions) - label_horizon - 1
    first_validation_end = min_train_days + embargo_days + validation_days - 1
    candidate_ends = list(range(latest_validation_end, first_validation_end - 1, -validation_days))
    selected_ends = list(reversed(candidate_ends[:cv_folds]))
    if len(selected_ends) < cv_folds:
        raise ValueError("Not enough trading sessions for the configured walk-forward folds")

    folds = []
    for fold_number, validation_end_index in enumerate(selected_ends, start=1):
        validation_start_index = validation_end_index - validation_days + 1
        train_sample_end_index = validation_start_index - embargo_days - 1
        train_data_end_index = train_sample_end_index + label_horizon
        context_start_index = max(0, validation_start_index - warmup_days)
        validation_data_end_index = validation_end_index + label_horizon

        if train_sample_end_index < min_train_days - 1:
            raise ValueError("Fold does not contain the configured minimum training history")
        if train_data_end_index >= validation_start_index:
            raise AssertionError("Training labels overlap the validation decision period")

        folds.append(
            {
                "name": f"fold_{fold_number:02d}",
                "train_sample_end": sessions[train_sample_end_index],
                "train_data_end": sessions[train_data_end_index],
                "validation_context_start": sessions[context_start_index],
                "validation_start": sessions[validation_start_index],
                "validation_end": sessions[validation_end_index],
                "validation_data_end": sessions[validation_data_end_index],
            }
        )
    return folds


def select_frame_period(frame, start_date=None, end_date=None):
    selected = frame
    if start_date is not None:
        selected = selected[selected["日期"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        selected = selected[selected["日期"] <= pd.Timestamp(end_date)]
    return selected.copy()
