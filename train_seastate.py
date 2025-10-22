#!/usr/bin/env python3
"""Train sea state forecasting models from the former research notebook.

This script reproduces the main parts of the notebook
``(336,12)_M6_Best_Univariate.ipynb`` in an executable and
reproducible way.  It focuses on the univariate wave-height forecast
pipeline (336 hours of context to predict the next 12 hours) and
implements the three model families that were evaluated in the report:
LSTM, Temporal Convolutional Network (TCN) and XGBoost.

The script can be invoked from the command line and offers a handful of
flags to tweak the experiment.  Data can either be downloaded from the
Marine Institute ERDDAP endpoint (default) or provided locally.

Example:
    python train_seastate.py --models lstm tcn xgboost \
        --dataset m6_buoy --context 336 --horizon 12
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


_LOGGER = logging.getLogger(__name__)
DEFAULT_DATA_DIR = Path(__file__).parent / "data"
MARINE_INSTITUTE_URL = (
    "https://erddap.marine.ie/erddap/tabledap/iwbnetwork.csv?"
    "station_id%2CCallSign%2Clongitude%2Clatitude%2Ctime%2C"
    "AtmosphericPressure%2CWindDirection%2CWindSpeed%2CGust%2C"
    "WaveHeight%2CWavePeriod%2CMeanWaveDirection%2CHmax%2CAirTemperature%2C"
    "DewPoint%2CSeaTemperature%2CSalinity%2CRelativeHumidity%2CSprTP%2C"
    "ThTp%2CTp%2CQC_Flag&station_id=%22M6%22"
)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.set_index("time")
    elif isinstance(df.index, pd.DatetimeIndex):
        pass
    else:
        raise ValueError("DataFrame must contain a 'time' column or already be indexed by time.")
    df = df.sort_index()
    return df


def load_dataset(dataset: str, data_path: Path | None) -> pd.DataFrame:
    """Load the requested dataset.

    Parameters
    ----------
    dataset:
        Either ``"m6_buoy"`` or ``"weather"``.
    data_path:
        Optional path to a local file.  When omitted the original notebook
        remote source is used.  ``.json`` files are expected to contain a
        list of records, whereas ``.csv`` files are passed directly to
        :func:`pandas.read_csv`.
    """
    if data_path is None:
        if dataset == "m6_buoy":
            _LOGGER.info("Downloading M6 buoy dataset from Marine Institute ERDDAP.")
            df = pd.read_csv(MARINE_INSTITUTE_URL)
        elif dataset == "weather":
            default_csv = DEFAULT_DATA_DIR / "weather.csv"
            _LOGGER.info("Loading weather dataset from %s", default_csv)
            df = pd.read_csv(default_csv)
        else:
            raise ValueError(f"Unsupported dataset '{dataset}'.")
    else:
        _LOGGER.info("Loading dataset from %s", data_path)
        if data_path.suffix.lower() == ".json":
            with data_path.open() as fh:
                records = json.load(fh)
            df = pd.DataFrame.from_records(records)
        else:
            df = pd.read_csv(data_path)

    if dataset == "weather":
        if "date" in df.columns and "time" not in df.columns:
            df.rename(columns={"date": "time"}, inplace=True)
    df = _ensure_datetime_index(df)

    # Coerce every non-numeric column except the datetime index to numeric values.
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.astype(float)

    # Interpolate using time so that occasional gaps are filled.
    df = df.interpolate(method="time").ffill().bfill()
    return df


@dataclass
class WindowedDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    target_shape: Tuple[int, int]


def extract_windows(
    df: pd.DataFrame,
    target_column: str,
    context: int,
    horizon: int,
    step: int,
    selected_columns: Sequence[str] | None = None,
    max_windows: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """Create sliding windows without leaking future information."""
    if selected_columns is None:
        feature_columns = [target_column]
    else:
        feature_columns = list(dict.fromkeys([target_column, *selected_columns]))
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Columns {missing!r} were not found in the dataset.")

    df = df[feature_columns]
    total_length = context + horizon
    windows: List[np.ndarray] = []
    starts: List[pd.Timestamp] = []

    values = df.to_numpy()
    index = df.index
    for start in range(0, len(df) - total_length + 1, step):
        window = values[start : start + total_length]
        if np.isnan(window).any():
            continue
        windows.append(window)
        starts.append(index[start])
        if max_windows is not None and len(windows) >= max_windows:
            break

    if not windows:
        raise ValueError("No valid windows could be generated. Try lowering the stride or cleaning the data.")

    stacked = np.stack(windows)
    return stacked[:, :context, :], stacked[:, context:, :], starts


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronologically split the windows following the notebook convention."""
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio must be > 0, val_ratio >= 0 and their sum < 1.")

    n = len(X)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    return X_train, y_train, X_val, y_val, X_test, y_test


def scale_datasets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> WindowedDataset:
    feature_scaler = StandardScaler()
    n_samples, seq_len, n_features = X_train.shape
    feature_scaler.fit(X_train.reshape(n_samples * seq_len, n_features))

    def scale_features(arr: np.ndarray) -> np.ndarray:
        flat = arr.reshape(-1, n_features)
        scaled = feature_scaler.transform(flat)
        return scaled.reshape(arr.shape)

    target_scaler = StandardScaler()
    target_scaler.fit(y_train.reshape(-1, y_train.shape[-1]))

    def scale_targets(arr: np.ndarray) -> np.ndarray:
        flat = arr.reshape(-1, arr.shape[-1])
        scaled = target_scaler.transform(flat)
        return scaled.reshape(arr.shape)

    target_shape = y_train.shape[1:]

    return WindowedDataset(
        X_train=scale_features(X_train),
        y_train=scale_targets(y_train),
        X_val=scale_features(X_val),
        y_val=scale_targets(y_val),
        X_test=scale_features(X_test),
        y_test=scale_targets(y_test),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        target_shape=(target_shape[0], target_shape[1] if len(target_shape) > 1 else 1),
    )


class WaveForecastDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.reshape(y.shape[0], -1), dtype=torch.float32)

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover - simple delegation
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


class LSTMForecastModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.lstm(x)
        last_hidden = hn[-1]
        return self.fc(last_hidden)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size]
        return x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self) -> None:
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu1(out + res)


class TCNForecastModel(nn.Module):
    def __init__(self, input_size: int, num_channels: Sequence[int], kernel_size: int, dropout: float, output_size: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for i, channels in enumerate(num_channels):
            dilation = 2**i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(in_channels, channels, kernel_size, 1, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out[:, :, -1]
        return self.fc(out)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _train_torch_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    learning_rate: float,
    target_shape: Tuple[int, int],
    target_scaler: StandardScaler,
) -> Tuple[Dict[str, float], np.ndarray]:
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    epochs_no_improve = 0

    model.to(device)
    start = perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()
        val_loss /= len(val_loader)

        _LOGGER.info("Epoch %d/%d — train %.4f — val %.4f", epoch, epochs, train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                _LOGGER.info("Early stopping triggered after %d epochs.", epoch)
                break

    train_time = perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)

    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch).cpu().numpy()
            preds.append(pred)
            targets.append(y_batch.numpy())
    flat_pred = np.concatenate(preds, axis=0)
    flat_true = np.concatenate(targets, axis=0)

    pred_reshaped = flat_pred.reshape(-1, *target_shape)
    true_reshaped = flat_true.reshape(-1, *target_shape)

    y_pred_orig = pred_reshaped
    y_true_orig = true_reshaped
    if target_scaler is not None:
        y_pred_orig = target_scaler.inverse_transform(
            pred_reshaped.reshape(-1, target_shape[-1])
        ).reshape(pred_reshaped.shape)
        y_true_orig = target_scaler.inverse_transform(
            true_reshaped.reshape(-1, target_shape[-1])
        ).reshape(true_reshaped.shape)

    if y_pred_orig.shape[-1] == 1:
        y_pred_flat = y_pred_orig[..., 0]
        y_true_flat = y_true_orig[..., 0]
    else:
        y_pred_flat = y_pred_orig.reshape(y_pred_orig.shape[0], -1)
        y_true_flat = y_true_orig.reshape(y_true_orig.shape[0], -1)

    metrics = {
        "mae": float(mean_absolute_error(y_true_flat, y_pred_flat)),
        "mse": float(mean_squared_error(y_true_flat, y_pred_flat)),
        "rmse": float(math.sqrt(mean_squared_error(y_true_flat, y_pred_flat))),
        "time": train_time,
    }
    return metrics, y_pred_orig


def train_lstm(
    data: WindowedDataset,
    batch_size: int,
    epochs: int,
    patience: int,
    hidden_size: int,
    num_layers: int,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray]:
    train_loader = torch.utils.data.DataLoader(
        WaveForecastDataset(data.X_train, data.y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        WaveForecastDataset(data.X_val, data.y_val), batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        WaveForecastDataset(data.X_test, data.y_test), batch_size=batch_size, shuffle=False
    )

    model = LSTMForecastModel(
        input_size=data.X_train.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=int(np.prod(data.target_shape)),
    )
    return _train_torch_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        epochs,
        patience,
        learning_rate=1e-3,
        target_shape=data.target_shape,
        target_scaler=data.target_scaler,
    )


def train_tcn(
    data: WindowedDataset,
    batch_size: int,
    epochs: int,
    patience: int,
    num_channels: Sequence[int],
    kernel_size: int,
    dropout: float,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray]:
    train_loader = torch.utils.data.DataLoader(
        WaveForecastDataset(data.X_train, data.y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        WaveForecastDataset(data.X_val, data.y_val), batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        WaveForecastDataset(data.X_test, data.y_test), batch_size=batch_size, shuffle=False
    )

    model = TCNForecastModel(
        input_size=data.X_train.shape[-1],
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        output_size=int(np.prod(data.target_shape)),
    )
    return _train_torch_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        epochs,
        patience,
        learning_rate=1e-3,
        target_shape=data.target_shape,
        target_scaler=data.target_scaler,
    )


def train_xgboost(data: WindowedDataset) -> Tuple[Dict[str, float], np.ndarray]:
    X_train = data.X_train.reshape(data.X_train.shape[0], -1)
    X_val = data.X_val.reshape(data.X_val.shape[0], -1)
    X_test = data.X_test.reshape(data.X_test.shape[0], -1)

    y_train = data.y_train.reshape(data.y_train.shape[0], -1)
    y_val = data.y_val.reshape(data.y_val.shape[0], -1)
    y_test = data.y_test.reshape(data.y_test.shape[0], -1)

    model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42,
        )
    )
    start = perf_counter()
    model.fit(np.vstack([X_train, X_val]), np.vstack([y_train, y_val]))
    train_time = perf_counter() - start

    y_pred_scaled = model.predict(X_test)
    if y_pred_scaled.ndim == 1:
        y_pred_scaled = y_pred_scaled[:, np.newaxis]

    target_scaler = data.target_scaler
    y_pred = target_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, data.target_shape[-1])
    ).reshape(-1, *data.target_shape)
    y_true = target_scaler.inverse_transform(
        y_test.reshape(-1, data.target_shape[-1])
    ).reshape(-1, *data.target_shape)

    if data.target_shape[-1] == 1:
        y_pred_eval = y_pred[..., 0]
        y_true_eval = y_true[..., 0]
    else:
        y_pred_eval = y_pred.reshape(y_pred.shape[0], -1)
        y_true_eval = y_true.reshape(y_true.shape[0], -1)

    metrics = {
        "mae": float(mean_absolute_error(y_true_eval, y_pred_eval)),
        "mse": float(mean_squared_error(y_true_eval, y_pred_eval)),
        "rmse": float(math.sqrt(mean_squared_error(y_true_eval, y_pred_eval))),
        "time": train_time,
    }
    return metrics, y_pred


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["m6_buoy", "weather"], default="m6_buoy")
    parser.add_argument("--data-path", type=Path, default=None, help="Optional local dataset path.")
    parser.add_argument("--context", type=int, default=336, help="Context window length (hours).")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon (hours).")
    parser.add_argument("--stride", type=int, default=4, help="Stride between successive windows.")
    parser.add_argument("--target", default="WaveHeight", help="Target column name.")
    parser.add_argument("--features", nargs="*", default=None, help="Additional feature columns to keep.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lstm", "tcn", "xgboost"],
        default=["lstm", "tcn", "xgboost"],
        help="Models to train.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden size for the LSTM.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument(
        "--tcn-channels",
        type=int,
        nargs="+",
        default=[64, 64],
        help="Number of channels for each temporal block in the TCN.",
    )
    parser.add_argument("--tcn-kernel", type=int, default=3)
    parser.add_argument("--tcn-dropout", type=float, default=0.2)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-windows", type=int, default=None, help="Optional cap on the number of generated windows.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON file where metrics will be saved.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s | %(message)s")

    torch.manual_seed(42)
    np.random.seed(42)

    df = load_dataset(args.dataset, args.data_path)
    X_context, Y_future, starts = extract_windows(
        df,
        target_column=args.target,
        context=args.context,
        horizon=args.horizon,
        step=args.stride,
        selected_columns=args.features,
        max_windows=args.max_windows,
    )
    _LOGGER.info("Generated %d windows starting at %s", len(starts), starts[0])

    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(
        X_context, Y_future, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )
    windowed = scale_datasets(X_train, y_train, X_val, y_val, X_test, y_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOGGER.info("Using device %s", device)

    metrics: Dict[str, Dict[str, float]] = {}

    if "lstm" in args.models:
        _LOGGER.info("Training LSTM model…")
        lstm_metrics, _ = train_lstm(
            windowed,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            device=device,
        )
        metrics["lstm"] = lstm_metrics
        _LOGGER.info("LSTM metrics: %s", lstm_metrics)

    if "tcn" in args.models:
        _LOGGER.info("Training TCN model…")
        tcn_metrics, _ = train_tcn(
            windowed,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            num_channels=args.tcn_channels,
            kernel_size=args.tcn_kernel,
            dropout=args.tcn_dropout,
            device=device,
        )
        metrics["tcn"] = tcn_metrics
        _LOGGER.info("TCN metrics: %s", tcn_metrics)

    if "xgboost" in args.models:
        _LOGGER.info("Training XGBoost model…")
        xgb_metrics, _ = train_xgboost(windowed)
        metrics["xgboost"] = xgb_metrics
        _LOGGER.info("XGBoost metrics: %s", xgb_metrics)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as fh:
            json.dump(metrics, fh, indent=2)
        _LOGGER.info("Saved metrics to %s", args.output)

    _LOGGER.info("Finished training. Metrics summary: %s", metrics)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
