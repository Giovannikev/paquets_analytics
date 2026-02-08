from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _to_int(value: object) -> Optional[int]:
    try:
        if pd.isna(value):
            return None
        if isinstance(value, (int, np.integer)):
            return int(value)
        s = str(value).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None


def _ip_to_int(ip: object) -> Optional[int]:
    try:
        if pd.isna(ip):
            return None
        s = str(ip).strip()
        parts = s.split(".")
        if len(parts) == 4 and all(p.isdigit() for p in parts):
            a, b, c, d = [int(p) for p in parts]
            if 0 <= a <= 255 and 0 <= b <= 255 and 0 <= c <= 255 and 0 <= d <= 255:
                return (a << 24) + (b << 16) + (c << 8) + d
        return None
    except Exception:
        return None


def _parse_ports(info: object) -> Tuple[Optional[int], Optional[int]]:
    try:
        if pd.isna(info):
            return None, None
        s = str(info)
        m = re.search(r"(\d+)\s*>\s*(\d+)", s)
        if m:
            return _to_int(m.group(1)), _to_int(m.group(2))
        return None, None
    except Exception:
        return None, None


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        lc = cand.lower()
        if lc in lower_cols:
            return lower_cols[lc]
    for c in df.columns:
        for cand in candidates:
            if cand.lower() in c.lower():
                return c
    return None


def build_feature_matrix(df: pd.DataFrame, top_protocols: int = 12) -> pd.DataFrame:
    time_col = _find_column(df, ["Time", "frame.time_relative", "timestamp", "ts"])
    src_col = _find_column(df, ["Source", "src", "ip.src", "source"])
    dst_col = _find_column(df, ["Destination", "dst", "ip.dst", "destination"])
    proto_col = _find_column(df, ["Protocol", "proto", "protocol"])
    length_col = _find_column(df, ["Length", "frame.len", "len", "size"])  # noqa: E501
    info_col = _find_column(df, ["Info", "_ws.col.Info", "details", "summary"])  # noqa: E501

    if length_col is None:
        raise ValueError("Length column not found")

    lengths = df[length_col].apply(_to_int).fillna(0).astype(int)

    if time_col is not None:
        times = pd.to_numeric(df[time_col], errors="coerce")
        try:
            times = times.ffill()
        except Exception:
            times = times.fillna(0.0)
        durations = times.diff().fillna(0.0)
    else:
        durations = pd.Series([0.0] * len(df), index=df.index)

    if info_col is not None:
        ports = df[info_col].apply(_parse_ports)
        src_ports = ports.apply(lambda t: t[0]).apply(_to_int).fillna(0).astype(int)
        dst_ports = ports.apply(lambda t: t[1]).apply(_to_int).fillna(0).astype(int)
    else:
        src_ports = pd.Series([0] * len(df), index=df.index)
        dst_ports = pd.Series([0] * len(df), index=df.index)

    if src_col is not None:
        src_ip_int = df[src_col].apply(_ip_to_int).fillna(0).astype(int)
    else:
        src_ip_int = pd.Series([0] * len(df), index=df.index)

    if dst_col is not None:
        dst_ip_int = df[dst_col].apply(_ip_to_int).fillna(0).astype(int)
    else:
        dst_ip_int = pd.Series([0] * len(df), index=df.index)

    if proto_col is not None:
        protos = df[proto_col].astype(str).fillna("UNKNOWN")
    else:
        protos = pd.Series(["UNKNOWN"] * len(df), index=df.index)

    top_counts = protos.value_counts().head(top_protocols).index.tolist()
    proto_one_hot = pd.get_dummies(protos.where(protos.isin(top_counts), other="OTHER"), prefix="proto")

    features = pd.DataFrame(
        {
            "length": lengths,
            "duration": durations.astype(float),
            "src_port": src_ports,
            "dst_port": dst_ports,
            "src_ip": src_ip_int,
            "dst_ip": dst_ip_int,
        }
    )

    features = pd.concat([features, proto_one_hot], axis=1)
    return features


def scale_features(features: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    try:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = scaler.fit_transform(features.values)
        return pd.DataFrame(X, index=features.index, columns=features.columns), scaler.mean_, scaler.scale_
    except Exception:
        X = features.values.astype(float)
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        std = np.where(std == 0, 1.0, std)
        Xs = (X - mean) / std
        return pd.DataFrame(Xs, index=features.index, columns=features.columns), mean, std


def compute_reference_metrics(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    diffs = X - mu
    euclid = np.linalg.norm(diffs, axis=1)
    denom = np.linalg.norm(X, axis=1) * np.linalg.norm(mu)
    denom = np.where(denom == 0, 1.0, denom)
    cosine = (X @ mu) / denom
    return euclid, cosine


def pca_project(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=0)
        comps = pca.fit_transform(X)
        var = pca.explained_variance_ratio_
        return comps, var
    except Exception:
        Xc = X - np.nanmean(X, axis=0)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        comps = U[:, :2] * S[:2]
        var = (S[:2] ** 2) / np.sum(S ** 2)
        return comps, var


def detect_anomalies(X: np.ndarray, contamination: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor

        iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=0)
        iso.fit(X)
        iso_score = -iso.decision_function(X)
        iso_label = iso.predict(X)

        lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        lof_label = lof.fit_predict(X)
        lof_score = -lof.negative_outlier_factor_

        score = (iso_score + lof_score) / 2.0
        label = np.where((iso_label == -1) | (lof_label == -1), -1, 1)
        return score, label
    except Exception:
        z = np.abs((X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0))
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        score = np.nanmean(z, axis=1)
        thr = np.quantile(score, 1.0 - contamination)
        label = np.where(score >= thr, -1, 1)
        return score, label


def save_projection_plot(points: np.ndarray, labels: np.ndarray, output_path: Path, title: str) -> Path:
    try:
        import plotly.express as px

        df_plot = pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "label": labels})
        fig = px.scatter(df_plot, x="x", y="y", color=df_plot["label"].map({1: "normal", -1: "anomaly"}), title=title)
        fig.write_html(str(output_path))
        return output_path
    except Exception:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = np.where(labels == -1, "red", "steelblue")
        plt.figure(figsize=(8, 6))
        plt.scatter(points[:, 0], points[:, 1], c=colors, s=14)
        plt.title(title)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        png_path = output_path.with_suffix(".png")
        plt.savefig(str(png_path))
        return png_path


def run(input_csv: Path, output_dir: Path, contamination: float, top_protocols: int) -> Dict[str, Path]:
    df = pd.read_csv(input_csv)
    features = build_feature_matrix(df, top_protocols=top_protocols)
    Xs, mean, scale = scale_features(features)
    X = Xs.values.astype(float)
    euclid, cosine = compute_reference_metrics(X)
    comps, var = pca_project(X)
    score, label = detect_anomalies(X, contamination=contamination)

    output_dir.mkdir(parents=True, exist_ok=True)

    features_out = output_dir / "features.csv"
    features_with_metrics = features.copy()
    features_with_metrics["euclidean_to_mean"] = euclid
    features_with_metrics["cosine_to_mean"] = cosine
    features_with_metrics.to_csv(features_out, index=False)

    anomalies_out = output_dir / "anomalies.csv"
    anomalies_df = pd.DataFrame({"index": np.arange(len(label)), "score": score, "label": label})
    anomalies_df.sort_values("score", ascending=False).to_csv(anomalies_out, index=False)

    plot_out = output_dir / "pca_projection.html"
    actual_plot = save_projection_plot(comps, label, plot_out, title=f"PCA variance={var.sum():.2f}")

    return {"features": features_out, "anomalies": anomalies_out, "projection": actual_plot}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--top-protocols", type=int, default=12)
    args = parser.parse_args(argv)

    input_csv = Path(args.input)
    if not input_csv.exists():
        print("Input CSV not found:", input_csv)
        return 2

    output_dir = Path(args.output) if args.output else input_csv.parent / "outputs"
    try:
        result_paths = run(input_csv, output_dir, contamination=args.contamination, top_protocols=args.top_protocols)
        for k, p in result_paths.items():
            print(f"{k}: {p}")
        return 0
    except Exception as e:
        print("Error:", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

