"""
utils.py — Rekayasa fitur & target.

>>> LENGKAPI semua bagian bertanda TODO. <<<
Bagian ini dikerjakan di PERTEMUAN 1 (lihat Panduan Build).

Kenapa file ini penting: fungsi di sini dipakai OLEH train_model.py DAN app.py,
jadi perhitungan fitur selalu konsisten antara saat melatih & saat memprediksi.
"""
import numpy as np
import pandas as pd

# Daftar fitur yang dipakai model (SUDAH disiapkan — jangan diubah urutannya).
FITUR = ["ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5", "ret_lag10",
         "ma5", "ma20", "vol10", "rsi"]


def buat_fitur(df: pd.DataFrame) -> pd.DataFrame:
    """Terima DataFrame berkolom 'Close', kembalikan + kolom-kolom fitur teknikal."""
    df = df.copy()

    # TODO 1 — Return harian: persentase perubahan harga penutupan.
    #   Petunjuk: df["Close"].pct_change()
    df["return"] = None  # <-- GANTI

    # TODO 2 — Lag return untuk lag 1, 2, 3, 5, 10 hari.
    #   Petunjuk: df["return"].shift(lag)
    for lag in [1, 2, 3, 5, 10]:
        df[f"ret_lag{lag}"] = None  # <-- GANTI

    # TODO 3 — Moving average RELATIF 5 & 20 hari (rasio terhadap harga sekarang).
    #   Petunjuk: df["Close"].rolling(5).mean() / df["Close"] - 1
    df["ma5"] = None   # <-- GANTI
    df["ma20"] = None  # <-- GANTI

    # TODO 4 — Volatilitas 10 hari (standar deviasi dari return).
    #   Petunjuk: df["return"].rolling(10).std()
    df["vol10"] = None  # <-- GANTI

    # --- RSI (14) sudah disediakan, TIDAK perlu diubah ---
    delta = df["Close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn)

    return df


def buat_target(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan kolom 'target' = 1 jika harga BESOK naik, 0 jika tidak."""
    df = df.copy()

    # TODO 5 — Target arah: 1 kalau return BESOK > 0, selain itu 0.
    #   Petunjuk: (df["Close"].pct_change().shift(-1) > 0).astype(int)
    #   Catatan: shift(-1) melihat ke depan, tapi ini BOLEH karena hanya untuk LABEL.
    df["target"] = None  # <-- GANTI

    return df
