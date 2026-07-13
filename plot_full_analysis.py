#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полный анализ RSSI: EEMD (sequential) + авто-выбор мод + PCA + Welch PSD.
Строит графики: исходный сигнал, выбранные IMF, PC1, Welch-спектр PC1 с пиком пульса.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from PyEMD import EMD
import warnings
warnings.filterwarnings("ignore")

# ========== ПАРАМЕТРЫ (синхронизированы с server_ws.py) ==========
SAMPLE_INTERVAL = 0.05      # 20 Гц
FILTER_LOW      = 0.5
FILTER_HIGH     = 3.5
FILTER_ORDER    = 4
PCA_VAR_THRESHOLD = 0.45

# EEMD (sequential, 30 trials for plot speed)
CEEMDAN_TRIALS = 30
CEEMDAN_NOISE  = 0.2

# Z-score нормализация перед CEEMDAN
ZSCORE_NORMALIZE = True

# Авто-выбор мод
IMF_AUTO_FMIN = 0.5         # нижняя граница поиска «сердечной» моды, Гц
IMF_AUTO_FMAX = 3.0         # верхняя граница поиска, Гц
IMF_AUTO_N    = 3           # топ-N мод по пиковой амплитуде → в PCA

# Диапазон для обнаружения BPM
BPM_MIN = 60
BPM_MAX = 120
# =====================================================================

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def run_eemd_sequential(signal, trials=CEEMDAN_TRIALS, noise_amp=CEEMDAN_NOISE):
    """Sequential EEMD — no subprocess/multiprocessing, safe in venv on Windows."""
    all_imfs = []
    for i in range(trials):
        np.random.seed(42 + i)
        noisy  = signal + noise_amp * np.random.randn(len(signal))
        imfs_i = EMD()(noisy)
        all_imfs.append(imfs_i)
    min_n = min(x.shape[0] for x in all_imfs)
    return np.mean([x[:min_n] for x in all_imfs], axis=0)

def auto_select_imfs(imfs, fs, fmin=IMF_AUTO_FMIN, fmax=IMF_AUTO_FMAX, top_n=IMF_AUTO_N):
    """Выбирает top_n мод с наибольшим спектральным пиком в [fmin, fmax] Гц."""
    peak_amps = []
    for i in range(imfs.shape[0]):
        fft_i   = np.fft.fft(imfs[i])
        freqs_i = np.fft.fftfreq(len(imfs[i]), d=1/fs)
        mag_i   = np.abs(fft_i)
        mask    = (freqs_i >= fmin) & (freqs_i <= fmax)
        peak_amps.append(np.max(mag_i[mask]) if np.any(mask) else 0.0)
    sorted_idxs   = np.argsort(peak_amps)[::-1]
    selected_idxs = sorted(sorted_idxs[:min(top_n, imfs.shape[0])].tolist())
    return selected_idxs, np.array(peak_amps)

def main():
    # UTF-8 stdout: fix encoding for arrow/sigma/emoji on Windows cp1251 consoles
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

    if len(sys.argv) < 2:
        print("Usage: python plot_full_analysis.py <csv_file> [--save output.png]")
        sys.exit(1)

    csv_file = sys.argv[1]
    save_path = None
    if "--save" in sys.argv:
        idx = sys.argv.index("--save")
        if idx+1 < len(sys.argv):
            save_path = sys.argv[idx+1]

    # Загрузка данных
    df = pd.read_csv(csv_file)
    if "timestamp" not in df.columns or "rssi" not in df.columns:
        print("Файл должен содержать колонки 'timestamp' и 'rssi'")
        sys.exit(1)

    # Удаление дубликатов по timestamp (артефакт батч-отправки ESP32)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    time_raw = df["timestamp"].values
    rssi_raw = df["rssi"].values.astype(float)
    t_raw    = time_raw - time_raw[0]
    duration = t_raw[-1]
    print(f"Исходных отсчётов: {len(df)},  длительность: {duration:.2f} с")

    # Ресэмплинг на регулярную сетку 20 Гц (как в server_ws.py)
    fs  = 1.0 / SAMPLE_INTERVAL
    t   = np.arange(0, duration, SAMPLE_INTERVAL)
    itp = interp1d(t_raw, rssi_raw, kind="linear", bounds_error=False, fill_value="extrapolate")
    rssi = itp(t)
    print(f"Ресэмплинг: {len(t)} отсчётов @ {fs:.0f} Hz")

    # 1. Детрендинг + полосовой фильтр
    data_detrended = rssi - np.mean(rssi)
    data_filtered  = bandpass_filter(data_detrended, FILTER_LOW, FILTER_HIGH, fs, FILTER_ORDER)

    # 2. Z-score нормализация перед EEMD
    if ZSCORE_NORMALIZE:
        sig_std   = np.std(data_filtered)
        data_norm = data_filtered / sig_std if sig_std > 1e-9 else data_filtered
        print(f"Z-score: std = {sig_std:.4f}  ->  сигнал нормализован")
    else:
        data_norm = data_filtered

    # 3. EEMD (sequential)
    print(f"Выполняется EEMD ({CEEMDAN_TRIALS} trials, noise={CEEMDAN_NOISE})...")
    try:
        imfs = run_eemd_sequential(data_norm)
    except Exception as e:
        print(f"Ошибка EEMD: {e}")
        sys.exit(1)
    print(f"Получено мод: {imfs.shape[0]}")

    # 4. Авто-выбор мод по пику в [{IMF_AUTO_FMIN}–{IMF_AUTO_FMAX}] Гц
    selected_idxs, auto_amps = auto_select_imfs(imfs, fs)
    if not selected_idxs:
        print("Не удалось выбрать моды")
        sys.exit(1)
    selected_imfs = imfs[selected_idxs, :]
    print(f"Выбраны моды: {selected_idxs}  (по пику в {IMF_AUTO_FMIN}–{IMF_AUTO_FMAX} Гц)")

    # 5. PCA
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(selected_imfs.T).flatten()
    explained_var = pca.explained_variance_ratio_[0]
    print(f"Объяснённая дисперсия PC1: {explained_var:.3f}")
    if explained_var < PCA_VAR_THRESHOLD:
        print(f"⚠️  Дисперсия ниже порога {PCA_VAR_THRESHOLD} — результат может быть ненадёжным")

    # 6. Welch PSD + параболическая интерполяция пика
    freq_lo = BPM_MIN / 60.0
    freq_hi = BPM_MAX / 60.0
    nperseg = max(len(pc1) // 2, 64)
    freqs, psd = welch(pc1, fs=fs, nperseg=nperseg, window='hann', scaling='density')
    magnitudes  = psd

    mask = (freqs >= freq_lo) & (freqs <= freq_hi)
    if not np.any(mask):
        print("Пульс не обнаружен в диапазоне")
        hr_freq = np.nan
        raw_bpm = np.nan
    else:
        mask_indices = np.where(mask)[0]
        peak_idx_rel = np.argmax(magnitudes[mask])
        peak_abs_idx = mask_indices[peak_idx_rel]

        # Параболическая интерполяция
        if 1 <= peak_abs_idx < len(magnitudes) - 1:
            y1, y2, y3 = magnitudes[peak_abs_idx-1], magnitudes[peak_abs_idx], magnitudes[peak_abs_idx+1]
            denom = 2 * (y1 - 2*y2 + y3)
            if denom != 0:
                delta       = (y1 - y3) / denom
                refined_idx = peak_abs_idx + delta
                hr_freq     = np.interp(refined_idx, np.arange(len(freqs)), freqs)
            else:
                hr_freq = freqs[peak_abs_idx]
        else:
            hr_freq = freqs[peak_abs_idx]
        raw_bpm = hr_freq * 60
        print(f"🎯 Пульс: {raw_bpm:.1f} BPM  (частота {hr_freq:.3f} Гц)")
    
    # ========== ПОСТРОЕНИЕ ГРАФИКОВ ==========
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axes

    # Исходный сигнал
    ax1.plot(t, rssi, linewidth=0.5, color='gray')
    ax1.set_ylabel("RSSI, dBm")
    ax1.set_title("Исходный сигнал RSSI")
    ax1.set_xlabel("Время, с")
    ax1.grid(True)

    # Авто-выбранные моды CEEMDAN (сдвинуты для наглядности)
    offsets = np.max(np.abs(selected_imfs)) * 2.2  # автоматический сдвиг
    for j, imf_idx in enumerate(selected_idxs):
        ax2.plot(t, selected_imfs[j] + j * offsets, linewidth=0.7,
                 label=f"IMF {imf_idx}  (пик={auto_amps[imf_idx]:.1f})")
    ax2.set_ylabel("Амплитуда (сдвиг)")
    ax2.set_title(f"Авто-выбранные моды CEEMDAN {selected_idxs}  "
                  f"(пик в {IMF_AUTO_FMIN}–{IMF_AUTO_FMAX} Гц, топ-{IMF_AUTO_N})")
    ax2.set_xlabel("Время, с")
    ax2.legend(fontsize=8)
    ax2.grid(True)

    # Первая главная компонента
    ax3.plot(t, pc1, linewidth=0.8, color='darkred')
    ax3.set_ylabel("PC1")
    ax3.set_title(f"PC1 (объяснённая дисперсия = {explained_var:.3f})")
    ax3.set_xlabel("Время, с")
    ax3.grid(True)

    # Welch PSD с пиком
    ax4.plot(freqs, magnitudes, color='steelblue', linewidth=1.0)
    ax4.axvspan(freq_lo, freq_hi, alpha=0.08, color='green', label=f"{BPM_MIN}–{BPM_MAX} BPM зона")
    ax4.set_xlim(0, 5)
    ax4.set_xlabel("Частота, Гц")
    ax4.set_ylabel("PSD (мощность)")
    ax4.set_title("Welch PSD  (PC1)")
    ax4.grid(True)
    if not np.isnan(hr_freq):
        ax4.axvline(hr_freq, color='red', linestyle='--',
                    label=f"Пульс: {raw_bpm:.1f} BPM  ({hr_freq:.3f} Гц)")
    ax4.legend(fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"График сохранён в {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    main()