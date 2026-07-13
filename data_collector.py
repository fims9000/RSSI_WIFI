#!/usr/bin/env python3
"""
WebSocket сервер для сбора RSSI с ESP8266.
Режимы:
  --mode realtime  – обработка сигнала и вывод пульса (как в оригинале)
  --mode collect   – запись всех входящих данных в CSV-файл без обработки
"""
#тут крч в файл csv собирается инфа

import asyncio
import websockets
import argparse
import signal
import sys
from datetime import datetime
import numpy as np
from collections import deque
import concurrent.futures

# Импорт для режима realtime (не обязателен, если режим не используется)
try:
    from sklearn.decomposition import PCA
    from PyEMD.EMD import EMD
    from scipy.signal import butter, filtfilt
    PROCESSING_AVAILABLE = True
except ImportError:
    PROCESSING_AVAILABLE = False
    print("⚠️ Для режима realtime установите: numpy scikit-learn PyEMD scipy")

# ================== ПАРАМЕТРЫ (общие) ==================
BUFFER_SIZE = 400          # 20 секунд при 20 Гц
SAMPLE_INTERVAL = 0.05     # 50 мс (20 Гц)
PROCESS_INTERVAL = 20      # секунд между расчётами (только realtime)

# Параметры обработки (для realtime)
FILTER_LOW = 0.5
FILTER_HIGH = 3.5
FILTER_ORDER = 4
EEMD_TRIALS = 300
EEMD_NOISE = 0.2
IMF_START = 1
IMF_END = 4
PCA_VAR_THRESHOLD = 0.45
MEDIAN_WINDOW = 5
# =======================================================

# Глобальные буферы для обоих режимов
rssi_buffer = deque(maxlen=BUFFER_SIZE)
timestamps = deque(maxlen=BUFFER_SIZE)
last_bpms = deque(maxlen=MEDIAN_WINDOW) if MEDIAN_WINDOW > 0 else None

# Файл для записи в режиме collect
collect_file = None

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def eemd(signal, trials, noise_width):
    all_imfs = []
    for _ in range(trials):
        noisy = signal + noise_width * np.random.randn(len(signal))
        emd = EMD()
        imfs = emd(noisy)
        all_imfs.append(imfs)
    min_imfs = min(imfs.shape[0] for imfs in all_imfs)
    avg_imfs = np.mean([imfs[:min_imfs, :] for imfs in all_imfs], axis=0)
    return avg_imfs

async def handler(websocket):
    print("✅ Клиент подключён")
    try:
        async for message in websocket:
            parts = message.strip().split(',')
            if len(parts) == 2:
                ts = int(parts[0])
                rssi = int(parts[1])
                rssi_buffer.append(rssi)
                timestamps.append(ts)

                # Режим сбора: запись в файл
                if collect_file is not None:
                    # Используем реальное время для метки (можно заменить на ts)
                    now = datetime.now().timestamp()
                    collect_file.write(f"{now},{rssi}\n")
                    collect_file.flush()
            else:
                print(f"⚠️ Неверный формат: {message}")
    except websockets.exceptions.ConnectionClosed:
        print("❌ Клиент отключился")
    except Exception as e:
        print(f"⚠️ Ошибка в обработчике: {e}")

def process_data(data):
    """Обработка сигнала (оригинальный алгоритм)"""
    if not PROCESSING_AVAILABLE:
        print("❌ Обработка недоступна: установите необходимые пакеты")
        return

    # 1. Удаление тренда
    data_detrended = data - np.mean(data)

    # 2. Полосовая фильтрация
    fs = 1.0 / SAMPLE_INTERVAL
    data_filtered = bandpass_filter(data_detrended, FILTER_LOW, FILTER_HIGH, fs, FILTER_ORDER)

    # 3. EEMD
    try:
        IMFs = eemd(data_filtered, trials=EEMD_TRIALS, noise_width=EEMD_NOISE)
    except Exception as e:
        print(f"❌ Ошибка EEMD: {e}")
        return

    if IMFs.shape[0] <= IMF_START:
        print(f"⚠️ Недостаточно IMF (всего {IMFs.shape[0]})")
        return

    end_idx = min(IMF_END, IMFs.shape[0])
    selected = IMFs[IMF_START:end_idx, :]

    # 4. PCA
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(selected.T).flatten()
    explained_var = pca.explained_variance_ratio_[0]

    if explained_var < PCA_VAR_THRESHOLD:
        print(f"⚠️ Объяснённая дисперсия {explained_var:.3f} ниже порога")
        return

    # 5. FFT с интерполяцией
    fft_vals = np.fft.fft(pc1)
    freqs = np.fft.fftfreq(len(pc1), d=SAMPLE_INTERVAL)
    magnitudes = np.abs(fft_vals)

    mask = (freqs >= 1.0) & (freqs <= 2.0)
    if not np.any(mask):
        print("😴 Пульс не обнаружен")
        return

    mask_indices = np.where(mask)[0]
    peak_idx_rel = np.argmax(magnitudes[mask])
    peak_abs_idx = mask_indices[peak_idx_rel]

    # Параболическая интерполяция
    if 1 <= peak_abs_idx < len(magnitudes) - 1:
        y1, y2, y3 = magnitudes[peak_abs_idx-1], magnitudes[peak_abs_idx], magnitudes[peak_abs_idx+1]
        denom = 2 * (y1 - 2*y2 + y3)
        if denom != 0:
            delta = (y1 - y3) / denom
            refined_idx = peak_abs_idx + delta
            hr_freq = np.interp(refined_idx, np.arange(len(freqs)), freqs)
        else:
            hr_freq = freqs[peak_abs_idx]
    else:
        hr_freq = freqs[peak_abs_idx]

    raw_bpm = hr_freq * 60

    # Медианный фильтр
    if MEDIAN_WINDOW > 0 and last_bpms is not None:
        last_bpms.append(raw_bpm)
        median_bpm = np.median(last_bpms)
        print(f"❤️ Пульс: {median_bpm:.1f} BPM (сырой: {raw_bpm:.1f}, var={explained_var:.3f})")
    else:
        print(f"❤️ Пульс: {raw_bpm:.1f} BPM (var={explained_var:.3f})")

async def periodic_processing():
    """Фоновая задача для обработки каждые PROCESS_INTERVAL секунд (только realtime)"""
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        while True:
            await asyncio.sleep(PROCESS_INTERVAL)
            if len(rssi_buffer) < BUFFER_SIZE:
                continue
            data_copy = np.array(list(rssi_buffer))
            await loop.run_in_executor(pool, process_data, data_copy)

async def main():
    parser = argparse.ArgumentParser(description="WebSocket сервер для RSSI")
    parser.add_argument("--mode", choices=["realtime", "collect"], default="realtime",
                        help="realtime: обработка пульса; collect: запись в CSV")
    parser.add_argument("--output", type=str, default=None,
                        help="Имя выходного CSV-файла (только для collect). По умолчанию rssi_YYYYMMDD_HHMMSS.csv")
    args = parser.parse_args()

    global collect_file
    if args.mode == "collect":
        if args.output is None:
            filename = f"rssi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            filename = args.output
        collect_file = open(filename, "w")
        collect_file.write("timestamp,rssi\n")
        print(f"📁 Режим СБОРА данных. Файл: {filename}")
        print("   Для остановки нажмите Ctrl+C")
    else:
        print("🚀 Режим РЕАЛЬНОГО ВРЕМЕНИ (вычисление пульса)")
        if not PROCESSING_AVAILABLE:
            print("❌ Ошибка: не установлены зависимости. Установите: pip install numpy scikit-learn PyEMD scipy")
            sys.exit(1)

    print(f"Сервер запущен на ws://0.0.0.0:5000")
    
    stop_event = asyncio.Event()
    
    # Кросс-платформенная обработка Ctrl+C
    def signal_handler():
        print("\n🛑 Остановка сервера...")
        stop_event.set()

    # Для Unix используем add_signal_handler, для Windows просто обрабатываем исключение
    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, signal_handler)
    # ---------------------------------

    async with websockets.serve(handler, "0.0.0.0", 5000):
        tasks = [asyncio.Future()]
        if args.mode == "realtime":
            tasks.append(periodic_processing())
        
        # Ожидаем остановки либо по событию, либо по KeyboardInterrupt
        try:
            await stop_event.wait()
        except KeyboardInterrupt:
            print("\n🛑 Получен Ctrl+C, завершаем...")
        finally:
            if collect_file:
                collect_file.close()
                print(f"✅ Данные сохранены в {collect_file.name}")

if __name__ == "__main__":
    asyncio.run(main())