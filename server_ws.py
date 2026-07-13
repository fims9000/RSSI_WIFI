#основной файл, где происходит отработка в моменте

import asyncio
import websockets
import numpy as np
from sklearn.decomposition import PCA
from collections import deque
import concurrent.futures
import re
from PyEMD import EMD
from scipy.signal import butter, filtfilt, welch
from datetime import datetime

# ── ANSI-цвета (Windows Terminal / PowerShell >= Win10) ──────────────────────
try:
    import colorama
    colorama.init(autoreset=True)
    C_RESET   = colorama.Style.RESET_ALL
    C_BOLD    = colorama.Style.BRIGHT
    C_DIM     = colorama.Style.DIM
    C_RED     = colorama.Fore.RED
    C_GREEN   = colorama.Fore.GREEN
    C_YELLOW  = colorama.Fore.YELLOW
    C_CYAN    = colorama.Fore.CYAN
    C_MAGENTA = colorama.Fore.MAGENTA
    C_WHITE   = colorama.Fore.WHITE
    C_BLUE    = colorama.Fore.BLUE
except ImportError:
    C_RESET = C_BOLD = C_DIM = C_RED = C_GREEN = C_YELLOW = ""
    C_CYAN  = C_MAGENTA = C_WHITE = C_BLUE = ""

# ══════════════════════ ПАРАМЕТРЫ ════════════════════════════════════════════
BUFFER_SIZE     = 400   # отсчётов  (20 с при 20 Гц)
SAMPLE_INTERVAL = 0.05  # с / отсчёт  (20 Гц)

# CEEMDAN
CEEMDAN_TRIALS = 100    # число ансамблей
CEEMDAN_NOISE  = 0.2    # амплитуда шума (σ)

# Полосовой фильтр Баттерворта
FILTER_LOW   = 0.5      # Гц
FILTER_HIGH  = 3.5      # Гц
FILTER_ORDER = 4

# Z-score нормализация перед CEEMDAN
ZSCORE_NORMALIZE = True

# Авто-выбор мод: диапазон и число
IMF_AUTO_FMIN = 0.5     # Гц
IMF_AUTO_FMAX = 3.0     # Гц
IMF_AUTO_N    = 3       # топ-N мод → PCA

# PCA
PCA_VAR_THRESHOLD = 0.45

# Медианное сглаживание BPM
MEDIAN_WINDOW = 5       # 0 — отключить

# Итоговый диапазон BPM
BPM_MIN = 60
BPM_MAX = 120
# ═════════════════════════════════════════════════════════════════════════════

rssi_buffer      = deque()
timestamps       = deque()
last_bpms        = deque(maxlen=MEDIAN_WINDOW) if MEDIAN_WINDOW > 0 else None
analysis_count   = 0
last_rssi        = None
client_connected = False

# ─────────────────────────────── Helpers ─────────────────────────────────────

def now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _strip_ansi(text: str) -> str:
    return re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]').sub('', text)

def _vis_len(text: str) -> int:
    return len(_strip_ansi(text))

def _pad(text: str, width: int) -> str:
    """Дополняет строку пробелами до width видимых символов."""
    return text + " " * max(0, width - _vis_len(text))

def _progress_bar(filled: int, total: int, width: int = 20) -> str:
    n = int(round(filled / total * width))
    return "█" * n + "░" * (width - n)

def _level_bar(ratio: float, width: int = 8) -> str:
    """Полоска уровня 0..1 с градиентом символов."""
    ratio  = max(0.0, min(1.0, ratio))
    filled = int(round(ratio * width))
    bar    = ""
    for i in range(width):
        if i < filled - 1:   bar += "█"
        elif i == filled - 1: bar += "▓"
        elif i == filled:     bar += "▒"
        else:                 bar += "░"
    return bar

# ─────────────────────────────── Баннер ──────────────────────────────────────

def print_banner():
    W   = 62
    SEP = "─"
    TOP = "┌" + "─" * W + "┐"
    MID = "├" + "─" * W + "┤"
    BOT = "└" + "─" * W + "┘"

    def row(label: str, value: str) -> str:
        content = f"  {label:<14}  {value}"
        return "│" + _pad(content, W) + "│"

    def title_row(text: str) -> str:
        return "│" + _pad("  " + text, W) + "│"

    lines = [
        TOP,
        title_row("HEART RATE MONITOR  —  RSSI / WiFi"),
        MID,
        row("WebSocket",  "ws://0.0.0.0:5000"),
        row("Частота",    f"{int(1/SAMPLE_INTERVAL)} Гц  ({int(SAMPLE_INTERVAL*1000)} мс/отсч.)"),
        row("Буфер",      f"{BUFFER_SIZE} отсч.  ({BUFFER_SIZE*SAMPLE_INTERVAL:.0f} с)"),
        "│" + " " * W + "│",
        title_row("Алгоритм"),
        row("  Фильтр",   f"Butterworth  {FILTER_LOW}–{FILTER_HIGH} Hz  (порядок {FILTER_ORDER})"),
        row("  Z-score",  f"{'вкл' if ZSCORE_NORMALIZE else 'выкл'}  (нормализация перед CEEMDAN)"),
        row("  CEEMDAN",  f"{CEEMDAN_TRIALS} ансамблей  |  шум σ = {CEEMDAN_NOISE:.2f}"),
        row("  Авто-мода",f"пик в {IMF_AUTO_FMIN}–{IMF_AUTO_FMAX} Hz  |  топ-{IMF_AUTO_N} → PCA"),
        row("  PCA",      f"порог дисперсии  ≥ {PCA_VAR_THRESHOLD}"),
        row("  Welch PSD",f"hann  |  BPM {BPM_MIN}–{BPM_MAX}  |  параболич. интерп."),
        row("  Медиана",  f"окно {MEDIAN_WINDOW}" if MEDIAN_WINDOW > 0 else "  Медиана    выкл"),
        BOT,
    ]

    print()
    for line in lines:
        print(C_CYAN + C_BOLD + line + C_RESET)
    print()

# ─────────────────────────── Подключение ─────────────────────────────────────

def print_connected():
    global client_connected
    client_connected = True
    msg = f"  CONNECTED   {now_str()}  "
    bar = "─" * (len(msg) + 2)
    print()
    print(C_GREEN + C_BOLD + "┌" + bar + "┐" + C_RESET)
    print(C_GREEN + C_BOLD + "│  " + msg + "  │" + C_RESET)
    print(C_GREEN + C_BOLD + "└" + bar + "┘" + C_RESET)
    print()

def print_disconnected():
    global client_connected
    client_connected = False
    msg = f"  DISCONNECTED   {now_str()}  "
    bar = "─" * (len(msg) + 2)
    print()
    print(C_RED + C_BOLD + "┌" + bar + "┐" + C_RESET)
    print(C_RED + C_BOLD + "│  " + msg + "  │" + C_RESET)
    print(C_RED + C_BOLD + "└" + bar + "┘" + C_RESET)
    print()

# ─────────────────────────── Прогресс ────────────────────────────────────────

def print_progress(current: int, total: int, rssi=None):
    if not client_connected:
        return
    pct     = min(current / total * 100, 100.0)
    bar     = _progress_bar(min(current, total), total)
    rssi_s  = f"  {rssi:+.0f} dBm" if rssi is not None else ""
    line = (
        f"\r  {C_CYAN}Accumulating{C_RESET}"
        f"  [{C_GREEN}{bar}{C_RESET}]"
        f"  {C_BOLD}{current:3d}/{total}{C_RESET}"
        f"  ({pct:4.1f}%)"
        f"  {C_YELLOW}RSSI{rssi_s}{C_RESET}     "
    )
    print(line, end="", flush=True)

# ─────────────────────── Блок анализа ────────────────────────────────────────

def print_analysis_block(count, n_samples, steps, imf_info):
    ts = now_str()
    W  = 56
    TOP = "┌" + "─" * W + "┐"
    MID = "├" + "─" * W + "┤"
    BOT = "└" + "─" * W + "┘"

    def row_raw(content: str) -> str:
        return "│ " + _pad(content, W - 1) + "│"

    def step_row(num: str, name: str, ok: bool, detail: str = "") -> str:
        status     = C_GREEN + "OK" + C_RESET if ok else C_RED + "FAIL" + C_RESET
        raw_status = "OK" if ok else "FAIL"
        name_col   = C_WHITE + f"{name:<20}" + C_RESET
        detail_col = C_DIM + detail + C_RESET
        # видимая длина: num(2) + space(1) + name(20) + space(1) + status(2-4) + spaces + detail
        raw_len    = 2 + 1 + 20 + 2 + len(raw_status) + 2 + len(detail)
        pad        = max(0, W - raw_len - 1)
        return "│ " + f"{num} " + name_col + " " + status + "  " + detail_col + " " * pad + "│"

    print()
    print(C_CYAN + TOP + C_RESET)

    # заголовок
    hdr = f"  Analysis #{count:<3}  {ts}  |  {n_samples} samples"
    print(C_CYAN + C_BOLD + row_raw(hdr) + C_RESET)
    print(C_CYAN + MID + C_RESET)

    # шаги
    steps_order = [
        ("1", "Detrend + Z-score", "detrend"),
        ("2", "Bandpass Filter",   "filter"),
        ("3", "CEEMDAN",           "eemd"),
        ("4", "Mode Selection",    "pca_sel"),
        ("5", "PCA",               "pca"),
        ("6", "Welch PSD",         "fft"),
    ]
    for num, name, key in steps_order:
        val = steps.get(key)
        if val is None:
            continue
        ok, detail = val
        print(step_row(num, name, ok, detail))

    # спектр мод
    if imf_info:
        print(C_CYAN + MID + C_RESET)
        print(row_raw(f"  IMF Spectrum  ({BPM_MIN}–{BPM_MAX} BPM range)"))
        for (idx, bpm, amp_ratio, selected) in imf_info:
            bar   = _level_bar(amp_ratio)
            bpm_s = f"{bpm:6.1f} BPM" if bpm is not None else "   ---  BPM"
            if selected:
                marker = C_GREEN + C_BOLD + ">" + C_RESET
                col    = C_GREEN + C_BOLD
                tag    = C_GREEN + " [selected]" + C_RESET
            else:
                marker = " "
                col    = C_WHITE
                tag    = ""
            # видимая длина: marker(1) + "  IMF X:  " + bar(8) + "  " + bpm_s(10) + tag
            raw_len = 1 + 10 + 8 + 2 + len(bpm_s) + (11 if selected else 0)
            pad     = max(0, W - raw_len - 1)
            line    = f" {marker}  IMF {idx}:  {col}{bar}{C_RESET}  {col}{bpm_s}{C_RESET}{tag}"
            print("│ " + line + " " * pad + "│")

    print(C_CYAN + BOT + C_RESET)

# ─────────────────────── Результат BPM ───────────────────────────────────────

def print_bpm_result(median_bpm: float, raw_bpm: float, var: float, n_in_median: int):
    W = 44
    TOP = "╔" + "═" * W + "╗"
    BOT = "╚" + "═" * W + "╝"

    def drow(text: str) -> str:
        return "║" + _pad(text, W) + "║"

    bpm_text  = f"   HR   {median_bpm:.1f} BPM"
    meta_text = f"   raw {raw_bpm:.1f}  |  median n={n_in_median}/{MEDIAN_WINDOW}  |  PCA var {var:.3f}"

    print()
    print(C_MAGENTA + C_BOLD + TOP + C_RESET)
    print(C_MAGENTA + C_BOLD + drow(bpm_text) + C_RESET)
    print(C_MAGENTA +          drow(meta_text) + C_RESET)
    print(C_MAGENTA + C_BOLD + BOT + C_RESET)
    print()

def print_low_var(var: float):
    msg = f"  Low signal quality: PCA var={var:.3f} < {PCA_VAR_THRESHOLD}  — skipping  "
    bar = "─" * (len(msg) + 2)
    print()
    print(C_YELLOW + "┌" + bar + "┐" + C_RESET)
    print(C_YELLOW + "│ " + msg + " │" + C_RESET)
    print(C_YELLOW + "└" + bar + "┘" + C_RESET)
    print()

# ─────────────────────── История измерений ───────────────────────────────────

def print_history(bpms: list):
    if not bpms:
        return
    bpm_vals = np.array(bpms)
    lo, hi   = bpm_vals.min(), bpm_vals.max()
    BLOCKS   = " ▁▂▃▄▅▆▇█"
    sparkline = ""
    for v in bpm_vals:
        sparkline += BLOCKS[4 if hi == lo else int((v - lo) / (hi - lo) * 8)]

    items = "  ".join(f"#{i+1} {v:.1f}" for i, v in enumerate(bpms))
    print(f"  {C_CYAN}History:{C_RESET}  {C_WHITE}{items}{C_RESET}   {C_DIM}{sparkline}{C_RESET}")
    print()

# ─────────────────────── DSP ─────────────────────────────────────────────────

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq  = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def _is_monotone(x: np.ndarray) -> bool:
    """Возвращает True, если сигнал монотонный (больше нечего разлагать)."""
    d = np.diff(x)
    return not (np.any(d > 0) and np.any(d < 0))


def run_ceemdan(signal: np.ndarray, trials: int, noise_amp: float) -> np.ndarray:
    N = len(signal)

    # Шаг 1: получить EMD белого шума для каждого ансамбля (один раз)
    noise_imfs_list = []
    for i in range(trials):
        np.random.seed(42 + i)
        w = np.random.randn(N)
        try:
            noise_imfs_list.append(EMD()(w))
        except Exception:
            noise_imfs_list.append(np.zeros((1, N)))

    max_noise_modes = min(x.shape[0] for x in noise_imfs_list)

    collected_imfs = []
    residue        = signal.copy()

    # Шаг 2: итеративно извлекаем моды
    for k in range(max_noise_modes):
        # Собрать k-ю моду белого шума из каждого ансамбля
        noise_mode_k = np.array([
            nif[k] if k < nif.shape[0] else np.zeros(N)
            for nif in noise_imfs_list
        ])  # shape: (trials, N)

        # Адаптивное масштабирование: eps_k = noise_amp * std(residue) / std(E_k(w))
        std_r = np.std(residue)
        std_n = np.std(noise_mode_k)
        eps_k = noise_amp * (std_r / std_n) if std_n > 1e-10 and std_r > 1e-10 else noise_amp

        # Извлечь первую моду из (residue + eps_k * E_k(wᵢ)) для каждого ансамбля
        first_imfs = []
        for i in range(trials):
            noisy_residue = residue + eps_k * noise_mode_k[i]
            try:
                imf0 = EMD()(noisy_residue)[0]
            except Exception:
                imf0 = noisy_residue
            first_imfs.append(imf0)

        # k-я мода CEEMDAN = среднее по ансамблям
        imf_k   = np.mean(first_imfs, axis=0)
        residue = residue - imf_k
        collected_imfs.append(imf_k)

        # Остановиться, если остаток монотонный
        if _is_monotone(residue):
            break

    # Добавить финальный остаток как последнюю "моду"
    if np.any(np.abs(residue) > 1e-10):
        collected_imfs.append(residue)

    return np.array(collected_imfs)

# ─────────────────────── WebSocket ───────────────────────────────────────────

async def handler(websocket):
    global last_rssi
    print_connected()
    try:
        async for message in websocket:
            parts = message.strip().split(',')
            if len(parts) == 2:
                rssi_buffer.append(int(parts[1]))
                timestamps.append(int(parts[0]))
                last_rssi = int(parts[1])
            else:
                print(f"\n  {C_YELLOW}Bad format:{C_RESET}  '{message}'")
    except websockets.exceptions.ConnectionClosed:
        print_disconnected()
    except Exception as e:
        print(f"\n  {C_RED}Handler error:{C_RESET}  {e}")

# ─────────────────────── Обработка данных ────────────────────────────────────

def process_data(data: np.ndarray):
    global analysis_count
    analysis_count += 1
    count = analysis_count
    n     = len(data)

    steps    = {}
    imf_info = []
    fs       = 1.0 / SAMPLE_INTERVAL

    # 1. Удаление среднего + Z-score нормализация
    mean_val       = np.mean(data)
    data_detrended = data - mean_val
    data_filtered  = bandpass_filter(data_detrended, FILTER_LOW, FILTER_HIGH, fs, FILTER_ORDER)

    if ZSCORE_NORMALIZE:
        sig_std = np.std(data_filtered)
        data_norm = data_filtered / sig_std if sig_std > 1e-9 else data_filtered
        steps['detrend'] = (True, f"mean={mean_val:.1f}  std={sig_std:.3f}  z-score=on")
    else:
        data_norm = data_filtered
        steps['detrend'] = (True, f"mean={mean_val:.1f}  z-score=off")

    steps['filter'] = (True, f"{FILTER_LOW}–{FILTER_HIGH} Hz  order={FILTER_ORDER}")

    # 2. CEEMDAN
    try:
        IMFs = run_ceemdan(data_norm, trials=CEEMDAN_TRIALS, noise_amp=CEEMDAN_NOISE)
    except Exception as e:
        steps['eemd'] = (False, str(e))
        print_analysis_block(count, n, steps, imf_info)
        return
    steps['eemd'] = (True, f"{CEEMDAN_TRIALS} trials  →  {IMFs.shape[0]} modes")

    # 3. Спектральная оценка мод + авто-выбор
    freq_lo = BPM_MIN / 60.0
    freq_hi = BPM_MAX / 60.0
    peak_amps      = []
    auto_peak_amps = []
    mode_bpms      = []

    for i in range(IMFs.shape[0]):
        fft_i   = np.fft.fft(IMFs[i])
        freqs_i = np.fft.fftfreq(len(IMFs[i]), d=SAMPLE_INTERVAL)
        mag_i   = np.abs(fft_i)

        mask_bpm = (freqs_i >= freq_lo) & (freqs_i <= freq_hi)
        if np.any(mask_bpm):
            pk_idx = np.argmax(mag_i[mask_bpm])
            peak_amps.append(np.max(mag_i[mask_bpm]))
            mode_bpms.append(freqs_i[mask_bpm][pk_idx] * 60.0)
        else:
            peak_amps.append(0.0)
            mode_bpms.append(None)

        mask_wide = (freqs_i >= IMF_AUTO_FMIN) & (freqs_i <= IMF_AUTO_FMAX)
        auto_peak_amps.append(np.max(mag_i[mask_wide]) if np.any(mask_wide) else 0.0)

    sorted_idxs   = np.argsort(auto_peak_amps)[::-1]
    top_n         = min(IMF_AUTO_N, IMFs.shape[0])
    selected_idxs = sorted(sorted_idxs[:top_n].tolist())

    if not selected_idxs:
        steps['pca_sel'] = (False, "no modes found")
        print_analysis_block(count, n, steps, imf_info)
        return
    steps['pca_sel'] = (True, f"modes {selected_idxs}  ({IMF_AUTO_FMIN}–{IMF_AUTO_FMAX} Hz peak)")

    max_amp = max(peak_amps) if peak_amps else 1.0
    for i in range(IMFs.shape[0]):
        ratio    = peak_amps[i] / max_amp if max_amp > 0 else 0.0
        imf_info.append((i, mode_bpms[i], ratio, i in selected_idxs))

    selected_imfs = IMFs[selected_idxs, :]

    # 4. PCA
    pca           = PCA(n_components=1)
    pc1           = pca.fit_transform(selected_imfs.T).flatten()
    explained_var = pca.explained_variance_ratio_[0]
    steps['pca']  = (explained_var >= PCA_VAR_THRESHOLD,
                     f"var={explained_var:.3f}  threshold={PCA_VAR_THRESHOLD}")

    # 5. Welch PSD + параболическая интерполяция
    nperseg    = max(len(pc1) // 2, 64)
    freqs, psd = welch(pc1, fs=fs, nperseg=nperseg, window='hann', scaling='density')
    magnitudes = psd

    mask = (freqs >= freq_lo) & (freqs <= freq_hi)
    if not np.any(mask):
        steps['fft'] = (False, "no peak in BPM range")
        print_analysis_block(count, n, steps, imf_info)
        return

    mask_idx     = np.where(mask)[0]
    peak_rel     = np.argmax(magnitudes[mask])
    peak_abs     = mask_idx[peak_rel]

    if 1 <= peak_abs < len(magnitudes) - 1:
        y1, y2, y3 = magnitudes[peak_abs-1], magnitudes[peak_abs], magnitudes[peak_abs+1]
        denom = 2 * (y1 - 2*y2 + y3)
        hr_freq = (np.interp(peak_abs + (y1 - y3) / denom, np.arange(len(freqs)), freqs)
                   if denom != 0 else freqs[peak_abs])
    else:
        hr_freq = freqs[peak_abs]

    raw_bpm      = hr_freq * 60
    steps['fft'] = (True, f"peak {hr_freq:.3f} Hz  →  {raw_bpm:.1f} BPM")

    # Вывод блока анализа
    print_analysis_block(count, n, steps, imf_info)

    if explained_var < PCA_VAR_THRESHOLD:
        print_low_var(explained_var)
        return

    # Медианное сглаживание
    if MEDIAN_WINDOW > 0 and last_bpms is not None:
        last_bpms.append(raw_bpm)
        median_bpm = float(np.median(last_bpms))
        n_in_med   = len(last_bpms)
    else:
        median_bpm = raw_bpm
        n_in_med   = 1

    print_bpm_result(median_bpm, raw_bpm, explained_var, n_in_med)

    if last_bpms:
        print_history(list(last_bpms))

# ─────────────────────── Планировщик ─────────────────────────────────────────

async def periodic_processing():
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        while True:
            await asyncio.sleep(0.5)

            if not client_connected:
                continue

            cur = len(rssi_buffer)
            print_progress(cur, BUFFER_SIZE, last_rssi)

            if cur >= BUFFER_SIZE:
                data_copy = np.array(list(rssi_buffer)[:BUFFER_SIZE])
                rssi_buffer.clear()
                timestamps.clear()
                print()
                await loop.run_in_executor(pool, process_data, data_copy)

async def main():
    print_banner()
    async with websockets.serve(handler, "0.0.0.0", 5000):
        await asyncio.gather(
            asyncio.Future(),
            periodic_processing()
        )

if __name__ == "__main__":
    asyncio.run(main())