import streamlit as st
from scipy import signal
import pandas as pd
import numpy as np
import scipy
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def butterworth_filter(data, cutoff, fs, order=4, btype='low'):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return filtfilt(b, a, data)


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Filtro Butterworth passa-banda com fase zero.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    if not (0 < low < high < 1):
        raise ValueError(
            "As frequências do passa-banda devem obedecer: "
            "0 < baixa < alta < Nyquist."
        )

    b, a = butter(
        order,
        [low, high],
        btype="bandpass",
        analog=False
    )

    return filtfilt(b, a, data)


def read_sensor_file(uploaded_file):
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, sep=";", dtype=str)
    if df.shape[1] < 4:
        raise ValueError("O arquivo deve possuir pelo menos quatro colunas.")
    df = df.iloc[:, :4].copy()
    df.columns = ["Tempo", "X", "Y", "Z"]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().copy()
    df = df.sort_values("Tempo")
    df = df.drop_duplicates(subset="Tempo", keep="first")
    df = df.reset_index(drop=True)
    if len(df) < 20:
        raise ValueError("Poucos dados válidos no arquivo.")
    return df


def preprocess_accelerometer(df, fs_target=100, filter_cutoff=1.0, filter_order=4):
    time = df["Tempo"].to_numpy(float)
    x = df["X"].to_numpy(float)
    y = df["Y"].to_numpy(float)
    z = df["Z"].to_numpy(float)

    if (
        np.max(np.abs(x)) > 9 or
        np.max(np.abs(y)) > 9 or
        np.max(np.abs(z)) > 9
    ):
        x = x / 9.81
        y = y / 9.81
        z = z / 9.81
        unit = "g"
    else:
        unit = "unidade original"

    x = signal.detrend(x)
    y = signal.detrend(y)
    z = signal.detrend(z)

    time_uniform = np.arange(
        start=time[0],
        stop=time[-1],
        step=1000 / fs_target
    )

    x = scipy.interpolate.interp1d(time, x, kind="linear")(time_uniform)
    y = scipy.interpolate.interp1d(time, y, kind="linear")(time_uniform)
    z = scipy.interpolate.interp1d(time, z, kind="linear")(time_uniform)

    t = (time_uniform - time_uniform[0]) / 1000.0

    norm = np.sqrt(x**2 + y**2 + z**2)

    norm_filtered = butterworth_filter(
        norm,
        cutoff=filter_cutoff,
        fs=fs_target,
        order=filter_order,
        btype="low"
    )

    return {
        "t": t,
        "norm_filtered": norm_filtered,
        "unit": unit,
        "filter_cutoff": filter_cutoff
    }


def kmeans_states(values, n_states=5, random_state=42):
    values = np.asarray(values).reshape(-1, 1)
    km = KMeans(n_clusters=n_states, random_state=random_state, n_init=20)
    labels_raw = km.fit_predict(values)
    centers_raw = km.cluster_centers_.ravel()
    order = np.argsort(centers_raw)
    mapping = {old_label: new_label for new_label, old_label in enumerate(order)}
    states = np.array([mapping[label] for label in labels_raw], dtype=int)
    centers = centers_raw[order]
    return states, centers


def identify_baseline(t, states, baseline_seconds=2.0, n_states=5):
    mask = t < baseline_seconds
    counts = np.bincount(states[mask], minlength=n_states)
    baseline_state = int(np.argmax(counts))
    return baseline_state, counts


def detect_start(t, states, baseline_state, baseline_seconds=2.0, sequence_length=5):
    first = max(1, np.searchsorted(t, baseline_seconds, side="left"))
    last = len(states) - sequence_length + 1
    for i in range(first, last):
        if (
            states[i - 1] == baseline_state
            and np.all(states[i:i + sequence_length] > baseline_state)
        ):
            return i
    return None


def detect_end(states, baseline_state, start_index, sequence_length=5):
    if start_index is None:
        return None
    first = start_index + sequence_length
    last = len(states) - sequence_length + 1
    for i in range(first, last):
        if (
            states[i - 1] != baseline_state
            and np.all(states[i:i + sequence_length] == baseline_state)
        ):
            return i
    return None


def linear_crossing_time(t1, y1, t2, y2, threshold):
    if y2 == y1:
        return t1
    fraction = (threshold - y1) / (y2 - y1)
    return t1 + fraction * (t2 - t1)


def calculate_motion_features(t, signal_data, start_index, end_index, baseline_seconds=2.0):
    if start_index is None or end_index is None or end_index <= start_index:
        return None

    activity_signal = signal_data[start_index:end_index + 1]
    total_duration = t[end_index] - t[start_index]

    local_peak_idx = int(np.argmax(activity_signal))
    peak_index = start_index + local_peak_idx
    peak_amplitude = float(signal_data[peak_index])
    peak_time = float(t[peak_index])
    peak_latency = peak_time - float(t[start_index])

    baseline_mask = t < baseline_seconds
    baseline_value = float(np.mean(signal_data[baseline_mask]))

    threshold_90 = baseline_value + 0.90 * (peak_amplitude - baseline_value)

    pre90_time = None
    for i in range(start_index + 1, peak_index + 1):
        y1 = signal_data[i - 1]
        y2 = signal_data[i]
        if (y1 < threshold_90 <= y2) or (y1 > threshold_90 >= y2):
            pre90_time = linear_crossing_time(
                t[i - 1], y1, t[i], y2, threshold_90
            )
            break

    if pre90_time is None:
        pre_indices = np.arange(start_index, peak_index + 1)
        nearest_idx = pre_indices[
            np.argmin(np.abs(signal_data[pre_indices] - threshold_90))
        ]
        pre90_time = float(t[nearest_idx])

    acceleration_time = pre90_time - float(t[start_index])

    post90_time = None
    for i in range(peak_index + 1, end_index + 1):
        y1 = signal_data[i - 1]
        y2 = signal_data[i]
        if (y1 >= threshold_90 > y2) or (y1 <= threshold_90 < y2):
            post90_time = linear_crossing_time(
                t[i - 1], y1, t[i], y2, threshold_90
            )
            break

    if post90_time is None:
        post_indices = np.arange(peak_index, end_index + 1)
        nearest_idx = post_indices[
            np.argmin(np.abs(signal_data[post_indices] - threshold_90))
        ]
        post90_time = float(t[nearest_idx])

    deceleration_time = float(t[end_index]) - post90_time
    time_between_90 = post90_time - pre90_time

    return {
        "start_time": float(t[start_index]),
        "end_time": float(t[end_index]),
        "total_duration": float(total_duration),
        "baseline_value": baseline_value,
        "peak_time": peak_time,
        "peak_amplitude": peak_amplitude,
        "peak_latency": float(peak_latency),
        "threshold_90": float(threshold_90),
        "pre90_time": float(pre90_time),
        "acceleration_time": float(acceleration_time),
        "post90_time": float(post90_time),
        "deceleration_time": float(deceleration_time),
        "time_between_90": float(time_between_90)
    }



def calculate_steps(
    t,
    norm_signal,
    start_index,
    end_index,
    fs=100,
    lowcut=1.0,
    highcut=5.0,
    filter_order=4,
    min_peak_distance_s=0.30,
    prominence_factor=0.20
):
    """
    Filtra a norma euclidiana entre 1 e 5 Hz e detecta picos
    dentro do intervalo segmentado da atividade.

    prominence_factor é multiplicado pelo desvio-padrão do
    sinal passa-banda dentro da atividade.
    """
    band_signal = bandpass_filter(
        norm_signal,
        lowcut=lowcut,
        highcut=highcut,
        fs=fs,
        order=filter_order
    )

    if start_index is None or end_index is None or end_index <= start_index:
        return {
            "band_signal": band_signal,
            "peaks": np.array([], dtype=int),
            "n_steps": None,
            "cadence": None,
            "prominence_used": None
        }

    movement = band_signal[start_index:end_index + 1]

    min_distance_samples = max(
        1,
        int(round(min_peak_distance_s * fs))
    )

    movement_std = float(np.std(movement))
    prominence = max(
        np.finfo(float).eps,
        prominence_factor * movement_std
    )

    local_peaks, properties = find_peaks(
        movement,
        distance=min_distance_samples,
        prominence=prominence
    )

    peaks = local_peaks + start_index
    n_steps = int(len(peaks))

    duration = float(t[end_index] - t[start_index])

    cadence = (
        60.0 * n_steps / duration
        if duration > 0
        else np.nan
    )

    return {
        "band_signal": band_signal,
        "peaks": peaks,
        "n_steps": n_steps,
        "cadence": float(cadence),
        "prominence_used": float(prominence)
    }


st.set_page_config(page_title="iTUG - Características do Acelerômetro", layout="wide")
st.title("Segmentação e extração de características do acelerômetro")

st.sidebar.header("Parâmetros")
n_states = st.sidebar.number_input("Número de estados do K-means", 2, 10, 5, 1)
baseline_seconds = st.sidebar.number_input("Baseline inicial (s)", 0.5, 10.0, 2.0, 0.5)
sequence_length = st.sidebar.number_input("Amostras consecutivas", 1, 500, 5, 1)
filter_cutoff = st.sidebar.number_input("Frequência de corte do filtro (Hz)", 0.1, 20.0, 1.0, 0.1)
filter_order = st.sidebar.number_input("Ordem do filtro Butterworth", 1, 8, 4, 1)

st.sidebar.markdown("### Detecção de passos")
min_peak_distance_s = st.sidebar.number_input(
    "Distância mínima entre picos (s)",
    min_value=0.10,
    max_value=1.50,
    value=0.30,
    step=0.05
)
prominence_factor = st.sidebar.number_input(
    "Prominência mínima (× DP do sinal)",
    min_value=0.00,
    max_value=3.00,
    value=0.20,
    step=0.05
)

uploaded_acc = st.file_uploader("Arquivo do acelerômetro", type=["txt"])

if uploaded_acc is not None:
    try:
        df = read_sensor_file(uploaded_acc)
        acc = preprocess_accelerometer(
            df,
            fs_target=100,
            filter_cutoff=float(filter_cutoff),
            filter_order=int(filter_order)
        )

        t = acc["t"]
        norm_filtered = acc["norm_filtered"]

        states, centers = kmeans_states(norm_filtered, n_states=int(n_states))
        baseline_state, baseline_counts = identify_baseline(
            t, states,
            baseline_seconds=float(baseline_seconds),
            n_states=int(n_states)
        )

        start_index = detect_start(
            t, states, baseline_state,
            baseline_seconds=float(baseline_seconds),
            sequence_length=int(sequence_length)
        )

        end_index = detect_end(
            states, baseline_state, start_index,
            sequence_length=int(sequence_length)
        )

        features = calculate_motion_features(
            t, norm_filtered, start_index, end_index,
            baseline_seconds=float(baseline_seconds)
        )

        # Velocidade média para percurso conhecido de 4 m
        walking_speed = None
        if features is not None and features["total_duration"] > 0:
            walking_speed = 4.0 / features["total_duration"]

        # Para detecção de passos usamos a norma euclidiana antes
        # do passa-baixa principal, filtrada separadamente entre 1 e 5 Hz.
        # Reconstrói a norma interpolada sem o filtro principal.
        # Como preprocess_accelerometer retorna apenas a norma filtrada,
        # reutilizamos os dados originais para obter a norma de trabalho.
        time_raw = df["Tempo"].to_numpy(float)
        x_raw = df["X"].to_numpy(float)
        y_raw = df["Y"].to_numpy(float)
        z_raw = df["Z"].to_numpy(float)

        if (
            np.max(np.abs(x_raw)) > 9 or
            np.max(np.abs(y_raw)) > 9 or
            np.max(np.abs(z_raw)) > 9
        ):
            x_raw = x_raw / 9.81
            y_raw = y_raw / 9.81
            z_raw = z_raw / 9.81

        x_raw = signal.detrend(x_raw)
        y_raw = signal.detrend(y_raw)
        z_raw = signal.detrend(z_raw)

        time_uniform_steps = np.arange(
            start=time_raw[0],
            stop=time_raw[-1],
            step=10
        )

        x_steps = scipy.interpolate.interp1d(
            time_raw, x_raw, kind="linear"
        )(time_uniform_steps)
        y_steps = scipy.interpolate.interp1d(
            time_raw, y_raw, kind="linear"
        )(time_uniform_steps)
        z_steps = scipy.interpolate.interp1d(
            time_raw, z_raw, kind="linear"
        )(time_uniform_steps)

        norm_for_steps = np.sqrt(
            x_steps**2 + y_steps**2 + z_steps**2
        )

        step_results = calculate_steps(
            t,
            norm_for_steps,
            start_index,
            end_index,
            fs=100,
            lowcut=1.0,
            highcut=5.0,
            filter_order=int(filter_order),
            min_peak_distance_s=float(min_peak_distance_s),
            prominence_factor=float(prominence_factor)
        )

        st.header("Segmentação")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Estado de baseline", baseline_state)
        c2.metric("Início", f"{t[start_index]:.3f} s" if start_index is not None else "Não encontrado")
        c3.metric("Fim", f"{t[end_index]:.3f} s" if end_index is not None else "Não encontrado")
        c4.metric("Duração total", f"{features['total_duration']:.3f} s" if features else "-")

        if features is not None:
            st.header("Características extraídas")
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Duração total", f"{features['total_duration']:.3f} s")
            f2.metric("Velocidade de marcha", f"{walking_speed:.3f} m/s")
            f3.metric("Pico de amplitude", f"{features['peak_amplitude']:.4f} {acc['unit']}")
            f4.metric("Latência do pico", f"{features['peak_latency']:.3f} s")

            f5, f6, f7, f8 = st.columns(4)
            f5.metric("Tempo de aceleração", f"{features['acceleration_time']:.3f} s")
            f6.metric("Tempo entre 90%", f"{features['time_between_90']:.3f} s")
            f7.metric("Tempo de desaceleração", f"{features['deceleration_time']:.3f} s")
            f8.metric(
                "Passos estimados",
                str(step_results["n_steps"]) if step_results["n_steps"] is not None else "-"
            )

            if step_results["cadence"] is not None:
                st.caption(
                    f"Cadência estimada: **{step_results['cadence']:.1f} passos/min** "
                    f"(picos detectados no sinal passa-banda de 1–5 Hz)."
                )

            st.subheader("Sinal segmentado e eventos temporais")
            fig, ax = plt.subplots(figsize=(13, 5))
            ax.plot(
                t, norm_filtered, "k", linewidth=1.5,
                label=f"Norma filtrada - {filter_cutoff:.1f} Hz"
            )

            ax.axvspan(0, baseline_seconds, alpha=0.10, label="Baseline inicial")
            ax.axvline(features["start_time"], linestyle="--", linewidth=2, label=f"Início = {features['start_time']:.2f} s")
            ax.axvline(features["pre90_time"], linestyle="--", linewidth=1.5, label=f"90% pré-pico = {features['pre90_time']:.2f} s")
            ax.axvline(features["peak_time"], linestyle="-.", linewidth=2, label=f"Pico = {features['peak_time']:.2f} s")
            ax.axvline(features["post90_time"], linestyle="--", linewidth=1.5, label=f"90% pós-pico = {features['post90_time']:.2f} s")
            ax.axvline(features["end_time"], linestyle=":", linewidth=2, label=f"Fim = {features['end_time']:.2f} s")
            ax.axhline(features["threshold_90"], linestyle=":", linewidth=1.2, label=f"Limiar 90% = {features['threshold_90']:.4f}")
            ax.scatter([features["peak_time"]], [features["peak_amplitude"]], s=60, zorder=5)

            ax.set_xlabel("Tempo (s)")
            ax.set_ylabel(f"Norma da aceleração ({acc['unit']})")
            ax.set_title("Norma euclidiana do acelerômetro com eventos detectados")
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=9)
            st.pyplot(fig)

            # =================================================
            # SEGUNDO GRÁFICO: PASSA-BANDA 1–5 Hz E PICOS
            # =================================================
            st.subheader("Sinal passa-banda de 1–5 Hz e picos detectados")

            fig_steps, ax_steps = plt.subplots(figsize=(13, 4.5))

            band_signal = step_results["band_signal"]
            peaks = step_results["peaks"]

            ax_steps.plot(
                t,
                band_signal,
                linewidth=1.2,
                label="Norma euclidiana passa-banda 1–5 Hz"
            )

            ax_steps.axvspan(
                features["start_time"],
                features["end_time"],
                alpha=0.10,
                label="Intervalo do movimento"
            )

            ax_steps.axvline(
                features["start_time"],
                linestyle="--",
                linewidth=1.5,
                label="Início"
            )

            ax_steps.axvline(
                features["end_time"],
                linestyle=":",
                linewidth=1.5,
                label="Fim"
            )

            if len(peaks) > 0:
                ax_steps.scatter(
                    t[peaks],
                    band_signal[peaks],
                    s=45,
                    zorder=5,
                    label=f"Picos detectados = {len(peaks)}"
                )

            ax_steps.set_xlabel("Tempo (s)")
            ax_steps.set_ylabel(f"Aceleração filtrada ({acc['unit']})")
            ax_steps.set_title(
                "Detecção de passos pela contagem de picos no sinal de 1–5 Hz"
            )
            ax_steps.grid(alpha=0.3)
            ax_steps.legend(loc="best", fontsize=9)

            st.pyplot(fig_steps)

            st.caption(
                "A contagem de picos é uma estimativa do número de passos. "
                "Os parâmetros de distância mínima entre picos e prominência "
                "podem ser ajustados na barra lateral."
            )

            features_df = pd.DataFrame({
                "Característica": [
                    "Duração total",
                    "Velocidade de marcha",
                    "Número de passos estimado",
                    "Cadência estimada",
                    "Pico de amplitude",
                    "Latência do pico",
                    "Tempo de aceleração",
                    "Tempo de desaceleração",
                    "Tempo entre 90% pré e pós-pico",
                    "Baseline média",
                    "Limiar de 90%",
                    "Tempo do início",
                    "Tempo 90% pré-pico",
                    "Tempo do pico",
                    "Tempo 90% pós-pico",
                    "Tempo do fim"
                ],
                "Valor": [
                    features["total_duration"],
                    walking_speed,
                    step_results["n_steps"],
                    step_results["cadence"],
                    features["peak_amplitude"],
                    features["peak_latency"],
                    features["acceleration_time"],
                    features["deceleration_time"],
                    features["time_between_90"],
                    features["baseline_value"],
                    features["threshold_90"],
                    features["start_time"],
                    features["pre90_time"],
                    features["peak_time"],
                    features["post90_time"],
                    features["end_time"]
                ],
                "Unidade": [
                    "s", "m/s", "passos", "passos/min", acc["unit"], "s", "s", "s", "s", acc["unit"],
                    acc["unit"], "s", "s", "s", "s", "s"
                ]
            })

            st.subheader("Tabela de características")
            st.dataframe(features_df, hide_index=True, use_container_width=True)

            csv_features = features_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Baixar características",
                data=csv_features,
                file_name="caracteristicas_acelerometro.csv",
                mime="text/csv"
            )

        else:
            st.warning(
                "Não foi possível extrair as características porque "
                "o início e/ou o final da atividade não foram identificados."
            )

        st.subheader("Estados do K-means")
        fig2, ax2 = plt.subplots(figsize=(12, 3.2))
        ax2.step(t, states, where="post", linewidth=1)
        ax2.axhline(
            baseline_state,
            linestyle="--",
            linewidth=1.5,
            label=f"Baseline = estado {baseline_state}"
        )
        if start_index is not None:
            ax2.axvline(t[start_index], linestyle="--", linewidth=2)
        if end_index is not None:
            ax2.axvline(t[end_index], linestyle=":", linewidth=2)
        ax2.set_xlabel("Tempo (s)")
        ax2.set_ylabel("Estado")
        ax2.set_yticks(np.arange(int(n_states)))
        ax2.grid(alpha=0.3)
        ax2.legend()
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Erro durante o processamento: {e}")
