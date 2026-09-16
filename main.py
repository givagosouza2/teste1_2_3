import streamlit as st
from scipy import signal
import pandas as pd
import numpy as np
import scipy
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# ============================================================
# FUNÇÕES GERAIS
# ============================================================

def butterworth_filter(data, cutoff, fs, order=4, btype='low'):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return filtfilt(b, a, data)


def read_sensor_file(uploaded_file, sensor="acc"):
    uploaded_file.seek(0)

    df = pd.read_csv(
        uploaded_file,
        sep=";",
        dtype=str
    )

    if df.shape[1] < 4:
        raise ValueError("O arquivo deve possuir pelo menos quatro colunas.")

    df = df.iloc[:, :4].copy()

    if sensor == "acc":
        df.columns = ["Tempo", "X", "Y", "Z"]
    else:
        df.columns = ["Tempo", "X", "Y", "Z"]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove cabeçalhos repetidos ou linhas inválidas
    df = df.dropna().copy()
    df = df.sort_values("Tempo")
    df = df.drop_duplicates(subset="Tempo", keep="first")
    df = df.reset_index(drop=True)

    if len(df) < 20:
        raise ValueError("Poucos dados válidos no arquivo.")

    return df


def preprocess_sensor(df, sensor="acc", fs_target=100, filter_cutoff=1.0, filter_order=4):
    """
    Detrend + interpolação 100 Hz + norma euclidiana + filtro 1 Hz.

    Acelerômetro:
      normaliza por 9,81 quando os dados parecem estar em m/s².

    Giroscópio:
      não realiza normalização por gravidade.
    """
    time = df["Tempo"].to_numpy(float)
    x = df["X"].to_numpy(float)
    y = df["Y"].to_numpy(float)
    z = df["Z"].to_numpy(float)

    if sensor == "acc":
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
    else:
        unit = "velocidade angular"

    # detrend
    x = signal.detrend(x)
    y = signal.detrend(y)
    z = signal.detrend(z)

    # interpolação a 100 Hz -> 10 ms
    time_uniform = np.arange(
        start=time[0],
        stop=time[-1],
        step=1000 / fs_target
    )

    x = scipy.interpolate.interp1d(
        time, x, kind="linear"
    )(time_uniform)

    y = scipy.interpolate.interp1d(
        time, y, kind="linear"
    )(time_uniform)

    z = scipy.interpolate.interp1d(
        time, z, kind="linear"
    )(time_uniform)

    t = (time_uniform - time_uniform[0]) / 1000.0

    # norma euclidiana
    norm = np.sqrt(
        x**2 +
        y**2 +
        z**2
    )

    # filtro passa-baixa ajustável
    norm_filtered = butterworth_filter(
        norm,
        cutoff=filter_cutoff,
        fs=fs_target,
        order=filter_order,
        btype="low"
    )

    return {
        "t": t,
        "x": x,
        "y": y,
        "z": z,
        "norm": norm,
        "norm_filtered": norm_filtered,
        "unit": unit,
        "filter_cutoff": filter_cutoff,
        "filter_order": filter_order
    }


def kmeans_states(values, n_states=5, random_state=42):
    values = np.asarray(values).reshape(-1, 1)

    km = KMeans(
        n_clusters=n_states,
        random_state=random_state,
        n_init=20
    )

    labels_raw = km.fit_predict(values)
    centers_raw = km.cluster_centers_.ravel()

    # ordena os estados pela magnitude dos centróides
    order = np.argsort(centers_raw)

    mapping = {
        old_label: new_label
        for new_label, old_label in enumerate(order)
    }

    states = np.array(
        [mapping[label] for label in labels_raw],
        dtype=int
    )

    centers = centers_raw[order]

    return states, centers


def identify_baseline(
    t,
    states,
    baseline_seconds=2.0,
    n_states=5
):
    mask = t < baseline_seconds

    counts = np.bincount(
        states[mask],
        minlength=n_states
    )

    baseline_state = int(
        np.argmax(counts)
    )

    return baseline_state, counts


def detect_start(
    t,
    states,
    baseline_state,
    baseline_seconds=2.0,
    sequence_length=5
):
    first = max(
        1,
        np.searchsorted(
            t,
            baseline_seconds,
            side="left"
        )
    )

    last = (
        len(states) -
        sequence_length +
        1
    )

    for i in range(first, last):

        if (
            states[i - 1] == baseline_state
            and
            np.all(
                states[
                    i:i + sequence_length
                ] > baseline_state
            )
        ):
            return i

    return None


def detect_end(
    states,
    baseline_state,
    start_index,
    sequence_length=5
):
    if start_index is None:
        return None

    first = (
        start_index +
        sequence_length
    )

    last = (
        len(states) -
        sequence_length +
        1
    )

    for i in range(first, last):

        if (
            states[i - 1] != baseline_state
            and
            np.all(
                states[
                    i:i + sequence_length
                ] == baseline_state
            )
        ):
            return i

    return None


def segment_signal(
    t,
    signal_1hz,
    n_states=5,
    baseline_seconds=2.0,
    sequence_length=5
):
    states, centers = kmeans_states(
        signal_1hz,
        n_states=n_states
    )

    baseline_state, counts = identify_baseline(
        t,
        states,
        baseline_seconds=baseline_seconds,
        n_states=n_states
    )

    start_index = detect_start(
        t,
        states,
        baseline_state,
        baseline_seconds=baseline_seconds,
        sequence_length=sequence_length
    )

    end_index = detect_end(
        states,
        baseline_state,
        start_index,
        sequence_length=sequence_length
    )

    return {
        "states": states,
        "centers": centers,
        "baseline_state": baseline_state,
        "baseline_counts": counts,
        "start_index": start_index,
        "end_index": end_index
    }


def transition_matrix(states, n_states=5):
    matrix = np.zeros(
        (n_states, n_states),
        dtype=float
    )

    for a, b in zip(
        states[:-1],
        states[1:]
    ):
        matrix[a, b] += 1

    row_sum = matrix.sum(
        axis=1,
        keepdims=True
    )

    return np.divide(
        matrix,
        row_sum,
        out=np.zeros_like(matrix),
        where=row_sum != 0
    )


def show_results(
    title,
    processed,
    segmentation,
    baseline_seconds,
    sequence_length,
    ylabel
):
    t = processed["t"]
    signal_1hz = processed["norm_filtered"]

    states = segmentation["states"]
    centers = segmentation["centers"]
    baseline_state = segmentation["baseline_state"]
    counts = segmentation["baseline_counts"]
    start_index = segmentation["start_index"]
    end_index = segmentation["end_index"]

    st.header(title)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Estado de baseline",
        baseline_state
    )

    if start_index is not None:
        start_time = float(
            t[start_index]
        )
        c2.metric(
            "Início",
            f"{start_time:.3f} s"
        )
    else:
        start_time = None
        c2.metric(
            "Início",
            "Não encontrado"
        )

    if end_index is not None:
        end_time = float(
            t[end_index]
        )
        c3.metric(
            "Fim",
            f"{end_time:.3f} s"
        )
    else:
        end_time = None
        c3.metric(
            "Fim",
            "Não encontrado"
        )

    if (
        start_time is not None and
        end_time is not None
    ):
        duration = end_time - start_time
        c4.metric(
            "Duração",
            f"{duration:.3f} s"
        )
    else:
        duration = None
        c4.metric(
            "Duração",
            "-"
        )

    # --------------------------------------------------------
    # GRÁFICO PRINCIPAL
    # --------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(12, 4.5)
    )

    ax.plot(
        t,
        signal_1hz,
        "k",
        linewidth=1.5,
        label="Norma euclidiana - 1 Hz"
    )

    ax.axvspan(
        0,
        baseline_seconds,
        alpha=0.15,
        label="Baseline inicial"
    )

    if start_index is not None:
        ax.axvline(
            t[start_index],
            linestyle="--",
            linewidth=2,
            label=f"Início = {t[start_index]:.2f} s"
        )

    if end_index is not None:
        ax.axvline(
            t[end_index],
            linestyle=":",
            linewidth=2,
            label=f"Fim = {t[end_index]:.2f} s"
        )

    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{title} — norma euclidiana filtrada em {processed['filter_cutoff']:.1f} Hz"
    )
    ax.grid(alpha=0.3)
    ax.legend()

    st.pyplot(fig)

    # --------------------------------------------------------
    # ESTADOS
    # --------------------------------------------------------
    fig_states, ax_states = plt.subplots(
        figsize=(12, 3.2)
    )

    ax_states.step(
        t,
        states,
        where="post",
        linewidth=1
    )

    ax_states.axhline(
        baseline_state,
        linestyle="--",
        linewidth=1.5,
        label=f"Baseline = estado {baseline_state}"
    )

    if start_index is not None:
        ax_states.axvline(
            t[start_index],
            linestyle="--",
            linewidth=2
        )

    if end_index is not None:
        ax_states.axvline(
            t[end_index],
            linestyle=":",
            linewidth=2
        )

    ax_states.set_xlabel("Tempo (s)")
    ax_states.set_ylabel("Estado")
    ax_states.set_yticks(
        np.arange(len(centers))
    )
    ax_states.set_title(
        f"Estados do K-means — {title}"
    )
    ax_states.grid(alpha=0.3)
    ax_states.legend()

    st.pyplot(fig_states)

    # --------------------------------------------------------
    # CENTRÓIDES
    # --------------------------------------------------------
    with st.expander(
        f"Centróides e baseline — {title}"
    ):

        centers_df = pd.DataFrame({
            "Estado": np.arange(
                len(centers)
            ),
            "Centróide": centers,
            "N na baseline": counts,
            "Estado baseline": [
                i == baseline_state
                for i in range(
                    len(centers)
                )
            ]
        })

        st.dataframe(
            centers_df,
            hide_index=True,
            use_container_width=True
        )

    # --------------------------------------------------------
    # MATRIZ DE TRANSIÇÃO
    # --------------------------------------------------------
    with st.expander(
        f"Matriz de transição — {title}"
    ):

        matrix = transition_matrix(
            states,
            n_states=len(centers)
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=[
                f"Estado {i}"
                for i in range(
                    len(centers)
                )
            ],
            columns=[
                f"→ {i}"
                for i in range(
                    len(centers)
                )
            ]
        )

        st.dataframe(
            matrix_df.style.format(
                "{:.3f}"
            ),
            use_container_width=True
        )

    return {
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration
    }


# ============================================================
# INTERFACE
# ============================================================

st.set_page_config(
    page_title="iTUG - Acelerômetro + Giroscópio",
    layout="wide"
)

st.title(
    "Segmentação iTUG — Acelerômetro e Giroscópio"
)

st.write(
    """
    O acelerômetro e o giroscópio são analisados separadamente.
    Para cada sensor é calculada a norma euclidiana, filtrada
    com uma frequência de corte ajustável, e a segmentação é realizada por K-means.
    """
)


# ============================================================
# PARÂMETROS
# ============================================================

st.sidebar.header(
    "Parâmetros"
)

n_states = st.sidebar.number_input(
    "Número de estados",
    min_value=2,
    max_value=10,
    value=5,
    step=1
)

baseline_seconds = st.sidebar.number_input(
    "Baseline inicial (s)",
    min_value=0.5,
    max_value=10.0,
    value=2.0,
    step=0.5
)

sequence_length = st.sidebar.number_input(
    "Amostras consecutivas",
    min_value=1,
    max_value=500,
    value=5,
    step=1
)

filter_cutoff = st.sidebar.number_input(
    "Frequência de corte do filtro passa-baixa (Hz)",
    min_value=0.1,
    max_value=20.0,
    value=1.0,
    step=0.1
)

filter_order = st.sidebar.number_input(
    "Ordem do filtro Butterworth",
    min_value=1,
    max_value=8,
    value=4,
    step=1
)


# ============================================================
# UPLOAD DOS DOIS SENSORES
# ============================================================

col_upload1, col_upload2 = st.columns(2)

with col_upload1:

    uploaded_acc = st.file_uploader(
        "Arquivo do acelerômetro",
        type=["txt"],
        key="acc"
    )

with col_upload2:

    uploaded_gyro = st.file_uploader(
        "Arquivo do giroscópio",
        type=["txt"],
        key="gyro"
    )


if (
    uploaded_acc is not None and
    uploaded_gyro is not None
):

    try:

        # ====================================================
        # ACELERÔMETRO
        # ====================================================

        df_acc = read_sensor_file(
            uploaded_acc,
            sensor="acc",
            filter_cutoff=float(filter_cutoff),
            filter_order=int(filter_order)
        )

        acc = preprocess_sensor(
            df_acc,
            sensor="acc",
            filter_cutoff=float(filter_cutoff),
            filter_order=int(filter_order)
        )

        acc_seg = segment_signal(
            acc["t"],
            acc["norm_filtered"],
            n_states=int(n_states),
            baseline_seconds=float(
                baseline_seconds
            ),
            sequence_length=int(
                sequence_length
            )
        )


        # ====================================================
        # GIROSCÓPIO
        # ====================================================

        df_gyro = read_sensor_file(
            uploaded_gyro,
            sensor="gyro",
            filter_cutoff=float(filter_cutoff),
            filter_order=int(filter_order)
        )

        gyro = preprocess_sensor(
            df_gyro,
            sensor="gyro",
            filter_cutoff=float(filter_cutoff),
            filter_order=int(filter_order)
        )

        gyro_seg = segment_signal(
            gyro["t"],
            gyro["norm_filtered"],
            n_states=int(n_states),
            baseline_seconds=float(
                baseline_seconds
            ),
            sequence_length=int(
                sequence_length
            )
        )


        # ====================================================
        # RESULTADOS EM GRÁFICOS DIFERENTES
        # ====================================================

        acc_results = show_results(
            "Acelerômetro",
            acc,
            acc_seg,
            baseline_seconds=float(
                baseline_seconds
            ),
            sequence_length=int(
                sequence_length
            ),
            ylabel="Norma da aceleração"
        )

        st.divider()

        gyro_results = show_results(
            "Giroscópio",
            gyro,
            gyro_seg,
            baseline_seconds=float(
                baseline_seconds
            ),
            sequence_length=int(
                sequence_length
            ),
            ylabel="Norma da velocidade angular"
        )


        # ====================================================
        # COMPARAÇÃO DOS RESULTADOS
        # ====================================================

        st.divider()

        st.header(
            "Comparação entre sensores"
        )

        comparison_df = pd.DataFrame({
            "Sensor": [
                "Acelerômetro",
                "Giroscópio"
            ],
            "Início_s": [
                acc_results["start_time"],
                gyro_results["start_time"]
            ],
            "Fim_s": [
                acc_results["end_time"],
                gyro_results["end_time"]
            ],
            "Duração_s": [
                acc_results["duration"],
                gyro_results["duration"]
            ]
        })

        st.dataframe(
            comparison_df,
            hide_index=True,
            use_container_width=True
        )


        # ====================================================
        # DADOS PROCESSADOS PARA DOWNLOAD
        # ====================================================

        acc_df_out = pd.DataFrame({
            "Tempo_s": acc["t"],
            "Norma_ACC_1Hz": acc[
                "norm_1hz"
            ],
            "Estado_ACC": acc_seg[
                "states"
            ]
        })

        gyro_df_out = pd.DataFrame({
            "Tempo_s": gyro["t"],
            "Norma_GYRO_1Hz": gyro[
                "norm_1hz"
            ],
            "Estado_GYRO": gyro_seg[
                "states"
            ]
        })


        col_down1, col_down2 = st.columns(2)

        with col_down1:

            st.download_button(
                "Baixar acelerômetro processado",
                data=acc_df_out.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=(
                    "acelerometro_segmentado.csv"
                ),
                mime="text/csv"
            )

        with col_down2:

            st.download_button(
                "Baixar giroscópio processado",
                data=gyro_df_out.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=(
                    "giroscopio_segmentado.csv"
                ),
                mime="text/csv"
            )


    except Exception as e:

        st.error(
            f"Erro durante o processamento: {e}"
        )


else:

    st.info(
        "Carregue os arquivos do acelerômetro e do giroscópio para iniciar a análise."
    )
