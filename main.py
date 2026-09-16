import streamlit as st
from scipy import signal
import pandas as pd
import numpy as np
import scipy
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# ============================================================
# FUNÇÕES
# ============================================================

def butterworth_filter(data, cutoff, fs, order=4, btype='low'):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype, analog=False)
    y = filtfilt(b, a, data)
    return y


def kmeans_states(signal_data, n_states=5, random_state=42):
    """
    Aplica K-means unidimensional e reorganiza os estados
    de acordo com a magnitude crescente dos centróides.

    Estado 0 = menor centróide
    Estado 1 = segundo menor
    ...
    """
    values = np.asarray(signal_data).reshape(-1, 1)

    model = KMeans(
        n_clusters=n_states,
        random_state=random_state,
        n_init=20
    )

    labels_raw = model.fit_predict(values)
    centers_raw = model.cluster_centers_.ravel()

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


def baseline_state_detection(t, states, baseline_seconds=2.0, n_states=5):
    """
    Estado de baseline = estado predominante nos
    primeiros baseline_seconds segundos.
    """
    mask = t < baseline_seconds

    if not np.any(mask):
        return None, None

    counts = np.bincount(
        states[mask],
        minlength=n_states
    )

    baseline_state = int(np.argmax(counts))

    return baseline_state, counts


def detect_start(
    t,
    states,
    baseline_state,
    baseline_seconds=2.0,
    sequence_length=5
):
    """
    Início:
    - busca após os primeiros 2 s;
    - amostra anterior no estado baseline;
    - sequência de N amostras em qualquer estado
      numericamente superior ao baseline.
    """

    first_search_index = np.searchsorted(
        t,
        baseline_seconds,
        side="left"
    )

    first_search_index = max(1, first_search_index)

    last_possible = len(states) - sequence_length + 1

    for i in range(first_search_index, last_possible):

        previous_is_baseline = (
            states[i - 1] == baseline_state
        )

        sequence_is_above = np.all(
            states[i:i + sequence_length] > baseline_state
        )

        if previous_is_baseline and sequence_is_above:
            return i

    return None


def detect_end(
    states,
    baseline_state,
    start_index,
    sequence_length=5
):
    """
    Final:
    após o início, identifica a primeira volta ao estado
    de baseline por N amostras consecutivas.
    """

    if start_index is None:
        return None

    first_search_index = start_index + sequence_length
    last_possible = len(states) - sequence_length + 1

    for i in range(first_search_index, last_possible):

        previous_is_not_baseline = (
            states[i - 1] != baseline_state
        )

        sequence_is_baseline = np.all(
            states[i:i + sequence_length] == baseline_state
        )

        if previous_is_not_baseline and sequence_is_baseline:
            return i

    return None


def transition_matrix(states, n_states=5):
    """
    Matriz descritiva de transição entre estados.
    """
    matrix = np.zeros((n_states, n_states), dtype=float)

    for a, b in zip(states[:-1], states[1:]):
        matrix[a, b] += 1

    row_sum = matrix.sum(axis=1, keepdims=True)

    matrix = np.divide(
        matrix,
        row_sum,
        out=np.zeros_like(matrix),
        where=row_sum != 0
    )

    return matrix


# ============================================================
# INTERFACE
# ============================================================

st.set_page_config(
    page_title="Segmentação iTUG",
    layout="wide"
)

st.title("Segmentação automática do sinal de acelerometria")

st.write(
    """
    O sinal é submetido a detrend, normalização em relação à gravidade
    quando necessário, interpolação para 100 Hz, cálculo da norma
    euclidiana, filtragem e segmentação por K-means.
    """
)


# ============================================================
# PARÂMETROS
# ============================================================

st.sidebar.header("Parâmetros da segmentação")

n_states = st.sidebar.number_input(
    "Número de estados do K-means",
    min_value=2,
    max_value=10,
    value=5,
    step=1
)

baseline_seconds = st.sidebar.number_input(
    "Duração da baseline inicial (s)",
    min_value=0.5,
    max_value=10.0,
    value=2.0,
    step=0.5
)

sequence_length = st.sidebar.number_input(
    "Número de amostras consecutivas",
    min_value=1,
    max_value=500,
    value=5,
    step=1
)

segmentation_filter = st.sidebar.selectbox(
    "Filtro usado na segmentação",
    [1.0, 4.0, 10.0],
    index=2,
    format_func=lambda x: f"{x:g} Hz"
)


# ============================================================
# UPLOAD
# ============================================================

uploaded_acc_iTUG = st.file_uploader(
    "Carregue o arquivo de texto do acelerômetro",
    type=["txt"]
)


if uploaded_acc_iTUG is not None:

    # ========================================================
    # LEITURA
    # ========================================================

    custom_separator = ';'

    df = pd.read_csv(
        uploaded_acc_iTUG,
        sep=custom_separator,
        dtype=str
    )

    # Mantém as quatro primeiras colunas
    df = df.iloc[:, 0:4].copy()

    df.columns = [
        "Tempo",
        "Acc_X",
        "Acc_Y",
        "Acc_Z"
    ]

    # Converte em números e remove possíveis cabeçalhos repetidos
    for column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna().copy()

    df = df.sort_values("Tempo")
    df = df.drop_duplicates(
        subset="Tempo",
        keep="first"
    )

    df = df.reset_index(drop=True)

    if len(df) < 20:
        st.error("Poucos dados válidos no arquivo.")
        st.stop()

    time = df["Tempo"].to_numpy(dtype=float)
    x = df["Acc_X"].to_numpy(dtype=float)
    y = df["Acc_Y"].to_numpy(dtype=float)
    z = df["Acc_Z"].to_numpy(dtype=float)


    # ========================================================
    # PRÉ-PROCESSAMENTO ORIGINAL
    # ========================================================

    # Normalização para g quando os dados parecem estar em m/s²
    if (
        np.max(np.abs(x)) > 9 or
        np.max(np.abs(y)) > 9 or
        np.max(np.abs(z)) > 9
    ):
        x = signal.detrend(x / 9.81)
        y = signal.detrend(y / 9.81)
        z = signal.detrend(z / 9.81)

        unidade = "g"

    else:
        x = signal.detrend(x)
        y = signal.detrend(y)
        z = signal.detrend(z)

        unidade = "unidade original"


    # ========================================================
    # INTERPOLAÇÃO PARA 100 Hz
    # ========================================================

    time_ = np.arange(
        start=time[0],
        stop=time[-1],
        step=10
    )

    interpf = scipy.interpolate.interp1d(
        time,
        x,
        kind="linear"
    )
    x = interpf(time_)

    interpf = scipy.interpolate.interp1d(
        time,
        y,
        kind="linear"
    )
    y = interpf(time_)

    interpf = scipy.interpolate.interp1d(
        time,
        z,
        kind="linear"
    )
    z = interpf(time_)

    # tempo em segundos, começando em zero
    t = (time_ - time_[0]) / 1000


    # ========================================================
    # NORMA EUCLIDIANA
    # ========================================================

    norm_waveform = np.sqrt(
        x**2 +
        y**2 +
        z**2
    )


    # ========================================================
    # FILTRAGEM EM 1, 4 E 10 Hz
    # ========================================================

    fs = 100

    norm_1 = butterworth_filter(
        norm_waveform,
        1,
        fs,
        order=4,
        btype='low'
    )

    norm_4 = butterworth_filter(
        norm_waveform,
        4,
        fs,
        order=4,
        btype='low'
    )

    norm_10 = butterworth_filter(
        norm_waveform,
        10,
        fs,
        order=4,
        btype='low'
    )

    filtered_signals = {
        1.0: norm_1,
        4.0: norm_4,
        10.0: norm_10
    }

    norm_segment = filtered_signals[
        segmentation_filter
    ]


    # ========================================================
    # K-MEANS
    # ========================================================

    states, centers = kmeans_states(
        norm_segment,
        n_states=int(n_states)
    )


    # ========================================================
    # BASELINE
    # ========================================================

    baseline_state, baseline_counts = baseline_state_detection(
        t,
        states,
        baseline_seconds=float(baseline_seconds),
        n_states=int(n_states)
    )


    # ========================================================
    # INÍCIO
    # ========================================================

    start_index = detect_start(
        t,
        states,
        baseline_state,
        baseline_seconds=float(baseline_seconds),
        sequence_length=int(sequence_length)
    )


    # ========================================================
    # FINAL
    # ========================================================

    end_index = detect_end(
        states,
        baseline_state,
        start_index,
        sequence_length=int(sequence_length)
    )


    # ========================================================
    # RESULTADOS
    # ========================================================

    st.subheader("Resultados")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Estado de baseline",
        baseline_state
    )

    if start_index is not None:
        start_time = t[start_index]

        col2.metric(
            "Início",
            f"{start_time:.3f} s"
        )
    else:
        start_time = None

        col2.metric(
            "Início",
            "Não encontrado"
        )

    if end_index is not None:
        end_time = t[end_index]

        col3.metric(
            "Final",
            f"{end_time:.3f} s"
        )
    else:
        end_time = None

        col3.metric(
            "Final",
            "Não encontrado"
        )

    if (
        start_time is not None and
        end_time is not None
    ):
        activity_time = (
            end_time -
            start_time
        )

        col4.metric(
            "Tempo da atividade",
            f"{activity_time:.3f} s"
        )

    else:
        activity_time = None

        col4.metric(
            "Tempo da atividade",
            "-"
        )


    # ========================================================
    # GRÁFICO DOS FILTROS
    # ========================================================

    st.subheader(
        "Norma euclidiana filtrada"
    )

    fig1, ax1 = plt.subplots(
        figsize=(12, 4)
    )

    ax1.plot(
        t,
        norm_1,
        label="1 Hz"
    )

    ax1.plot(
        t,
        norm_4,
        label="4 Hz"
    )

    ax1.plot(
        t,
        norm_10,
        label="10 Hz"
    )

    ax1.set_xlabel(
        "Tempo (s)"
    )

    ax1.set_ylabel(
        f"Norma da aceleração ({unidade})"
    )

    ax1.legend()

    ax1.grid(
        alpha=0.3
    )

    st.pyplot(fig1)


    # ========================================================
    # GRÁFICO DE SEGMENTAÇÃO
    # ========================================================

    st.subheader(
        "Segmentação automática"
    )

    fig2, ax2 = plt.subplots(
        figsize=(12, 5)
    )

    ax2.plot(
        t,
        norm_segment,
        'k',
        linewidth=1.2,
        label=(
            f"Norma - "
            f"{segmentation_filter:g} Hz"
        )
    )

    # janela da baseline
    ax2.axvspan(
        0,
        baseline_seconds,
        alpha=0.15,
        label="Baseline inicial"
    )

    if start_index is not None:

        ax2.axvline(
            t[start_index],
            linestyle='--',
            linewidth=2,
            label=(
                f"Início = "
                f"{t[start_index]:.2f} s"
            )
        )

    if end_index is not None:

        ax2.axvline(
            t[end_index],
            linestyle=':',
            linewidth=2,
            label=(
                f"Fim = "
                f"{t[end_index]:.2f} s"
            )
        )

    ax2.set_xlabel(
        "Tempo (s)"
    )

    ax2.set_ylabel(
        f"Norma da aceleração ({unidade})"
    )

    ax2.legend()

    ax2.grid(
        alpha=0.3
    )

    st.pyplot(fig2)


    # ========================================================
    # GRÁFICO DOS ESTADOS
    # ========================================================

    st.subheader(
        "Estados do K-means ao longo do tempo"
    )

    fig3, ax3 = plt.subplots(
        figsize=(12, 3.5)
    )

    ax3.step(
        t,
        states,
        where="post",
        linewidth=1
    )

    ax3.axhline(
        baseline_state,
        linestyle='--',
        linewidth=1.5,
        label=(
            f"Baseline = "
            f"estado {baseline_state}"
        )
    )

    if start_index is not None:
        ax3.axvline(
            t[start_index],
            linestyle='--',
            linewidth=2
        )

    if end_index is not None:
        ax3.axvline(
            t[end_index],
            linestyle=':',
            linewidth=2
        )

    ax3.set_xlabel(
        "Tempo (s)"
    )

    ax3.set_ylabel(
        "Estado"
    )

    ax3.set_yticks(
        np.arange(int(n_states))
    )

    ax3.legend()

    ax3.grid(
        alpha=0.3
    )

    st.pyplot(fig3)


    # ========================================================
    # CENTRÓIDES
    # ========================================================

    st.subheader(
        "Centróides dos estados"
    )

    centers_df = pd.DataFrame({
        "Estado": np.arange(int(n_states)),
        "Centróide": centers,
        "N na baseline": baseline_counts,
        "Baseline": [
            i == baseline_state
            for i in range(int(n_states))
        ]
    })

    st.dataframe(
        centers_df,
        hide_index=True,
        use_container_width=True
    )


    # ========================================================
    # MATRIZ DE TRANSIÇÃO
    # ========================================================

    st.subheader(
        "Matriz de transição entre estados"
    )

    matrix = transition_matrix(
        states,
        n_states=int(n_states)
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            f"Estado {i}"
            for i in range(int(n_states))
        ],
        columns=[
            f"→ {i}"
            for i in range(int(n_states))
        ]
    )

    st.dataframe(
        matrix_df.style.format(
            "{:.3f}"
        ),
        use_container_width=True
    )


    # ========================================================
    # DADOS PROCESSADOS
    # ========================================================

    st.subheader(
        "Dados processados"
    )

    processed_df = pd.DataFrame({
        "Tempo_s": t,
        "X_detrend": x,
        "Y_detrend": y,
        "Z_detrend": z,
        "Norma_bruta": norm_waveform,
        "Norma_1Hz": norm_1,
        "Norma_4Hz": norm_4,
        "Norma_10Hz": norm_10,
        "Estado": states
    })

    processed_df["Baseline"] = (
        states == baseline_state
    )

    processed_df["Inicio"] = False
    processed_df["Fim"] = False

    if start_index is not None:
        processed_df.loc[
            start_index,
            "Inicio"
        ] = True

    if end_index is not None:
        processed_df.loc[
            end_index,
            "Fim"
        ] = True


    with st.expander(
        "Mostrar tabela completa"
    ):
        st.dataframe(
            processed_df,
            use_container_width=True
        )


    # ========================================================
    # DOWNLOAD
    # ========================================================

    csv = processed_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Baixar dados processados",
        data=csv,
        file_name="segmentacao_iTUG.csv",
        mime="text/csv"
    )


    # ========================================================
    # DESCRIÇÃO DA REGRA
    # ========================================================

    with st.expander(
        "Regra de segmentação"
    ):

        st.markdown(
            f"""
### Baseline

O estado predominante nos primeiros
**{baseline_seconds:g} segundos**
é considerado o estado da baseline.

### Início da atividade

O início é identificado quando:

- a amostra anterior pertence ao estado da baseline;
- aparecem **{sequence_length} amostras consecutivas**;
- todas pertencem a estados numericamente superiores
  ao estado da baseline.

### Final da atividade

Após o início, procura-se a **primeira sequência de
{sequence_length} amostras consecutivas no estado da baseline**.

A primeira amostra dessa sequência é considerada o final.
"""
        )
