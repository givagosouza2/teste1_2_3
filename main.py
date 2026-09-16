import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import signal
from sklearn.cluster import KMeans

# ============================================================
# CONFIGURAÇÃO
# ============================================================
st.set_page_config(
    page_title="Segmentação TUG - Norma Euclidiana",
    page_icon="📈",
    layout="wide"
)

st.title("Segmentação automática por norma euclidiana")
st.caption(
    "Detrend → interpolação a 100 Hz → filtros passa-baixa → "
    "norma euclidiana → K-means (5 estados) → detecção de início e fim"
)


# ============================================================
# FUNÇÕES
# ============================================================
def load_sensor_file(uploaded_file):
    """
    Lê arquivo TXT/CSV separado por ponto e vírgula.

    Formato esperado:
    Tempo (ms); Acc X (m/s²); Acc Y (m/s²); Acc Z (m/s²)

    Cabeçalhos repetidos no interior do arquivo são removidos
    automaticamente pela conversão numérica.
    """
    uploaded_file.seek(0)

    df = pd.read_csv(
        uploaded_file,
        sep=";",
        dtype=str,
        engine="python"
    )

    if df.shape[1] < 4:
        raise ValueError(
            "O arquivo deve possuir pelo menos 4 colunas: "
            "tempo, Acc X, Acc Y e Acc Z."
        )

    df = df.iloc[:, :4].copy()
    df.columns = ["Tempo_ms", "Acc_X", "Acc_Y", "Acc_Z"]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["Tempo_ms", "Acc_X", "Acc_Y", "Acc_Z"]
    )

    df = df.sort_values("Tempo_ms")
    df = df.drop_duplicates(subset="Tempo_ms", keep="first")
    df = df.reset_index(drop=True)

    if len(df) < 20:
        raise ValueError(
            "O arquivo possui poucos pontos válidos para análise."
        )

    return df


def detrend_and_resample(df, fs_target=100.0):
    """
    1) Detrend linear nos eixos X, Y e Z.
    2) Interpolação para frequência uniforme.
    """
    tempo_ms = df["Tempo_ms"].to_numpy(dtype=float)

    x = df["Acc_X"].to_numpy(dtype=float)
    y = df["Acc_Y"].to_numpy(dtype=float)
    z = df["Acc_Z"].to_numpy(dtype=float)

    t_original = (tempo_ms - tempo_ms[0]) / 1000.0

    # Detrend linear
    x_dt = signal.detrend(x, type="linear")
    y_dt = signal.detrend(y, type="linear")
    z_dt = signal.detrend(z, type="linear")

    # Interpolação
    dt = 1.0 / fs_target

    t_uniform = np.arange(
        0.0,
        t_original[-1] + dt / 2,
        dt
    )

    x_interp = np.interp(t_uniform, t_original, x_dt)
    y_interp = np.interp(t_uniform, t_original, y_dt)
    z_interp = np.interp(t_uniform, t_original, z_dt)

    return (
        t_original,
        t_uniform,
        x_interp,
        y_interp,
        z_interp
    )


def lowpass_filter(data, cutoff, fs=100.0, order=4):
    """
    Filtro Butterworth passa-baixa com fase zero.
    """
    sos = signal.butter(
        order,
        cutoff,
        btype="lowpass",
        fs=fs,
        output="sos"
    )

    return signal.sosfiltfilt(sos, data)


def calculate_filtered_norms(
    x,
    y,
    z,
    fs=100.0,
    cutoffs=(0.5),
    order=4
):
    """
    Filtra X, Y e Z separadamente e calcula a norma euclidiana:

        sqrt(X² + Y² + Z²)
    """
    result = {}

    for cutoff in cutoffs:
        xf = lowpass_filter(x, cutoff, fs, order)
        yf = lowpass_filter(y, cutoff, fs, order)
        zf = lowpass_filter(z, cutoff, fs, order)

        norm = np.sqrt(
            np.square(xf) +
            np.square(yf) +
            np.square(zf)
        )

        result[cutoff] = {
            "X": xf,
            "Y": yf,
            "Z": zf,
            "Norm": norm
        }

    return result


def kmeans_states(values, n_states=5, random_state=42):
    """
    K-means unidimensional.

    Os estados são reordenados pelos centróides:

        Estado 0 = menor centróide
        Estado 1 = segundo menor
        ...
        Estado N = maior centróide
    """
    X = np.asarray(values).reshape(-1, 1)

    km = KMeans(
        n_clusters=n_states,
        random_state=random_state,
        n_init=20
    )

    raw_labels = km.fit_predict(X)
    raw_centers = km.cluster_centers_.ravel()

    order = np.argsort(raw_centers)

    mapping = {
        old_label: new_label
        for new_label, old_label in enumerate(order)
    }

    states = np.array(
        [mapping[label] for label in raw_labels],
        dtype=int
    )

    centers_sorted = raw_centers[order]

    return states, centers_sorted


def detect_baseline_state(
    t,
    states,
    baseline_seconds=2.0,
    n_states=5
):
    """
    Baseline = estado predominante nos primeiros
    baseline_seconds segundos.
    """
    baseline_mask = t < baseline_seconds

    if not np.any(baseline_mask):
        raise ValueError(
            "Não existem amostras suficientes para definir a baseline."
        )

    counts = np.bincount(
        states[baseline_mask],
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
    - busca começa após a janela usada para baseline;
    - a amostra anterior deve estar na baseline;
    - as próximas N amostras precisam estar em estados
      numericamente superiores à baseline.
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
    após o início, procura a primeira volta ao estado baseline
    que permaneça nesse estado por N amostras consecutivas.
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


def markov_transition_matrix(states, n_states=5):
    """
    Matriz empírica de transição entre estados.
    É descritiva e não altera a regra de início/fim.
    """
    matrix = np.zeros(
        (n_states, n_states),
        dtype=float
    )

    for a, b in zip(states[:-1], states[1:]):
        matrix[a, b] += 1

    row_sums = matrix.sum(
        axis=1,
        keepdims=True
    )

    probabilities = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix),
        where=row_sums != 0
    )

    return probabilities


def plot_norms(t, filtered):
    fig = go.Figure()

    for cutoff in [1.0, 4.0, 10.0]:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=filtered[cutoff]["Norm"],
                mode="lines",
                name=f"Norma - {cutoff:g} Hz"
            )
        )

    fig.update_layout(
        title="Norma euclidiana após pré-processamento",
        xaxis_title="Tempo (s)",
        yaxis_title="Norma euclidiana (m/s²)",
        hovermode="x unified",
        height=500
    )

    return fig


def plot_segmentation(
    t,
    norm_signal,
    states,
    baseline_state,
    start_idx,
    end_idx,
    cutoff,
    baseline_seconds
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t,
            y=norm_signal,
            mode="lines",
            name=f"Norma euclidiana - {cutoff:g} Hz",
            line=dict(width=2)
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t,
            y=norm_signal,
            mode="markers",
            name="Estado K-means",
            marker=dict(
                size=4,
                color=states,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title="Estado"
                ),
                cmin=0,
                cmax=max(4, int(np.max(states)))
            ),
            hovertemplate=(
                "Tempo: %{x:.3f} s<br>"
                "Norma: %{y:.4f}<br>"
                "Estado: %{marker.color}<extra></extra>"
            )
        )
    )

    fig.add_vrect(
        x0=0,
        x1=baseline_seconds,
        opacity=0.10,
        line_width=0,
        annotation_text=f"Baseline inicial ({baseline_seconds:g} s)",
        annotation_position="top left"
    )

    if start_idx is not None:
        fig.add_vline(
            x=t[start_idx],
            line_width=3,
            line_dash="dash"
        )

        fig.add_annotation(
            x=t[start_idx],
            y=1.0,
            yref="paper",
            text=f"Início: {t[start_idx]:.3f} s",
            showarrow=True,
            arrowhead=2
        )

    if end_idx is not None:
        fig.add_vline(
            x=t[end_idx],
            line_width=3,
            line_dash="dot"
        )

        fig.add_annotation(
            x=t[end_idx],
            y=0.88,
            yref="paper",
            text=f"Fim: {t[end_idx]:.3f} s",
            showarrow=True,
            arrowhead=2
        )

    fig.update_layout(
        title=(
            f"Segmentação pela norma euclidiana — "
            f"baseline = estado {baseline_state}"
        ),
        xaxis_title="Tempo (s)",
        yaxis_title="Norma euclidiana (m/s²)",
        hovermode="closest",
        height=620
    )

    return fig


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Parâmetros")

fs_target = st.sidebar.number_input(
    "Frequência de interpolação (Hz)",
    min_value=20,
    max_value=500,
    value=100,
    step=10
)

filter_order = st.sidebar.number_input(
    "Ordem do filtro Butterworth",
    min_value=2,
    max_value=8,
    value=4,
    step=1
)

n_states = st.sidebar.number_input(
    "Número de estados do K-means",
    min_value=2,
    max_value=10,
    value=5,
    step=1
)

baseline_seconds = st.sidebar.number_input(
    "Duração inicial para baseline (s)",
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

segmentation_cutoff = st.sidebar.selectbox(
    "Filtro usado no K-means",
    options=[1.0, 4.0, 10.0],
    index=2,
    format_func=lambda x: f"{x:g} Hz"
)


# ============================================================
# UPLOAD
# ============================================================
uploaded_file = st.file_uploader(
    "Carregue o arquivo de acelerometria",
    type=["txt", "csv"]
)

if uploaded_file is None:
    st.info(
        "Carregue um arquivo TXT/CSV com as colunas "
        "Tempo (ms), Acc X, Acc Y e Acc Z."
    )
    st.stop()


# ============================================================
# LEITURA
# ============================================================
try:
    df = load_sensor_file(uploaded_file)

except Exception as e:
    st.error(f"Erro ao abrir o arquivo: {e}")
    st.stop()


st.subheader("1. Dados carregados")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Amostras válidas",
        len(df)
    )

with c2:
    duration_original = (
        df["Tempo_ms"].iloc[-1] -
        df["Tempo_ms"].iloc[0]
    ) / 1000

    st.metric(
        "Duração do registro",
        f"{duration_original:.2f} s"
    )

with c3:
    if len(df) > 1:
        dt_original = np.diff(
            df["Tempo_ms"].to_numpy()
        ) / 1000

        fs_original = 1 / np.median(
            dt_original
        )

        st.metric(
            "Frequência mediana original",
            f"{fs_original:.1f} Hz"
        )

with st.expander("Visualizar dados brutos"):
    st.dataframe(
        df,
        use_container_width=True
    )


# ============================================================
# PROCESSAMENTO
# ============================================================
try:
    (
        t_original,
        t_uniform,
        x_interp,
        y_interp,
        z_interp
    ) = detrend_and_resample(
        df,
        fs_target=float(fs_target)
    )

    filtered = calculate_filtered_norms(
        x_interp,
        y_interp,
        z_interp,
        fs=float(fs_target),
        cutoffs=(1.0, 4.0, 10.0),
        order=int(filter_order)
    )

except Exception as e:
    st.error(
        f"Erro no pré-processamento: {e}"
    )
    st.stop()


st.subheader("2. Pré-processamento")

st.write(
    f"O registro foi submetido a **detrend linear nos três eixos**, "
    f"interpolado para **{fs_target:g} Hz** e filtrado com "
    f"passa-baixa Butterworth em **1, 4 e 10 Hz**. "
    f"Após a filtragem de cada eixo, foi calculada a norma euclidiana: "
    f"**√(X² + Y² + Z²)**."
)

st.plotly_chart(
    plot_norms(
        t_uniform,
        filtered
    ),
    use_container_width=True
)


# ============================================================
# K-MEANS E SEGMENTAÇÃO
# ============================================================
norm_for_segmentation = filtered[
    segmentation_cutoff
]["Norm"]

try:
    states, centers = kmeans_states(
        norm_for_segmentation,
        n_states=int(n_states)
    )

    baseline_state, baseline_counts = detect_baseline_state(
        t_uniform,
        states,
        baseline_seconds=float(baseline_seconds),
        n_states=int(n_states)
    )

    start_idx = detect_start(
        t_uniform,
        states,
        baseline_state,
        baseline_seconds=float(baseline_seconds),
        sequence_length=int(sequence_length)
    )

    end_idx = detect_end(
        states,
        baseline_state,
        start_idx,
        sequence_length=int(sequence_length)
    )

except Exception as e:
    st.error(
        f"Erro na segmentação: {e}"
    )
    st.stop()


st.subheader("3. Estados e segmentação")

centers_df = pd.DataFrame({
    "Estado": np.arange(len(centers)),
    "Centróide da norma": centers
})

centers_df["Baseline"] = (
    centers_df["Estado"] == baseline_state
)

col_a, col_b = st.columns([1, 2])

with col_a:
    st.markdown("#### Centróides dos estados")

    st.dataframe(
        centers_df,
        hide_index=True,
        use_container_width=True
    )

    st.write(
        f"**Estado de baseline:** {baseline_state}"
    )

with col_b:
    baseline_distribution = pd.DataFrame({
        "Estado": np.arange(len(baseline_counts)),
        "N na janela inicial": baseline_counts
    })

    st.markdown(
        "#### Distribuição durante a baseline"
    )

    st.dataframe(
        baseline_distribution,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# RESULTADOS TEMPORAIS
# ============================================================
r1, r2, r3 = st.columns(3)

if start_idx is not None:
    start_time = float(
        t_uniform[start_idx]
    )

    r1.metric(
        "Início detectado",
        f"{start_time:.3f} s"
    )
else:
    start_time = None

    r1.metric(
        "Início detectado",
        "Não encontrado"
    )


if end_idx is not None:
    end_time = float(
        t_uniform[end_idx]
    )

    r2.metric(
        "Fim detectado",
        f"{end_time:.3f} s"
    )
else:
    end_time = None

    r2.metric(
        "Fim detectado",
        "Não encontrado"
    )


if (
    start_time is not None and
    end_time is not None
):
    activity_duration = (
        end_time - start_time
    )

    r3.metric(
        "Duração detectada",
        f"{activity_duration:.3f} s"
    )
else:
    activity_duration = None

    r3.metric(
        "Duração detectada",
        "—"
    )


# ============================================================
# GRÁFICO DA SEGMENTAÇÃO
# ============================================================
fig_segmentation = plot_segmentation(
    t_uniform,
    norm_for_segmentation,
    states,
    baseline_state,
    start_idx,
    end_idx,
    segmentation_cutoff,
    float(baseline_seconds)
)

st.plotly_chart(
    fig_segmentation,
    use_container_width=True
)


# ============================================================
# ESTADOS AO LONGO DO TEMPO
# ============================================================
st.subheader(
    "4. Sequência temporal dos estados"
)

fig_states = go.Figure()

fig_states.add_trace(
    go.Scatter(
        x=t_uniform,
        y=states,
        mode="lines",
        line_shape="hv",
        name="Estado"
    )
)

fig_states.add_hline(
    y=baseline_state,
    line_dash="dash",
    annotation_text=(
        f"Baseline = estado {baseline_state}"
    )
)

if start_idx is not None:
    fig_states.add_vline(
        x=t_uniform[start_idx],
        line_dash="dash"
    )

if end_idx is not None:
    fig_states.add_vline(
        x=t_uniform[end_idx],
        line_dash="dot"
    )

fig_states.update_layout(
    xaxis_title="Tempo (s)",
    yaxis_title="Estado",
    height=360
)

fig_states.update_yaxes(
    tickmode="linear",
    dtick=1
)

st.plotly_chart(
    fig_states,
    use_container_width=True
)


# ============================================================
# MATRIZ DE TRANSIÇÃO
# ============================================================
st.subheader(
    "5. Matriz de transição entre estados"
)

transition = markov_transition_matrix(
    states,
    n_states=int(n_states)
)

transition_df = pd.DataFrame(
    transition,
    index=[
        f"Estado {i}"
        for i in range(int(n_states))
    ],
    columns=[
        f"→ {i}"
        for i in range(int(n_states))
    ]
)

st.caption(
    "A matriz de transição é apresentada de forma descritiva. "
    "A identificação de início e fim continua seguindo a regra "
    "sequencial definida."
)

st.dataframe(
    transition_df.style.format(
        "{:.3f}"
    ),
    use_container_width=True
)


# ============================================================
# TABELA PROCESSADA
# ============================================================
processed_df = pd.DataFrame({
    "Tempo_s": t_uniform,

    "X_1Hz": filtered[1.0]["X"],
    "Y_1Hz": filtered[1.0]["Y"],
    "Z_1Hz": filtered[1.0]["Z"],
    "Norma_1Hz": filtered[1.0]["Norm"],

    "X_4Hz": filtered[4.0]["X"],
    "Y_4Hz": filtered[4.0]["Y"],
    "Z_4Hz": filtered[4.0]["Z"],
    "Norma_4Hz": filtered[4.0]["Norm"],

    "X_10Hz": filtered[10.0]["X"],
    "Y_10Hz": filtered[10.0]["Y"],
    "Z_10Hz": filtered[10.0]["Z"],
    "Norma_10Hz": filtered[10.0]["Norm"],

    "Estado": states
})

processed_df["Baseline_state"] = (
    baseline_state
)

processed_df["Inicio"] = False
processed_df["Fim"] = False

if start_idx is not None:
    processed_df.loc[
        start_idx,
        "Inicio"
    ] = True

if end_idx is not None:
    processed_df.loc[
        end_idx,
        "Fim"
    ] = True


st.subheader(
    "6. Dados processados"
)

with st.expander(
    "Visualizar tabela processada"
):
    st.dataframe(
        processed_df,
        use_container_width=True
    )


csv = processed_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Baixar dados processados em CSV",
    data=csv,
    file_name=(
        "dados_processados_norma_euclidiana.csv"
    ),
    mime="text/csv"
)


# ============================================================
# RESUMO DA REGRA
# ============================================================
with st.expander(
    "Regra de segmentação utilizada"
):
    st.markdown(
        f"""
### Sinal usado

A análise é realizada sobre a **norma euclidiana**:

**√(X² + Y² + Z²)**

calculada após o detrend, interpolação e filtragem
separada dos três eixos.

### Baseline

O estado predominante nos primeiros
**{baseline_seconds:g} segundos**
é definido como estado de baseline.

### Início

A busca começa após a janela inicial usada para baseline.

O início é definido quando ocorre:

1. uma amostra no estado de baseline;
2. seguida por **{sequence_length} amostras consecutivas**;
3. todas pertencentes a estados **superiores ao estado de baseline**.

### Final

Após o início, o final é definido pela **primeira transição**
para uma sequência de **{sequence_length} amostras consecutivas**
no próprio estado de baseline.
"""
    )
