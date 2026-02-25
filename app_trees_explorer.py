import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import make_moons, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# CABEÇALHO INSTITUCIONAL
# ============================================================

st.set_page_config(page_title="Explorador de Árvores de Decisão", layout="wide")

st.markdown("""
<div style="background-color:#f0f2f6; padding:15px; border-radius:10px; border-left: 5px solid #28a745;">
    <strong>Aprendizado de Máquina – Profa. Mariana Recamonde Mendoza</strong><br>
    Instituto de Informática, Universidade Federal do Rio Grande do Sul (UFRGS).<br>
    <em>Material interativo desenvolvido com apoio de IA generativa (Gemini 3.1 Pro).</em>
</div>
""", unsafe_allow_html=True)

st.title("🌳 Explorador Interativo de Árvores de Decisão")

st.markdown("""
As **Árvores de Decisão** aprendem regras lógicas para separar os dados. 
Diferente do kNN, elas não precisam de normalização e são altamente interpretáveis.
Este explorador permite ajustar hiperparâmetros e visualizar como o modelo "particiona" o espaço.
""")

# ============================================================
# SIDEBAR – CONFIGURAÇÕES
# ============================================================

st.sidebar.header("🛠️ Configurações da Árvore")

# Hiperparâmetros (Valores padrão da biblioteca como base)
criterion = st.sidebar.selectbox("Critério de Impureza", ["gini", "entropy"])
max_depth = st.sidebar.slider("Profundidade Máxima (max_depth)", 1, 20, 3)
max_features = st.sidebar.slider("Atributos p/ Divisão (max_features)", 1, 13, 13) # Default None (all)
min_samples_split = st.sidebar.slider("Mínimo de amostras p/ Nó Interno", 2, 20, 2)

st.sidebar.markdown("---")

dataset_name = st.sidebar.selectbox("Escolha o Dataset", ["Moons (2D Simples)", "Wine (Multidimensional)"])

# Opção de Normalização (Para demonstrar invariância)
st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Pré-processamento")
normalize = st.sidebar.checkbox("Ativar Normalização (Min-Max)", value=False, help="Árvores de decisão são invariantes à escala, então os resultados não devem mudar.")

# Gestão de Dados de Teste
st.sidebar.subheader("🔄 Controle de Teste")
if "test_seed" not in st.session_state:
    st.session_state.test_seed = 42

seed_input = st.sidebar.text_input("Seed de Teste (Reprodutibilidade):", value=str(st.session_state.test_seed))

if st.sidebar.button("Gerar novos dados de teste"):
    try:
        st.session_state.test_seed = int(seed_input)
        st.sidebar.success(f"Novos dados gerados (Seed {seed_input})")
    except:
        st.sidebar.error("Seed deve ser um número inteiro.")

# ============================================================
# CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ============================================================

@st.cache_data
def get_data(name, seed):
    if name == "Moons (2D Simples)":
        X, y = make_moons(n_samples=200, noise=0.20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        feature_names = ["X1", "X2"]
        target_names = ["Lua A", "Lua B"]
    else:
        data = load_wine()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        feature_names = data.feature_names
        target_names = data.target_names
    return X_train, X_test, y_train, y_test, feature_names, target_names

X_train, X_test, y_train, y_test, features, targets = get_data(dataset_name, st.session_state.test_seed)

# Aplicação de Normalização
if normalize:
    scaler = MinMaxScaler()
    X_train_processed = scaler.fit_transform(X_train)
    X_test_processed = scaler.transform(X_test)
else:
    X_train_processed = X_train
    X_test_processed = X_test

# Variáveis para exibição e malha
X_disp_train = X_train_processed
X_disp_test = X_test_processed

# ============================================================
# TREINAMENTO
# ============================================================

# Ajustamos max_features para não exceder o número de colunas do dataset atual
current_max_features = min(max_features, X_train_processed.shape[1])

clf = DecisionTreeClassifier(
    criterion=criterion,
    max_depth=max_depth,
    max_features=current_max_features,
    min_samples_split=min_samples_split,
    random_state=42
)
clf.fit(X_train_processed, y_train)

# ============================================================
# VISUALIZAÇÃO - COLUNAS
# ============================================================

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🌲 Estrutura da Árvore")
    fig_tree, ax_tree = plt.subplots(figsize=(10, 8))
    plot_tree(clf, 
              feature_names=features, 
              class_names=targets, 
              filled=True, 
              rounded=True, 
              ax=ax_tree,
              fontsize=8)
    st.pyplot(fig_tree)

with col2:
    st.subheader("🗺️ Fronteira de Decisão")
    
    # Lógica para visualização 2D em datasets > 2D
    if dataset_name == "Moons (2D Simples)":
        idx_x, idx_y = 0, 1
    else:
        # Pega as duas características mais importantes
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        idx_x, idx_y = indices[0], indices[1]
        st.caption(f"Visualizando os eixos: **{features[idx_x]}** e **{features[idx_y]}** (as mais informativas).")

    # Geração da Malha (Grid)
    h = 0.05
    x_min, x_max = X_disp_train[:, idx_x].min() - 0.5, X_disp_train[:, idx_x].max() + 0.5
    y_min, y_max = X_disp_train[:, idx_y].min() - 0.5, X_disp_train[:, idx_y].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Para prever, precisamos de um vetor completo (mesma dimensão que o dataset)
    # Criamos um vetor preenchido com as médias do treino, e substituímos X e Y pela malha
    X_grid = np.tile(np.mean(X_disp_train, axis=0), (xx.ravel().shape[0], 1))
    X_grid[:, idx_x] = xx.ravel()
    X_grid[:, idx_y] = yy.ravel()
    
    Z = clf.predict(X_grid).reshape(xx.shape)
    
    fig_map, ax_map = plt.subplots(figsize=(10, 8))
    ax_map.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    
    # Plotar pontos de treino
    scatter = ax_map.scatter(X_disp_train[:, idx_x], X_disp_train[:, idx_y], c=y_train, edgecolors='k', cmap='viridis', alpha=0.5, label='Treino')
    
    # Plotar pontos de teste destacados
    preds_test = clf.predict(X_disp_test)
    for i in range(len(y_test)):
        color = 'white' if preds_test[i] == y_test[i] else 'red'
        marker = 'o' if preds_test[i] == y_test[i] else 'X'
        ax_map.scatter(X_disp_test[i, idx_x], X_disp_test[i, idx_y], c=color, edgecolors='black', s=100, marker=marker)

    ax_map.set_xlabel(features[idx_x])
    ax_map.set_ylabel(features[idx_y])
    ax_map.set_title("Fronteira e Pontos de Teste (🔴 = Erro)")
    st.pyplot(fig_map)

# ============================================================
# MÉTRICAS DE DESEMPENHO
# ============================================================

st.markdown("---")
mcol1, mcol2, mcol3 = st.columns(3)

acc_train = clf.score(X_disp_train, y_train)
acc_test = clf.score(X_disp_test, y_test)

mcol1.metric("Acurácia (Treino/Simulado)", f"{acc_train:.1%}")
mcol2.metric("Acurácia (Teste/Prova Real)", f"{acc_test:.1%}")

if normalize:
    st.info("💡 **Destaque Pedagógico**: Note que, ao ativar a normalização, os eixos mudam para o intervalo [0, 1], mas a **forma da fronteira** e a **acurácia** permanecem idênticas. Isso prova que as Árvores de Decisão são invariantes à escala!")

if acc_train - acc_test > 0.15:
    mcol3.warning("⚠️ Possível Overfitting: Modelo decorou os dados de treino!")
elif acc_train < 0.70:
    mcol3.info("ℹ️ Possível Underfitting: Tente aumentar a profundidade.")
else:
    mcol3.success("✅ Modelo Equilibrado: Boa generalização.")

# ============================================================
# DETALHAMENTO DE ATRIBUTOS
# ============================================================

with st.expander("📊 Ver Importância dos Atributos"):
    impact = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    st.bar_chart(impact)
    st.write("Atributos com maior importância foram usados para definir a raiz e os primeiros nós da árvore.")
