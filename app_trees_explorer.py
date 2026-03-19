import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import make_classification, make_blobs
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Regressão Logística - INF/UFRGS", layout="wide")

# ============================================================
# CABEÇALHO INSTITUCIONAL
# ============================================================
st.markdown("""
<div style="background-color:#f0f2f6; padding:15px; border-radius:10px; border-left: 5px solid #007bff;">
    <strong>Machine Learning – Profa. Mariana Recamonde Mendoza</strong><br>
    Instituto de Informática, Universidade Federal do Rio Grande do Sul (UFRGS).<br>
    <em>Material interativo para exploração de Regressão Logística e Classificação Linear.</em>
</div>
""", unsafe_allow_html=True)

st.title("⚖️ Regressão Logística: Classificação Linear")
st.markdown("""
A **Regressão Logística** é usada para problemas de **Classificação**. Em vez de prever um número contínuo (como a Regressão Linear), ela prevê a **probabilidade** de um dado pertencer a uma classe particular.

Fazemos isso passando a saída de uma "reta" ($z = wx+b$) através de uma função chamada **Sigmóide**, que "esmaga" qualquer valor para ficar entre 0 e 1.
r"$$ \sigma(z) = \\frac{1}{1 + e^{-z}} $$"
""")

tabs = st.tabs(["1. Entendendo a Sigmóide (1D)", "2. Fronteira de Decisão Linear (2D)"])

# ============================================================
# TAB 1: A SIGMÓIDE EM 1D
# ============================================================
with tabs[0]:
    st.header("1. Da Reta à Probabilidade")
    st.write("Ajuste os parâmetros da equação $z = wx+b$ e veja o efeito da função Sigmóide para converter valores na variável preditora em probabilidade.")
    
    col1_sig, col2_sig = st.columns([1, 2])
    
    with col1_sig:
        w_sig = st.slider("Peso ($w$)", -5.0, 5.0, 2.0, 0.1, key='w_sig')
        b_sig = st.slider("Viés ($b$)", -10.0, 10.0, 0.0, 0.5, key='b_sig')
        
        st.markdown("---")
        st.subheader("Simular um Ponto")
        x_pt = st.number_input("Valor de Feature (X):", -10.0, 10.0, 0.0)
        z_pt = w_sig * x_pt + b_sig
        p_pt = 1 / (1 + np.exp(-z_pt))
        st.write(f"**Cálculo da Reta ($z$):** {w_sig:.1f} * ({x_pt:.1f}) + {b_sig:.1f} = **{z_pt:.2f}**")
        st.write(f"**Probabilidade ($\\sigma(z)$):** **{p_pt:.4f}** (ou {p_pt*100:.1f}%)")
        
        if p_pt >= 0.5:
            st.success("Predição: **Classe 1 (Positiva)**")
        else:
            st.error("Predição: **Classe 0 (Negativa)**")

    with col2_sig:
        x_vals = np.linspace(-10, 10, 200)
        z_vals = w_sig * x_vals + b_sig
        p_vals = 1 / (1 + np.exp(-z_vals))
        
        fig_sig = go.Figure()
        # A Curva Sigmoide
        fig_sig.add_trace(go.Scatter(x=x_vals, y=p_vals, mode='lines', 
                                     name='Probabilidade $\\sigma(wx+b)$', line=dict(color='blue', width=3)))
        
        # O ponto simulado
        fig_sig.add_trace(go.Scatter(x=[x_pt], y=[p_pt], mode='markers', 
                                     name='Ponto Atual', marker=dict(color='red', size=12, symbol='star')))
        
        # Linha de Decisão (Limiar 0.5)
        fig_sig.add_hline(y=0.5, line_dash='dash', line_color='gray', annotation_text="Limiar de Decisão (0.5)")
        
        fig_sig.update_layout(
            title="A Função Sigmóide",
            xaxis_title="Entrada (Feature X)",
            yaxis_title="Probabilidade p(y=1|x)",
            template="plotly_white",
            height=400,
            yaxis=dict(range=[-0.1, 1.1])
        )
        st.plotly_chart(fig_sig, use_container_width=True)
        
        st.info("💡 **Dica Didática**: Observe que quanto maior a magnitude de $w$, mais 'íngreme' fica a curva (transição mais abrupta). Mexer no $b$ desloca a curva para a esquerda ou para a direita.")

# ============================================================
# TAB 2: FRONTEIRA 2D
# ============================================================
with tabs[1]:
    st.header("2. Fronteira de Decisão em Problemas 2D")
    st.write("Quando temos múltiplas features (ex: $X_1$ e $X_2$), o modelo continua sendo linear: $z = w_1 X_1 + w_2 X_2 + b$. No entanto, no espaço de 2 dimensões, a fronteira onde as chances são de 50% ($p=0.5$) forma uma **reta que separa as duas classes**.")
    
    col1_2d, col2_2d = st.columns([1, 2])
    
    with col1_2d:
        st.subheader("Gerador de Dados")
        noise_level = st.slider("Mistura de Classes (Overlapping)", 0.1, 3.0, 1.0, 0.1)
        
        # Gerar dados
        np.random.seed(42)
        X_2d, y_2d = make_blobs(n_samples=200, centers=2, cluster_std=noise_level, random_state=42)
        
        # Treinar o Modelo
        logreg = LogisticRegression(C=1e5) # C alto = sem regularização para mostrar a separação pura
        logreg.fit(X_2d, y_2d)
        
        acc = logreg.score(X_2d, y_2d)
        st.metric("Acurácia no Treinamento", f"{acc*100:.1f}%")
        
        st.markdown("### Os Pesos Encontrados:")
        w1, w2 = logreg.coef_[0]
        st.write(f"$w_1$: {w1:.2f}")
        st.write(f"$w_2$: {w2:.2f}")
        st.write(f"$b$ (Viés): {logreg.intercept_[0]:.2f}")
        
    with col2_2d:
        # Plot Plotly Contour
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        
        # Previsão das probabilidades para o contorno de fundo
        grid_probs = logreg.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        grid_probs = grid_probs.reshape(xx.shape)
        
        fig_2d = go.Figure()

        # Heatmap da probabilidade (Contorno Suave)
        fig_2d.add_trace(go.Contour(
            x=np.arange(x_min, x_max, 0.1), y=np.arange(y_min, y_max, 0.1), z=grid_probs,
            colorscale='RdBu', opacity=0.4, showscale=True, 
            colorbar=dict(title='Probabilidade Classe 1')
        ))
        
        # Adicionar a RETA (Equação da reta: w1*x1 + w2*x2 + b = 0  => x2 = -(w1*x1 + b)/w2)
        if w2 != 0:
            x1_line = np.array([x_min, x_max])
            x2_line = -(w1 * x1_line + logreg.intercept_[0]) / w2
            fig_2d.add_trace(go.Scatter(x=x1_line, y=x2_line, mode='lines', 
                                        line=dict(color='black', dash='dash', width=2), name='Fronteira (p=0.5)'))

        # Pontos da Classe 0
        fig_2d.add_trace(go.Scatter(x=X_2d[y_2d==0, 0], y=X_2d[y_2d==0, 1], mode='markers', 
                                    name='Classe 0', marker=dict(color='red', size=8, line=dict(width=1, color='darkred'))))
        # Pontos da Classe 1
        fig_2d.add_trace(go.Scatter(x=X_2d[y_2d==1, 0], y=X_2d[y_2d==1, 1], mode='markers', 
                                    name='Classe 1', marker=dict(color='blue', size=8, line=dict(width=1, color='darkblue'))))
        
        fig_2d.update_layout(
            title="Zonas de Probabilidade e Fronteira de Decisão",
            xaxis_title="Feature 1 ($X_1$)",
            yaxis_title="Feature 2 ($X_2$)",
            height=500,
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max])
        )
        st.plotly_chart(fig_2d, use_container_width=True)
