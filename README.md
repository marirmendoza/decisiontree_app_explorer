# 🌳 App Decision Tree Explorer — Visualizando a Lógica das Árvores de Decisão

Este aplicativo interativo em ***Streamlit*** e ***Python*** permite explorar, de maneira visual e intuitiva, os fundamentos das **Árvores de Decisão** — um dos modelos mais interpretáveis e amplamente utilizados em Aprendizado de Máquina, capaz de transformar dados em regras lógicas de decisão.

Este projeto foi desenvolvido como material de apoio para a disciplina de **Aprendizado de Máquina**, da **Profa. Mariana Recamonde Mendoza**, no **Instituto de Informática — Universidade Federal do Rio Grande do Sul (UFRGS)**.

---

## Objetivo do Aplicativo

As Árvores de Decisão buscam a "pureza" nos dados através de perguntas sequenciais. Este explorador permite visualizar:

- Como os hiperparâmetro **max_depth** (profundidade) e **min_samples_split** alteram a complexidade do modelo.
- O impacto da escolha do **critério de impureza** (Gini vs. Entropia).
- A **geometria das fronteiras de decisão**, que diferentemente do k-Nearest Neighbors, são sempre ortogonais (alinhadas aos eixos).
- O fenômeno do **Overfitting**, que ocorre quando a árvore cresce sem limites.
- A **Importância dos Atributos**, revelando quais características do dado mais influenciam na decisão final.
- Uma solução visual para **datasets multidimensionais** (como o dataset Wine), projetando a decisão nos dois atributos mais informativos.

---

## Visão Geral do App

O aplicativo possui três áreas principais:

1. **Configurações (barra lateral)**: Ajuste de hiperparâmetros e escolha do conjunto de dados (Moons 2D ou Wine Multidimensional).
2. **Visualização da Árvore e Fronteira**: Lado a lado, veja a estrutura lógica da árvore (nós e folhas) e como isso se traduz no mapa de decisão.
3. **Métricas de Performance**: Comparação em tempo real da acurácia entre os dados de Treino (Simulado) e Teste (Prova Real).

---

## Configurações do Modelo

Na barra lateral, você pode ajustar:

- **Critério de Impureza**: Gini ou Entropia.
- **Profundidade Máxima (max_depth)**: Controla o crescimento da árvore.
- **Atributos Analisados por Divisão (max_features)**: Quantas colunas a árvore considera ao criar um novo nó.
- **Mínimo de amostras**: Evita divisões em nós com poucos dados.
- **Seed de Teste**: Garante que os resultados sejam reprodutíveis.

---

## Por que usar este Explorador?

Diferente do kNN, as árvores não exigem normalização de dados e são modelos de **"Caixa Branca"**. Através deste app, o aluno consegue "enxergar" o raciocínio da máquina, auditando cada corte feito no espaço de atributos.

---

## Créditos
**Autora:** Profa. Mariana Recamonde Mendoza, Instituto de Informática, Universidade Federal do Rio Grande do Sul (UFRGS).

*Nota: O código foi desenvolvido com o apoio de IA generativa (Gemini 3.1 e ChatGPT 5.2).*
