import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import make_moons, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# INSTITUTIONAL HEADER
# ============================================================

st.set_page_config(page_title="Decision Tree Explorer", layout="wide")

st.markdown("""
<div style="background-color:#f0f2f6; padding:15px; border-radius:10px; border-left: 5px solid #28a745;">
    <strong>Machine Learning – Profa. Mariana Recamonde Mendoza</strong><br>
    Institute of Informatics, Federal University of Rio Grande do Sul (UFRGS).<br>
    <em>Interactive material developed with generative AI support (Gemini 3.1 Pro and ChatGPT 5.2).</em>
</div>
""", unsafe_allow_html=True)

st.title("Decision Tree Interactive Explorer")

st.markdown("""
**Decision Trees** learn logical rules to separate data. 
Unlike kNN, they do not require normalization and are highly interpretable.
This explorer allows you to adjust hyperparameters and visualize how the model partitions the space.
""")

# ============================================================
# SIDEBAR – SETTINGS
# ============================================================

st.sidebar.header("Tree Settings")

# Hyperparameters
criterion = st.sidebar.selectbox("Impurity Criterion", ["gini", "entropy"])
max_depth = st.sidebar.slider("Maximum Depth (max_depth)", 1, 20, 3)
max_features = st.sidebar.slider("Features for Split (max_features)", 1, 13, 13) 
min_samples_split = st.sidebar.slider("Minimum Samples for Split", 2, 20, 2)

st.sidebar.markdown("---")

dataset_name = st.sidebar.selectbox("Select Dataset", ["Moons (2D Simple)", "Wine (Multidimensional)"])

# Normalization Option
st.sidebar.markdown("---")
st.sidebar.subheader("Preprocessing")
normalize = st.sidebar.checkbox("Enable Normalization (Min-Max)", value=False, help="Decision Trees are scale-invariant, so results should not change.")

# Test Data Management
st.sidebar.subheader("Test Control")
if "test_seed" not in st.session_state:
    st.session_state.test_seed = 42

seed_input = st.sidebar.text_input("Test Seed (Reproducibility):", value=str(st.session_state.test_seed))

if st.sidebar.button("Generate new test data"):
    try:
        st.session_state.test_seed = int(seed_input)
        st.sidebar.success(f"New data generated (Seed {seed_input})")
    except:
        st.sidebar.error("Seed must be an integer.")

# ============================================================
# DATA LOADING AND PREPARATION
# ============================================================

@st.cache_data
def get_data(name, seed):
    if name == "Moons (2D Simple)":
        X, y = make_moons(n_samples=200, noise=0.20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        feature_names = ["X1", "X2"]
        target_names = ["Moon A", "Moon B"]
    else:
        data = load_wine()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        feature_names = data.feature_names
        target_names = data.target_names
    return X_train, X_test, y_train, y_test, feature_names, target_names

X_train, X_test, y_train, y_test, features, targets = get_data(dataset_name, st.session_state.test_seed)

# Normalization Application
if normalize:
    scaler = MinMaxScaler()
    X_train_processed = scaler.fit_transform(X_train)
    X_test_processed = scaler.transform(X_test)
else:
    X_train_processed = X_train
    X_test_processed = X_test

# Variables for display and mesh
X_disp_train = X_train_processed
X_disp_test = X_test_processed

# ============================================================
# TRAINING
# ============================================================

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
# VISUALIZATION - COLUMNS
# ============================================================

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Tree Structure")
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
    st.subheader("Decision Boundary")
    
    if dataset_name == "Moons (2D Simple)":
        idx_x, idx_y = 0, 1
    else:
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        idx_x, idx_y = indices[0], indices[1]
        st.caption(f"Visualizing axes: **{features[idx_x]}** and **{features[idx_y]}** (most informative).")

    # Mesh Grid Generation
    h = 0.05
    x_min, x_max = X_disp_train[:, idx_x].min() - 0.5, X_disp_train[:, idx_x].max() + 0.5
    y_min, y_max = X_disp_train[:, idx_y].min() - 0.5, X_disp_train[:, idx_y].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    X_grid = np.tile(np.mean(X_disp_train, axis=0), (xx.ravel().shape[0], 1))
    X_grid[:, idx_x] = xx.ravel()
    X_grid[:, idx_y] = yy.ravel()
    
    Z = clf.predict(X_grid).reshape(xx.shape)
    
    fig_map, ax_map = plt.subplots(figsize=(10, 8))
    ax_map.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    
    # Plot training points
    ax_map.scatter(X_disp_train[:, idx_x], X_disp_train[:, idx_y], c=y_train, edgecolors='k', cmap='viridis', alpha=0.5, label='Train')
    
    # Plot testing points
    preds_test = clf.predict(X_disp_test)
    
    # Point Inspection functionality
    st.markdown("---")
    st.subheader("Test Point Inspection")
    
    selected_point_idx = st.selectbox(
        "Select a test point to highlight:",
        options=range(len(y_test)),
        format_func=lambda i: f"Point {i+1} (True Class: {targets[y_test[i]]})"
    )

    for i in range(len(y_test)):
        is_selected = (i == selected_point_idx)
        color = 'white' if preds_test[i] == y_test[i] else 'red'
        marker = 'o' if preds_test[i] == y_test[i] else 'X'
        size = 200 if is_selected else 100
        zorder = 5 if is_selected else 3
        edge_color = 'yellow' if is_selected else 'black'
        edge_width = 3 if is_selected else 1
        
        ax_map.scatter(X_disp_test[i, idx_x], X_disp_test[i, idx_y], 
                       c=color, edgecolors=edge_color, linewidths=edge_width,
                       s=size, marker=marker, zorder=zorder)

    ax_map.set_xlabel(features[idx_x])
    ax_map.set_ylabel(features[idx_y])
    ax_map.set_title("Boundary and Test Points (Red/X = Error)")
    st.pyplot(fig_map)

# ============================================================
# PERFORMANCE METRICS
# ============================================================

st.markdown("---")
mcol1, mcol2, mcol3 = st.columns(3)

acc_train = clf.score(X_disp_train, y_train)
acc_test = clf.score(X_disp_test, y_test)

mcol1.metric("Accuracy (Train/Simulated)", f"{acc_train:.1%}")
mcol2.metric("Accuracy (Test/Real Test)", f"{acc_test:.1%}")

if normalize:
    st.info("Pedagogical Note: Notice that when enabling normalization, the axes change to the [0, 1] range, but the boundary shape and accuracy remain identical. This proves that Decision Trees are scale-invariant.")

if acc_train - acc_test > 0.15:
    mcol3.warning("Potential Overfitting: Model memorized training data!")
elif acc_train < 0.70:
    mcol3.info("Potential Underfitting: Try increasing depth.")
else:
    mcol3.success("Balanced Model: Good generalization.")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

with st.expander("Show Feature Importance"):
    impact = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    st.bar_chart(impact)
    st.write("Attributes with higher importance were used define the root and initial nodes of the tree.")

# Detailed Test Results Table
with st.expander("Show Detailed Test Results Table"):
    results_df = pd.DataFrame({
        "Point": [f"Point {i+1}" for i in range(len(y_test))],
        "True Class": [targets[i] for i in y_test],
        "Prediction": [targets[i] for i in preds_test],
        "Result": ["Correct" if p == r else "Error" for p, r in zip(preds_test, y_test)]
    })
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    st.write(f"The model predicted {np.sum(preds_test == y_test)} out of {len(y_test)} unknown samples correctly.")
