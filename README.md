# Decision Tree Explorer App — Visualizing Decision Tree Logic

This interactive application, built with **Streamlit** and **Python**, allows you to explore the fundamentals of **Decision Trees** in a visual and intuitive way.

Decision Trees are among the most interpretable and widely used models in Machine Learning. They work by recursively splitting the feature space into regions and transforming data into a sequence of simple, logical decision rules.

Beyond being powerful on their own, decision trees are the foundation of many modern, state-of-the-art algorithms. Popular ensemble methods such as **Random Forests** and **Gradient Boosted Trees** (e.g., XGBoost) are built by combining multiple decision trees to improve predictive performance, stability, and generalization.

Understanding how a single decision tree works is therefore essential for understanding many of the most successful models used in practice today.

This app was developed by **Prof. Mariana Recamonde Mendoza** as supporting material for the **Machine Learning** course taught at the **Institute of Informatics — Federal University of Rio Grande do Sul (UFRGS)**.

---

## App Goal

Decision Trees seek "purity" in data through sequential questions. This translates into sequential "cuts" in the input space. 

This explorer allows you to visualize:

- How the hyperparameters **max_depth** and **min_samples_split** alter model complexity.
- The impact of choosing different **impurity criteria** (Gini vs. Entropy).
- The **geometry of decision boundaries**, which unlike kNN, are always orthogonal (aligned with the axes).
- The phenomenon of **Overfitting** when the tree grows without limits.
- **Feature Importance**, revealing which data characteristics most influence the final decision.
- A visual solution for **multidimensional datasets** (such as the Wine dataset), projecting the decision onto the two most informative attributes.

---

## App Overview

The application has three main areas:

1. **Settings (Sidebar)**: Adjust hyperparameters and choose the dataset (2D Moons or Multidimensional Wine).
2. **Tree and Boundary Visualization**: Side-by-side, see the logical structure of the tree (nodes and leaves) and how it translates into the decision map.
3. **Performance Metrics**: Real-time comparison of accuracy between Training (Simulated) and Testing (Real Test) data.

---

## Decision Tree Model Settings

In the sidebar, you can adjust:

- **Impurity Criterion**: Gini or Entropy.
- **Maximum Depth (max_depth)**: Controls tree growth.
- **Features for Split (max_features)**: How many columns the tree considers when creating a new node.
- **Minimum Samples**: Prevents splits in nodes with too few data points.
- **Test Seed**: Ensures results are reproducible.


---

## Exploratory Scenarios

The app includes four main exploratory scenarios:

### 1️⃣ Model Complexity (Tree Depth)

This scenario allows you to observe:

- How shallow trees (small `max_depth`) produce simpler, more general decision rules (higher bias)  
- How deeper trees create more complex partitions of the feature space (higher variance)  
- How excessive depth may lead to overfitting (high training accuracy but lower test accuracy)  

You can directly compare training and test accuracy to understand generalization behavior.

---

### 2️⃣ Impurity Criterion (Gini vs Entropy)

This scenario demonstrates:

- How different impurity measures influence split selection  
- That both criteria aim to create purer nodes  
- That small differences in split decisions may lead to slightly different tree structures  

You can visualize how the structure of the tree changes when switching between **Gini** and **Entropy**.

---

### 3️⃣ Feature Selection and Importance

This scenario highlights:

- How limiting `max_features` restricts the attributes considered at each split  
- How different features are selected as root or internal nodes  
- How feature importance reflects the contribution of each attribute  

For multidimensional datasets (e.g., Wine), the decision boundary visualization automatically focuses on the two most informative features.

---

### 4️⃣ Scale Invariance

This scenario demonstrates that:

- Decision Trees do not rely on distance computations  
- Enabling Min-Max normalization changes the axis scale  
- The decision boundary shape and accuracy remain unchanged  

This confirms that Decision Trees are **scale-invariant models**, 


---

## Credits

**Author:** Profa. Mariana Recamonde Mendoza. 

🔗 [Personal website.](https://www.inf.ufrgs.br/~mrmendoza/)

📍 [Institute of Informatics](https://www.inf.ufrgs.br/site/) - Federal University of Rio Grande do Sul (UFRGS), Porto Alegre - RS, Brazil


---
## Notes
*The code was developed with the support of Generative AI (Gemini 3.1 and ChatGPT 5.2).*
