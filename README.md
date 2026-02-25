# Decision Tree Explorer App — Visualizing Decision Tree Logic

This interactive application, built with **Streamlit** and **Python**, allows you to explore the fundamentals of **Decision Trees** in a visual and intuitive way.

Decision Trees are among the most interpretable and widely used models in Machine Learning. They work by recursively splitting the feature space into regions and transforming data into a sequence of simple, logical decision rules.

Beyond being powerful on their own, decision trees are the foundation of many modern, state-of-the-art algorithms. Popular ensemble methods such as **Random Forests** and **Gradient Boosted Trees** (e.g., XGBoost) are built by combining multiple decision trees to improve predictive performance, stability, and generalization.

Understanding how a single decision tree works is therefore essential for understanding many of the most successful models used in practice today.

This app was developed by **Prof. Mariana Recamonde Mendoza** as supporting material for the **Machine Learning** course taught at the **Institute of Informatics — Federal University of Rio Grande do Sul (UFRGS)**.

---

## App Goals

Decision Trees seek "purity" in data through sequential questions. This explorer allows you to visualize:

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

## Model Settings

In the sidebar, you can adjust:

- **Impurity Criterion**: Gini or Entropy.
- **Maximum Depth (max_depth)**: Controls tree growth.
- **Features for Split (max_features)**: How many columns the tree considers when creating a new node.
- **Minimum Samples**: Prevents splits in nodes with too few data points.
- **Test Seed**: Ensures results are reproducible.

---

## Why Use This Explorer?

Unlike kNN, trees do not require data normalization and are **"White Box"** models. Through this app, students can "see" the machine's reasoning, auditing every cut made in the feature space.

---

## Credits
**Author:** Prof. Mariana Recamonde Mendoza, Institute of Informatics, Federal University of Rio Grande do Sul (UFRGS).

*Note: The code was developed with the support of generative AI (Gemini 3.1 Pro).*
