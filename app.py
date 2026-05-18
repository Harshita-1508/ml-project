import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Logistic Regression with AdaGrad
# -------------------------------
class LogisticRegressionAdaGrad:
    def __init__(self, lr=0.1, epochs=500):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)
        self.G = np.zeros(n)
        epsilon = 1e-8

        for _ in range(self.epochs):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)

            gradient = (1/m) * np.dot(X.T, (h - y))

            # AdaGrad update
            self.G += gradient**2
            adjusted_lr = self.lr / (np.sqrt(self.G) + epsilon)

            self.theta -= adjusted_lr * gradient

    def predict(self, X):
        z = np.dot(X, self.theta)
        probs = self.sigmoid(z)
        return (probs >= 0.5).astype(int)


# -------------------------------
# Logistic Regression with SGD
# -------------------------------
class LogisticRegressionSGD:
    def __init__(self, lr=0.1, epochs=500):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)

        for _ in range(self.epochs):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)

            gradient = (1/m) * np.dot(X.T, (h - y))

            # Standard Gradient Descent
            self.theta -= self.lr * gradient

    def predict(self, X):
        z = np.dot(X, self.theta)
        probs = self.sigmoid(z)
        return (probs >= 0.5).astype(int)


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("AdaGrad vs SGD (Logistic Regression)")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # Handle categorical data
    X = pd.get_dummies(X)

    if y.dtype == 'object':
        y = pd.factorize(y)[0]

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if st.button("Train Models"):
        # AdaGrad Model
        adagrad_model = LogisticRegressionAdaGrad(lr=0.1, epochs=500)
        adagrad_model.fit(X_train, y_train)
        preds_ada = adagrad_model.predict(X_test)
        acc_ada = (preds_ada == y_test).mean()

        # SGD Model
        sgd_model = LogisticRegressionSGD(lr=0.1, epochs=500)
        sgd_model.fit(X_train, y_train)
        preds_sgd = sgd_model.predict(X_test)
        acc_sgd = (preds_sgd == y_test).mean()

        st.subheader("Results")

        st.write("AdaGrad Accuracy:", acc_ada)
        st.write("SGD Accuracy:", acc_sgd)

        if acc_ada > acc_sgd:
            st.success("AdaGrad performed better ✅")
        elif acc_sgd > acc_ada:
            st.success("SGD performed better ✅")
        else:
            st.info("Both performed equally 🤝")