import numpy as np
import time



class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        return self._sigmoid(np.dot(X, self.weights) + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


class KNNScratch:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        return np.bincount(k_nearest_labels).argmax()




if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 1000
    split = int(0.8 * n_samples)

    # Regression Dataset
    X_reg = np.random.randn(n_samples, 3)
    true_weights = np.array([3.5, -2.0, 1.0])
    y_reg = np.dot(X_reg, true_weights) + 4.0 + np.random.randn(n_samples) * 0.5
    X_train_reg, X_test_reg = X_reg[:split], X_reg[split:]
    y_train_reg, y_test_reg = y_reg[:split], y_reg[split:]

    # Classification Dataset
    X_clf = np.random.randn(n_samples, 2)
    y_clf = ((X_clf[:, 0] + X_clf[:, 1]) > 0).astype(int)
    X_train_clf, X_test_clf = X_clf[:split], X_clf[split:]
    y_train_clf, y_test_clf = y_clf[:split], y_clf[split:]

    # --- Linear Regression ---
    lin_reg = LinearRegressionScratch(learning_rate=0.1, epochs=500)
    t0 = time.time()
    lin_reg.fit(X_train_reg, y_train_reg)
    lin_train_time = time.time() - t0

    t0 = time.time()
    y_pred_reg = lin_reg.predict(X_test_reg)
    lin_inf_time = time.time() - t0
    mse = np.mean((y_pred_reg - y_test_reg) ** 2)

    # --- Logistic Regression ---
    log_reg = LogisticRegressionScratch(learning_rate=0.1, epochs=500)
    t0 = time.time()
    log_reg.fit(X_train_clf, y_train_clf)
    log_train_time = time.time() - t0

    t0 = time.time()
    y_pred_log = log_reg.predict(X_test_clf)
    log_inf_time = time.time() - t0
    log_accuracy = np.mean(y_pred_log == y_test_clf)

    # --- k-NN ---
    knn = KNNScratch(k=5)
    t0 = time.time()
    knn.fit(X_train_clf, y_train_clf)
    knn_train_time = time.time() - t0

    t0 = time.time()
    y_pred_knn = knn.predict(X_test_clf)
    knn_inf_time = time.time() - t0
    knn_accuracy = np.mean(y_pred_knn == y_test_clf)


    print("=" * 65)
    print(f"{'Algorithm':<22} | {'Train Time (s)':<14} | {'Inference (s)':<14} | {'Metric Score'}")
    print("-" * 65)
    print(f"{'Linear Regression':<22} | {lin_train_time:.5f}         | {lin_inf_time:.5f}         | MSE: {mse:.4f}")
    print(f"{'Logistic Regression':<22} | {log_train_time:.5f}         | {log_inf_time:.5f}         | Acc: {log_accuracy*100:.1f}%")
    print(f"{'k-NN (k=5)':<22} | {knn_train_time:.5f}         | {knn_inf_time:.5f}         | Acc: {knn_accuracy*100:.1f}%")
    print("=" * 65)
