import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.datasets import make_circles

class SLP:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.weights = np.random.randn(input_dim)
        self.bias = np.random.randn()

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        return self.activation(np.dot(x, self.weights) + self.bias)

    def train(self, x_train, y_train, epochs=300, learning_rate=0.1):
        for epoch in range(epochs):
            for i in range(len(x_train)):
                y_pred = self.predict(x_train[i])
                error = y_train[i] - y_pred
                self.weights += learning_rate * error * x_train[i]
                self.bias += learning_rate * error

class MLP:
    def __init__(self, hidden_layer_conf, num_output_nodes):
        self.hidden_layer_conf = hidden_layer_conf
        self.num_output_nodes = num_output_nodes
        self.logic_op_model = None
        self.build_model()

    def build_model(self):
        input_layer = tf.keras.Input(shape=[2, ])
        hidden_layers = input_layer

        for _ in range(2):
            hidden_layers = tf.keras.layers.Dense(units=self.hidden_layer_conf[0],
                                                  activation=tf.keras.activations.sigmoid,
                                                  use_bias=True)(hidden_layers)
        output = tf.keras.layers.Dense(units=self.num_output_nodes,
                                       activation=tf.keras.activations.sigmoid,
                                       use_bias=True)(hidden_layers)

        self.logic_op_model = tf.keras.Model(inputs=input_layer, outputs=output)

        sgd = tf.keras.optimizers.SGD(learning_rate=0.1)
        self.logic_op_model.compile(optimizer=sgd, loss="mse")

    def fit(self, x, y, batch_size, epochs):
        self.logic_op_model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs)

    def predict(self, x, batch_size):
        prediction = self.logic_op_model.predict(x=x, batch_size=batch_size)
        return prediction

def SLP_CircleClassify():
    # 데이터 생성
    n_samples = 400
    noise = 0.02
    factor = 0.5
    x_train, y_train = make_circles(n_samples=n_samples, noise=noise, factor=factor)
    x_test, y_test = make_circles(n_samples=n_samples, noise=noise, factor=factor)

    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, marker='.')
    plt.title("Train data distribution")
    plt.show()

    # SLP 분류기 훈련
    slp = SLP(input_dim=2)
    slp.train(x_train, y_train)

    # SLP 분류기 테스트
    predictions = [slp.predict(x) for x in x_test]

    # 테스트 데이터 분류 시각화
    plt.scatter(x_test[:, 0], x_test[:, 1], c=predictions, marker='.')
    plt.title("SLP prediction results")
    plt.show()

def MLP_CircleClassify():
    # 훈련 및 테스트 데이터 생성
    n_samples = 400
    noise = 0.02
    factor = 0.5
    x_train, y_train = make_circles(n_samples=n_samples, noise=noise, factor=factor)
    x_test, y_test = make_circles(n_samples=n_samples, noise=noise, factor=factor)

    # MLP 설정
    configs = {
        1: [3],
        2: [5],
        3: [10]
    }

    # 각 구성에 대한 분류 및 결과 플로팅
    for config_number, hidden_layer_conf in configs.items():
        print(f"Config {config_number}: Hidden Layer 층 수: [2], 각 층별 node 수: {hidden_layer_conf}")

        # MLP 모델 생성 및 훈련
        mlp = MLP(hidden_layer_conf=hidden_layer_conf, num_output_nodes=1)
        mlp.fit(x_train, y_train, batch_size=1, epochs=300)

        # 훈련된 모델 사용하여 예측
        predictions = mlp.predict(x_test, batch_size=1)

        # 테스트 데이터 분류 플로팅
        plt.scatter(x_test[:, 0], x_test[:, 1], c=predictions[:, 0], marker='.')
        plt.title(f"MLP prediction results - config {config_number}")
        plt.show()

if __name__ == '__main__':
    SLP_CircleClassify()
    MLP_CircleClassify()
