import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import util


class Model:
    """
    Abstract class for a machine learning model.
    """

    def get_features(self, x_input):
        pass

    def get_weights(self):
        pass

    def hypothesis(self, x):
        pass

    def predict(self, x):
        pass

    def loss(self, x, y):
        pass

    def gradient(self, x, y):
        pass

    def train(self, dataset):
        pass


# PA4 Q1
class PolynomialRegressionModel(Model):
    """
    Linear regression model with polynomial features (powers of x up to specified degree).
    x and y are real numbers. The goal is to fit y = hypothesis(x).
    """

    def __init__(self, degree=1, learning_rate=1e-3):
        self.degree = degree
        self.learning_rate = learning_rate
        self.weights = np.zeros(degree + 1)
        self.losses = []
        self.eval_iterations = 100000

    def get_features(self, x):
        # dummy feature for bias
        features = np.array([1] + [x ** i for i in range(1, self.degree + 1)])
        return features

    def get_weights(self):
        return self.weights

    def hypothesis(self, x):
        features = self.get_features(x)
        return np.dot(features, self.weights)

    def predict(self, x):
        return self.hypothesis(x)

    def loss(self, x, y):
        predicted_y = self.hypothesis(x)
        return (predicted_y - y) ** 2

    def gradient(self, x, y):
        predicted_y = self.hypothesis(x)
        features = self.get_features(x)
        gradient = [2 * (predicted_y - y) * feature for feature in features]
        return gradient

    def train(self, dataset, evalset=None):
        x_train = dataset.xs
        y_train = dataset.ys

        for iteration in range(self.eval_iterations):
            # for i in range(len(x_train)):
            i = np.random.randint(1, dataset.get_size() - 1)
            x = x_train[i]
            y = y_train[i]

                # Compute gradient and update weights
            grad = self.gradient(x, y)
            self.weights -= (self.learning_rate * np.array(grad))

            # Evaluate the model on the evaluation set if provided
            if iteration % 10 == 0:
                current_loss = np.mean([self.loss(x, y) for x, y in zip(x_train, y_train)])
                self.losses.append(current_loss)


# PA4 Q2
def linear_regression():
    # a)
    sine_train = util.get_dataset("sine_train")
    sine_model = PolynomialRegressionModel(degree=1, learning_rate=1e-4)
    sine_model.train(sine_train)

    print(sine_model.get_weights())
    print(f'Final Hypothesis of Linear Regression: y = {sine_model.weights[0]} + {sine_model.weights[1]}x')
    print('Average Loss of Linear Regression:', sine_train.compute_average_loss(sine_model))
    sine_train.plot_data(sine_model)

    # b)
    sine_train.plot_loss_curve(eval_iters=np.arange(1, (sine_model.eval_iterations / 10) + 1), losses=sine_model.losses)

    # c)
    sine_val = util.get_dataset("sine_val")

    degree = 1
    learning_rate = 1e-3
    sine_model = PolynomialRegressionModel(degree=degree, learning_rate=learning_rate)
    sine_model.train(sine_train)

    validation_loss = sine_val.compute_average_loss(sine_model)
    training_loss = sine_train.compute_average_loss(sine_model)
    print(f'1. Degree: {degree}, Learning Rate: {learning_rate}, Average Validation Loss: {validation_loss}, Average Training Loss: {training_loss}')

    degree = 1
    learning_rate = 1e-4
    sine_model = PolynomialRegressionModel(degree=degree, learning_rate=learning_rate)
    sine_model.train(sine_train)

    validation_loss = sine_val.compute_average_loss(sine_model)
    print(f'2. Degree: {degree}, Learning Rate: {learning_rate}, Average Validation Loss: {validation_loss}, Average Training Loss: {training_loss}')

    degree = 1
    learning_rate = 1e-5
    sine_model = PolynomialRegressionModel(degree=degree, learning_rate=learning_rate)
    sine_model.train(sine_train)

    validation_loss = sine_val.compute_average_loss(sine_model)
    print(f'3. Degree: {degree}, Learning Rate: {learning_rate}, Average Validation Loss: {validation_loss}, Average Training Loss: {training_loss}')

    degree = 2
    learning_rate = 1e-4
    sine_model = PolynomialRegressionModel(degree=degree, learning_rate=learning_rate)
    sine_model.train(sine_train)

    validation_loss = sine_val.compute_average_loss(sine_model)
    print(f'4. Degree: {degree}, Learning Rate: {learning_rate}, Average Validation Loss: {validation_loss}, Average Training Loss: {training_loss}')

    degree = 2
    learning_rate = 1e-5
    sine_model = PolynomialRegressionModel(degree=degree, learning_rate=learning_rate)
    sine_model.train(sine_train)

    validation_loss = sine_val.compute_average_loss(sine_model)
    print(f'5. Degree: {degree}, Learning Rate: {learning_rate}, Average Validation Loss: {validation_loss}, Average Training Loss: {training_loss}')

    degree = 2
    learning_rate = 1e-6
    sine_model = PolynomialRegressionModel(degree=degree, learning_rate=learning_rate)
    sine_model.train(sine_train)

    validation_loss = sine_val.compute_average_loss(sine_model)
    print(f'6. Degree: {degree}, Learning Rate: {learning_rate}, Average Validation Loss: {validation_loss}, Average Training Loss: {training_loss}')

    degree = 3
    learning_rate = 1e-6
    sine_model = PolynomialRegressionModel(degree=degree, learning_rate=learning_rate)
    sine_model.train(sine_train)

    validation_loss = sine_val.compute_average_loss(sine_model)
    print(f'7. Degree: {degree}, Learning Rate: {learning_rate}, Average Validation Loss: {validation_loss}, Average Training Loss: {training_loss}')

    degree = 3
    learning_rate = 1e-7
    sine_model = PolynomialRegressionModel(degree=degree, learning_rate=learning_rate)
    sine_model.train(sine_train)

    validation_loss = sine_val.compute_average_loss(sine_model)
    print(f'8. Degree: {degree}, Learning Rate: {learning_rate}, Average Validation Loss: {validation_loss}, Average Training Loss: {training_loss}')

    degree = 4
    learning_rate = 1e-8
    sine_model = PolynomialRegressionModel(degree=degree, learning_rate=learning_rate)
    sine_model.train(sine_train)

    validation_loss = sine_val.compute_average_loss(sine_model)
    print(f'9. Degree: {degree}, Learning Rate: {learning_rate}, Average Validation Loss: {validation_loss}, Average Training Loss: {training_loss}')

    degree = 4
    learning_rate = 1e-9
    sine_model = PolynomialRegressionModel(degree=degree, learning_rate=learning_rate)
    sine_model.train(sine_train)

    validation_loss = sine_val.compute_average_loss(sine_model)
    print(f'10. Degree: {degree}, Learning Rate: {learning_rate}, Average Validation Loss: {validation_loss}, Average Training Loss: {training_loss}')

# PA4 Q3
class BinaryLogisticRegressionModel(Model):
    """
    Binary logistic regression model with image-pixel features (num_features = image size, e.g., 28x28 = 784 for MNIST).
    x is a 2-D image, represented as a list of lists (28x28 for MNIST). y is either 0 or 1.
    The goal is to fit P(y = 1 | x) = hypothesis(x), and to make a 0/1 prediction using the hypothesis.
    """

    def __init__(self, num_features, learning_rate=1e-2):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.weights = np.random.randn(num_features + 1)
        self.train_accuracies = []
        self.test_accuracies = []
        self.eval_iterations = 250

    def get_features(self, x):
        features = np.array([1] + [pixel for row in x for pixel in row])
        return features

    def get_weights(self):
        return self.weights

    def hypothesis(self, x):
        return 1 / (1 + np.exp(-1 * np.dot(self.weights, self.get_features(x))))

    def predict(self, x):
        return 1 if self.hypothesis(x) >= 0.5 else 0

    def loss(self, x, y):
        hx = self.hypothesis(x)
        return y * np.log(hx) + (1 - y) * np.log(1 - hx)

    def gradient(self, x, y):
        predicted_y = self.hypothesis(x)
        features = self.get_features(x)
        gradient = [(y - predicted_y) * feature for feature in features]
        return gradient

    def train(self, dataset, evalset=None):
        x_train = dataset.xs
        y_train = dataset.ys

        for _ in range(self.eval_iterations):
            for i in range(len(x_train)):
                x = x_train[i]
                y = y_train[i]

                # Compute gradient and update weights
                grad = self.gradient(x, y)
                self.weights += (self.learning_rate * np.array(grad))

            if evalset is not None:
                self.test_accuracies.append(evalset.compute_average_accuracy(self))
            self.train_accuracies.append(dataset.compute_average_accuracy(self))


# PA4 Q4
def binary_classification():
    mnist_binary_train = util.get_dataset("mnist_binary_train")
    mnist_binary_test = util.get_dataset("mnist_binary_test")
    mnist_binary_model = BinaryLogisticRegressionModel(num_features=784)
    mnist_binary_model.train(mnist_binary_train, mnist_binary_test)

    #a) Accuracy Curves
    mnist_binary_train.plot_accuracy_curve(eval_iters=np.arange(1, mnist_binary_model.eval_iterations + 1), accuracies=mnist_binary_model.train_accuracies, title='Accuracy Curve on Training')
    mnist_binary_test.plot_accuracy_curve(eval_iters=np.arange(1, mnist_binary_model.eval_iterations + 1), accuracies=mnist_binary_model.test_accuracies, title='Accuracy Curve on Testing')

    #b) Confusion Matrix
    mnist_binary_test.plot_confusion_matrix(mnist_binary_model)

    #c) Non Bias-term weights
    mnist_binary_train.plot_image(x=mnist_binary_model.weights[1:])
    # The plot shows that the model has a hard time differing between 7s and 9s

    #d) Plotting 10 errors
    xs, ys = mnist_binary_test.get_all_samples()
    error_count = 0

    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        pred = mnist_binary_model.predict(x)

        if pred != y:
            mnist_binary_test.plot_image(x=x)
            error_count += 1

        if error_count >= 10:
            break


# PA4 Q5
class MultiLogisticRegressionModel(Model):
    """
    Multinomial logistic regression model with image-pixel features (num_features = image size, e.g., 28x28 = 784 for MNIST).
    x is a 2-D image, represented as a list of lists (28x28 for MNIST). y is an integer between 1 and num_classes.
    The goal is to fit P(y = k | x) = hypothesis(x)[k], where hypothesis is a discrete distribution (list of probabilities)
    over the K classes, and to make a class prediction using the hypothesis.
    """

    def __init__(self, num_features, num_classes, learning_rate=1e-2):
        self.num_features = num_features
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weights = np.random.randn(num_features + 1, num_classes)
        self.train_accuracies = []
        self.test_accuracies = []
        self.eval_iterations = 500

    def get_features(self, x):
        features = np.array([1] + [pixel for row in x for pixel in row])
        return features

    def get_weights(self):
        return self.weights

    def hypothesis(self, x):
        features = self.get_features(x)
        return (1 / np.sum(np.exp(np.dot(features, self.weights)))) * np.exp(np.dot(features, self.weights))

    def predict(self, x):
        probabilities = self.hypothesis(x)
        return np.argmax(probabilities)

    def loss(self, x, y):
        return -np.log(self.hypothesis(x)[y])

    def gradient(self, x, y):
        features = self.get_features(x)
        probabilities = self.hypothesis(x)

        gradient = np.zeros_like(self.weights).T

        for k in range(self.num_classes):
            gradient[k] += (features * probabilities[k])

        gradient[y] -= features  # Subtract features for the correct class

        return gradient.T

    def train(self, dataset, evalset=None):
        x_train = dataset.xs
        y_train = dataset.ys

        for iteration in range(self.eval_iterations):
            for i in range(len(x_train)):
                x = x_train[i]
                y = y_train[i]

                gradient = self.gradient(x, y)
                self.weights -= (self.learning_rate * gradient)

            if evalset is not None:
                self.test_accuracies.append(evalset.compute_average_accuracy(self))
            self.train_accuracies.append(dataset.compute_average_accuracy(self))

            if iteration % 10 == 0:
                print(f'iteration #{iteration}')


# PA4 Q6
def multi_classification():
    mnist_train = util.get_dataset("mnist_train")
    mnist_test = util.get_dataset("mnist_test")
    mnist_model = MultiLogisticRegressionModel(num_features=784, num_classes=10)
    mnist_model.train(mnist_train, mnist_test)

    #a) Accuracy Curves
    mnist_train.plot_accuracy_curve(eval_iters=np.arange(1, mnist_model.eval_iterations + 1), accuracies=mnist_model.train_accuracies, title='Accuracy Curve on Training')
    mnist_test.plot_accuracy_curve(eval_iters=np.arange(1, mnist_model.eval_iterations + 1), accuracies=mnist_model.test_accuracies, title='Accuracy Curve on Testing')

    #b) Confusion Matrix
    mnist_test.plot_confusion_matrix(mnist_model)

    #c) Non Bias-term weights
    mnist_train.plot_image(x=mnist_model.weights.T[0][1:])
    mnist_train.plot_image(x=mnist_model.weights.T[1][1:])
    mnist_train.plot_image(x=mnist_model.weights.T[2][1:])
    mnist_train.plot_image(x=mnist_model.weights.T[3][1:])
    mnist_train.plot_image(x=mnist_model.weights.T[4][1:])
    mnist_train.plot_image(x=mnist_model.weights.T[5][1:])
    mnist_train.plot_image(x=mnist_model.weights.T[6][1:])
    mnist_train.plot_image(x=mnist_model.weights.T[7][1:])
    mnist_train.plot_image(x=mnist_model.weights.T[8][1:])
    mnist_train.plot_image(x=mnist_model.weights.T[9][1:])





def main():
    linear_regression()
    binary_classification()
    multi_classification()


if __name__ == "__main__":
    main()
