import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random

def get_dataset(name):
    """
    Load a dataset by name. Returns a Dataset object.
    """
    # Change the mnist_path if MNIST data is extracted elsewhere
    mnist_path = "mnist_png/"
    if name == "sine_train":
        return RegressionDataset(random_seed = 0)
    elif name == "sine_val":
        return RegressionDataset(random_seed = 1)
    elif name == "sine_train_val":
        train = get_dataset("sine_train")
        val = get_dataset("sine_val")
        train.union(val)
        return train
    elif name == "sine_test":
        return RegressionDataset(random_seed = 2)
    elif name == "ex4q5i":
        return LogisticDataset(1)
    elif name == "ex4q5ii":
        return LogisticDataset(2)
    elif name == "ex4q5iii":
        return LogisticDataset(3)
    elif name == "ex4q5iv":
        return LogisticDataset(4)
    elif name == "mnist_binary_train":
        return BinaryMNISTDataset(mnist_path + "training/", classes = [7,9])
    elif name == "mnist_binary_test":
        return BinaryMNISTDataset(mnist_path + "/testing/", classes = [7,9])
    elif name == "mnist_train":
        return MNISTDataset(mnist_path + "training/")
    elif name == "mnist_test":
        return MNISTDataset(mnist_path + "testing/")
    else:
        print("ERROR: Dataset name not recognized.")


class Dataset:
    """
    Abstract class for a machine learning dataset.
    """

    def __init__(self):
        self.xs = [None]
        self.ys = [None]
        self.index = 0

    def get_size(self):
        return len(self.xs)

    def get_sample(self):
        sample = (self.xs[self.index], self.ys[self.index])
        self.index = 0 if self.index == self.get_size()-1 else self.index + 1
        return sample

    def get_samples(self, batch_size = 1):
        return [self.get_sample() for _ in range(batch_size)]

    def get_all_samples(self):
        return self.xs, self.ys

    def union(self, other):
        other.xs, other.ys = other.get_all_samples()
        self.xs += other.xs
        self.ys += other.ys

    def compute_average_loss(self, model, step = 1):
        return np.average([model.loss(self.xs[i], self.ys[i]) for i in range(0, self.get_size(), step)])

    def compute_average_accuracy(self, model, step = 1):
        return np.average([1 if model.predict(self.xs[i]) == self.ys[i] else 0 for i in range(0, self.get_size(), step)])

class RegressionDataset(Dataset):
    """
    Dataset used for linear regression (sine).
    """

    def __init__(self, num_samples = 100, random_seed = 0):
        np.random.seed(random_seed)
        xs = 10 * np.random.random_sample((num_samples,))
        ys = np.sin((2*np.pi/25) * xs) + 0.5 *(np.random.random_sample(xs.shape) - 0.5)
        self.xs = xs.tolist()
        self.ys = ys.tolist()
        self.index = 0

    def plot_data(self, model = None):
        plt.scatter(self.xs, self.ys, c = 'k')
        if model is not None:
            xs_plot = np.linspace(0, 10, 100)
            plt.plot(xs_plot, [model.hypothesis(x) for x in xs_plot], c = 'r')
        plt.show()
    
    def plot_loss_curve(self, eval_iters, losses, title = None):
        plt.plot(eval_iters, losses)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        if title is not None:
            plt.title(title)
        plt.show()


class MNISTDataset(Dataset):
    """
    Dataset used for multi-class classification (MNIST).
    """

    def __init__(self, data_path = "mnist_png/training/", classes = range(10), class_labels = None, random_seed = 0):
        random.seed(random_seed)
        self.classes = classes
        self.class_labels = [str(k) for k in self.classes] if class_labels is None else class_labels
        assert len(self.classes) == len(self.class_labels)
        
        data = []
        for k in self.classes:
            data_path_k = data_path + str(k) + '/'
            files = os.listdir(data_path_k)
            for file in files:
                im = Image.open(data_path_k + file)
                im_array = np.asarray(im) / 255.
                pixels = im_array.tolist()
                data += [(pixels, k)]

        random.shuffle(data)
        self.xs, self.ys = zip(*data)
        self.index = 0

    def plot_accuracy_curve(self, eval_iters, accuracies, title = None):
        plt.plot(eval_iters, accuracies)
        plt.ylim([0,1])
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        if title is not None:
            plt.title(title)
        plt.show()

    def plot_image(self, x):
        assert np.array(x).size == 28*28
        im = np.reshape(x, (28,28))
        plt.imshow(im)
        plt.colorbar()
        plt.show()

    def plot_confusion_matrix(self, model, step = 1, show_diagonal = False):
        confusion = [[0] * len(self.classes) for _ in self.classes]
        num_correct = 0
        num_evaluated = 0
        for i in range(0, self.get_size(), step):
            class_prediction = model.predict(self.xs[i])
            class_actual = self.ys[i]
            num_correct += 1 if class_prediction == class_actual else 0
            num_evaluated += 1
            confusion[class_actual][class_prediction] += 1
        print("Accuracy:", num_correct / num_evaluated, '%  (', num_correct, 'out of', num_evaluated, ')')

        print("Confusion matrix:")
        print("  ", *self.class_labels)
        for k in range(len(self.classes)):
            print(self.class_labels[k], confusion[k])

        confusion_plot = confusion
        if not show_diagonal:
            for k in range(len(self.classes)):
                confusion_plot[k][k] = 0
        plt.imshow(confusion_plot)
        ax = plt.gca()
        ax.set_xticks(range(len(self.classes)), self.class_labels)
        ax.set_yticks(range(len(self.classes)), self.class_labels)
        ax.tick_params(top = True, labeltop = True, bottom = False, labelbottom = False)
        plt.xlabel("Predicted class")
        ax.xaxis.set_label_position('top')
        plt.ylabel("Actual class")
        plt.title("Confusion matrix " + ("(including diagonal)" if show_diagonal else "(off-diagonal only)"))
        plt.colorbar()
        plt.show()


class BinaryMNISTDataset(MNISTDataset):
    """
    Dataset used for binary classification (MNIST).
    """

    def __init__(self, data_path = "mnist_png/training/", classes = range(2), class_labels = None, random_seed = 0):
        assert len(classes) == 2
        super().__init__(data_path, classes, class_labels, random_seed)
        self.ys = [0 if y == self.classes[0] else 1 for y in self.ys]


class LogisticDataset(Dataset):
    """
    Dataset used for binary logistic regression (Ex4 Q5(a)).
    """

    def __init__(self, dataset_id):
        np.random.seed(0)
        self.xs = [[np.random.rand(), np.random.rand()] for _ in range(1000)]
        if dataset_id == 1:
            self.ys = [1 if x[0] < 0.5 else 0 for x in self.xs]
        elif dataset_id == 2:
            self.ys = [1 if x[0] + x[1] > 1 else 0 for x in self.xs]
        elif dataset_id == 3:
            self.ys = [1 if (x[0]-0.5)**2 + (x[1]-0.5)**2 < 0.25**2 else 0 for x in self.xs]
        elif dataset_id == 4:
            self.ys = [1 if (x[0] < 0.5 and x[1] > 0.5) or (x[0] > 0.5 and x[1] < 0.5) else 0 for x in self.xs]
        else:
            "ERROR: Dataset ID not recognized."
        self.index = 0
