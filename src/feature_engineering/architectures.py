from torch.nn import Module, Sequential, Linear, ReLU, Softmax, LSTM


class MLPForTextTopicClassification(Module):
    def __init__(self, num_classes):
        super().__init__()

        self.sequential = Sequential(
            [
                Linear(1, 40),
                ReLU(),
                Linear(40, 200),
                ReLU(),
                Linear(40, 40),
                ReLU(),
                Linear(40, num_classes),
                Softmax(),
            ]
        )

    def forward(self, inputs):
        outputs = self.sequential(inputs)
        return outputs


class RecurrentModelForTextTopicClassification(Module):
    def __init__(self, num_classes, input_size, num_layers):
        super().__init__()
        self.hidden_size = 256

        self.lstm = LSTM(input_size, self.hidden_size, num_layers, bidirectional=True)
        self.linear = Linear(self.hidden_size * 2, num_classes)
        self.softmax = Softmax()

    def forward(self, inputs):
        _, (lstm_outputs, _) = self.lstm.forward(inputs)
        linear_outputs = self.linear(lstm_outputs)
        outputs = self.softmax(linear_outputs)
        return outputs


def choose_suitable_architecture(num_classes, input_shape):
    if len(input_shape) == 0:
        return MLPForTextTopicClassification(num_classes)
    input_size = input_shape[-1]
    num_layers = 4
    return RecurrentModelForTextTopicClassification(num_classes, input_size, num_layers)
