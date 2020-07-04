from numpy import exp, array, random, dot

# Grupo: José C. Pereira, Ademir José (@Azganoth)
# Classificação de vinho
#
# Amostras
# Input 01 [SECO]: Alcool 13,5%, Ph 3,41, Açucar  5,0g/l, Acidez 8,85g/l
# Input 02 [SECO]: Alcool 14.5%, Ph 3,30, Açucar  2,0g/l, Acidez 5,92g/l
# Input 03 [DOCE]: Alcool 11,5%, Ph 3,63, Açucar 50,0g/l, Acidez 5,32g/l
# Input 04 [DOCE]: Alcool 10,0%, Ph 3,10, Açucar 50,7g/l, Acidez 6,45g/l
# Input 05 [DOCE]: Alcool 10,0%, Ph 3,10, Açucar 42,2g/l, Acidez 7,12g/l
# Input 06 [DOCE]: Alcool  8,5%, Ph 3,21, Açucar 42,0g/l, Acidez 8,90g/l
# Input 07 [DOCE]: Alcool  8,5%, Ph 3,31, Açucar 42,0g/l, Acidez 5,85g/l
# Input 08 [SECO]: Alcool 14,0%, Ph 3,68, Açucar  3,3g/l, Acidez 5,50g/l
# Input 09 [SECO]: Alcool 14,0%, Ph 3,32, Açucar  3,6g/l, Acidez 6,10g/l
# Input 10 [SECO]: Alcool 14,0%, Ph 3,51, Açucar  3,2g/l, Acidez 5,98g/l
#
# VAL_OUTP
# SECO = 0
# DOCE = 1
#
# Dados: http://www.casavalduga.com.br/produtos/vinhos/
# Artigo: https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1

class NeuralNetwork:
    def __init__(self, n_neurons_input_layer: int, n_neurons_hidden_layer: int, n_neurons_output_layer: int):
        self.input_hidden_weights = random.rand(n_neurons_input_layer, n_neurons_hidden_layer)
        self.hidden_output_weights = random.rand(n_neurons_hidden_layer, n_neurons_output_layer)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def think(self, inputs):
        output_from_hidden_layer = self.sigmoid(dot(inputs, self.input_hidden_weights))
        output_from_output_layer = self.sigmoid(dot(output_from_hidden_layer, self.hidden_output_weights))
        return output_from_hidden_layer, output_from_output_layer

    def train(self, training_inputs, training_outputs, epochs):
        for epoch in range(epochs):
            output_from_hidden_layer, output_from_output_layer = self.think(training_inputs)

            output_layer_error = training_outputs - output_from_output_layer
            output_layer_delta = output_layer_error * self.sigmoid_derivative(output_from_output_layer)

            hidden_layer_error = output_layer_delta.dot(self.hidden_output_weights.T)
            hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(output_from_hidden_layer)

            self.input_hidden_weights += training_inputs.T.dot(hidden_layer_delta)

training_inputs = array([
    [0.135, 0.0341, 0.050, 0.885],
    [0.145, 0.0330, 0.020, 0.592],
    [0.115, 0.0363, 0.500, 0.532],
    [0.100, 0.0310, 0.507, 0.645],
    [0.100, 0.0310, 0.422, 0.712],
    [0.085, 0.0321, 0.420, 0.890],
    [0.085, 0.0331, 0.420, 0.585],
    [0.140, 0.0368, 0.033, 0.550],
    [0.140, 0.0332, 0.036, 0.610],
    [0.140, 0.0351, 0.032, 0.598]
])

training_outputs = array([[0, 0, 1, 1, 1, 1, 1, 0, 0, 0]]).T

# Instanciando rede neural
# 4 Neuronios camada de input
# 4 Neuronios camada escondida
# 1 Neuronio para output
neural_network = NeuralNetwork(4, 4, 1)

# Treinando a rede neural com 60 mil epocas
neural_network.train(training_inputs, training_outputs, 60000)

print(" Testando - Considerando a situação: \n\t  - Alcool: 0.120 \n\t  - Ph: 0.0340 \n\t  - Açucar: 0.380 \n\t  - Acidez: 0.300 \n ")
hidden_state, output = neural_network.think(array([0.120, 0.0340, 0.380, 0.300]))

print(" Resultado proximo de 0: Seco")
print(" Resultado proximo de 1: Doce/Suave")
print(" Resultado: ", output)
