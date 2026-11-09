import unittest
import numpy as np

from app.usecases.artificial_neuron.perceptron import Perceptron


class TesPerceptronNetInput(unittest.TestCase):
    def test_net_input_1_dimention(self):
        p = Perceptron()
        p.weights = np.array([1.0, 2.0, 3.0])
        x = np.array([4.0, 5.0])
        net_input = p.net_input(x)
        print(type(net_input))
        self.assertEqual(net_input, 4.0*2.0 + 5.0*3.0 + 1.0)
    
    def test_net_input_2_dimention(self):
        p = Perceptron()
        p.weights = np.array([1.0, 2.0, 3.0])
        x = np.array([[4.0, 5.0], [6.0, 7.0]])
        net_input = p.net_input(x)
        print(type(net_input))
        self.assertEqual(net_input[0], 4.0*2.0 + 5.0*3.0 + 1.0)
        self.assertEqual(net_input[1], 6.0*2.0 + 7.0*3.0 + 1.0)

class TestPerceptronPredict(unittest.TestCase):
    def setUp(self):
        self.perceptron = Perceptron()
        self.perceptron.weights = np.array([0.5, -0.5, 0.5])

    def test_predict_1(self):
        x_test = np.array([[2, 3], [1, -1], [0, 0]])
        expected_predictions = np.array([1, -1, 1])

        for x, expected in zip(x_test, expected_predictions):
            prediction = self.perceptron.predict(x)
            self.assertEqual(prediction, expected, f"Incorrect prediction for {x}: expected {expected}, obtained {prediction}")

class TestPerceptronFit(unittest.TestCase):
    def test_fit_initialization(self):
        # Crear una instancia de Perceptron
        perceptron = Perceptron(eta=0.01, n_iter=10, random_state=1)

        # Datos de entrenamiento de prueba
        x_train = np.array([[1, 2], [2, 3], [3, 4]])
        y_train = np.array([1, -1, 1])

        # Verificar que los pesos y los errores aún no se han inicializado
        self.assertIsNone(getattr(perceptron, 'weights', None), "Los pesos ya están inicializados antes del entrenamiento.")
        self.assertIsNone(getattr(perceptron, 'errors', None), "La lista de errores ya está inicializada antes del entrenamiento.")

        # Llamar al método fit
        perceptron.fit(x_train, y_train)

        # Verificar si los pesos y los errores se han inicializado correctamente
        self.assertIsNotNone(perceptron.weights, "Los pesos no se inicializaron después del entrenamiento.")
        self.assertIsInstance(perceptron.weights, np.ndarray, "Los pesos no son un ndarray.")
        self.assertIsNotNone(perceptron.errors, "La lista de errores no se inicializó después del entrenamiento.")
        self.assertIsInstance(perceptron.errors, list, "Los errores no son una lista.")
        self.assertEqual(len(perceptron.errors), perceptron.n_iter, "El número de entradas en la lista de errores no coincide con el número de iteraciones.")

if __name__ == '__main__':
    unittest.main()