import unittest
from app.usecases.example_functions.example_functions import hello_world
from app.usecases.example_functions.relative_import.relative_import import call_hello_world

class TestHelloWorld(unittest.TestCase):
    def test_hello_world(self):
        self.assertEqual(hello_world(), "Hello World!")
    def test_relative_import(self):
        self.assertEqual(call_hello_world(), "Hello World!")
if __name__ == '__main__':
    unittest.main()