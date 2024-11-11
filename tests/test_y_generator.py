from unittest import TestCase

from tests.y_generator import generate_marked_frame


class Test_Y_Generator(TestCase):
    def test_generate_marked_frame(self):
        f = generate_marked_frame(8, 2, 0, 0, 1, 5)
        print(f)
