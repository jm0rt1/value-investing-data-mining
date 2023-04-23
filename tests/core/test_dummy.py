
import unittest


class TestDummy(unittest.TestCase):

    def test_fails(self):
        self.assertTrue(False)

    def test_passes(self):
        self.assertTrue(True)
