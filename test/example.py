import unittest

# TODO(Process-ing): Erase this from existence
class TestStringMethods(unittest.TestCase):

    def test_upper(self) -> None:
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self) -> None:
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self) -> None:
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
