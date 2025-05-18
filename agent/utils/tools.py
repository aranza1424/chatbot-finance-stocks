""" Define tools for model"""
import inspect

class Toolbox:
    @staticmethod
    def multiply(a: int, b: int) -> int:
        """Multiply a and b."""
        return a * b

    @staticmethod
    def add(a: int, b: int) -> int:
        """Adds a and b."""
        return a + b

    @staticmethod
    def divide(a: int, b: int) -> float:
        """Divide a and b."""
        return a / b
    


def get_tools(cls):
    return [
        func for nombre, func in inspect.getmembers(cls, predicate=inspect.isfunction)
        if not nombre.startswith("__")
    ]

tools = get_tools(Toolbox)