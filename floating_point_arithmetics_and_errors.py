import math
import numpy as np


# the functions
def f(x, y):
    try:
        result = math.sqrt(x ** 2 + y ** 2) - x
    except OverflowError:
        result = float('inf')
    return result


def f_changed(x, y):
    """Computes the expected result of f(x, y) = sqrt(x^2 + y^2) - x without floating-point errors."""
    # f(x, y) = sqrt(x^2 + y^2) - x = y^2 / (sqrt(x^2 + y^2) + x)
    try:
        result = (y * y) / (math.sqrt(x ** 2 + y ** 2) + x)
    except OverflowError:
        result = float('inf')
    except ZeroDivisionError:
        result = 0.0  # for the case where y is too small
    return result


# associative-sensitive function to show violation of associative property
def g_standard_order(x, y, z):
    return (x + y) + z


def g_reordered(x, y, z):
    return x + (y + z)


# test cases for each error type
test_cases = {
    "Round-Off Error": [
        {"x": 1e16, "y": 1e-8},  # large x, very small y
    ],
    "Underflow": [
        {"x": -1.79769e+308, "y": -1.79769e+308, "z": 0},  # values lower than the smallest double value
    ],
    "Overflow": [
        {"x": 1e308, "y": 1e308},  # very large x and y
    ],
    "Associative Property Violation": [
        {"x": 1e16, "y": -1e16, "z": 1},  # large x and y with opposite signs and a small z
    ],
}


# function that runs test cases
def run_tests(test_cases):
    for error_type, cases in test_cases.items():
        print(f"\n=== {error_type} ===")
        for case in cases:
            if error_type == "Associative Property Violation":
                x, y, z = case["x"], case["y"], case["z"]
                result_standard = g_standard_order(x, y, z)
                result_reordered = g_reordered(x, y, z)
                print(f"g_standard_order({x}, {y}, {z}) = {result_standard}")
                print(f"g_reordered({x}, {y}, {z}) = {result_reordered}")
            elif error_type == "Underflow":
                x, y, z = case["x"], case["y"], case["z"]
                result = g_standard_order(x, y, z)
                print(f"f({x}, {y}, {z}) = {result}")
            elif error_type == "Round-Off Error":
                x, y = case["x"], case["y"]
                result_original = f(x, y)
                result_changed = f_changed(x, y)
                print(f"f_original({x}, {y}) = {result_original}")
                print(f"f_changed({x}, {y}) = {result_changed}")
            else:
                x, y = case["x"], case["y"]
                result = f(x, y)
                print(f"f({x}, {y}) = {result}")


run_tests(test_cases)
