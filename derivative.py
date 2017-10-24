#!/usr/bin/python

import matplotlib.pyplot as plot
import numpy

N = 1000 # points to plot real function graph

'''
Finding interpolation polynomial as c_0 + c_1 * x + ... + c_N * x^N

Interpolation condition:

    c_0 + c_1 * x_0 + ... + c_N * x_0^N = f_0
    c_0 + c_1 * x_1 + ... + c_N * x_1^N = f_1
                      ...
    c_0 + c_1 * x_N + ... + c_N * x_N^N = f_N

Сonsider this linear equations system as V * c  = f, where

    c = (c_0, c_1, ..., c_N)
    f = (f_0, f_1, ..., f_N)
    V - vandermond matrix

Now we can find c as V^-1 * f
'''
def get_interpolation_coefficients(points, function):
    degree = len(points) - 1 # maximum approximation order
    vandermond_matrix = numpy.matrix(numpy.vander(points, increasing=True))
    return numpy.array((vandermond_matrix.I * numpy.matrix(function).T).T)[0][::-1]

def make_polynomial(coefficients):
    return (lambda x: sum([coefficients[i] * x ** (len(coefficients) - i - 1) for i in range(len(coefficients))]))

def polynomial_derivative(coefficients):
    return make_polynomial([coefficient * (len(coefficients) - number - 1)  for number, coefficient in enumerate(coefficients[:-1])])

def find_derivative(function, points, point):
    interpolation_coefficients = get_interpolation_coefficients(points, function(points))
    # numpy.polyfit(points, function(points), degree)
    interpolation_function = make_polynomial(interpolation_coefficients)
    # interpolation_function = lambda x: numpy.poly1d(interpolation_coefficients)(x)
    derivative_function = polynomial_derivative(interpolation_coefficients)
    # derivative_function(x) = lambda x: numpy.polyder(numpy.poly1d(interpolation_coefficients))(x)

    print("f'({})={}\n".format(point, derivative_function(point)))

    continuous_points = numpy.linspace(points.min(), points.max(), N)
    plot.figure()
    plot.plot(continuous_points, function(continuous_points), label="Real function")
    plot.subplot()
    plot.plot(continuous_points, interpolation_function(continuous_points), label="Interpolated function")
    plot.subplot()
    plot.scatter(points, function(points), label="Nodes")

    '''
    # Here we can plot derivative
    plot.subplot()
    plot.plot(continuous_points, derivative_function(continuous_points), label="Derivative function")
    '''

    axis = plot.gca()
    axis.set_xlim(min(points) - 1, max(points) + 1)
    axis.set_ylim(min(interpolation_function(continuous_points)) - 1, max(interpolation_function(continuous_points)) + 1)
    plot.legend()
    plot.show()

    return derivative_function(point)

x = numpy.linspace(0, 6, 100)

def f(x):
    return abs(x) * numpy.sin(x)

find_derivative(f, x, 2.5)
