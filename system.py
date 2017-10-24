#!/usr/bin/python

import matplotlib.pyplot as plot
import numpy

BEGIN = 1
END = 10
TOL = 0.01

def generate_positive_matrix(n):
    numbers = numpy.random.uniform(BEGIN, END, n ** 2)
    matrix = numpy.matrix([[numbers[i * n + j] for j in range(n)] for i in range(n)])
    diagonal = numpy.diag(numpy.random.uniform(0, 1, n))
    return matrix * diagonal * matrix.T

def generate_vector(n):
    return numpy.matrix(numpy.random.uniform(BEGIN, END, n)).T

def solve(A, f):
    solution = numpy.linalg.solve(A, f)
    A_lambdas = numpy.linalg.eig(A)[0]
    tau = 2 / (max(A_lambdas) + min(A_lambdas))
    B = numpy.eye(len(f)) - tau * A
    B_lambdas = numpy.linalg.eig(B)[0]

    x = f.copy()
    counter = 0
    while numpy.linalg.norm(x - B * x - f * tau, 2) > TOL:
        x = B * x + f * tau
        if counter == 0:
            print("Theoretical N:", int(numpy.log(TOL / numpy.linalg.norm(f - x, 2)) / numpy.log(numpy.linalg.norm(B, 2))) + 1)
        counter += 1

    print("N:", counter)
    print("dalta:", numpy.linalg.norm(x - solution, numpy.inf))
    print("lambdas for B:", B_lambdas)
    print("tau:", tau)
    return x


def function(n):
    solve(generate_positive_matrix(n), generate_vector(n))

function(3)
