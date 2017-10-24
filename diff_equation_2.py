#!/usr/bin/python

import matplotlib.pyplot as plot
import numpy

def f(x, u):
    return -3 * u + 2 * numpy.exp(-3 * x) + x

begin = 0
end = 2
u_0 = -1
N_0 = 100

s = 3 # Число стадий

b = [1/6, 2/3, 1/6]
c = [0, 1/2, 1]
a = [[0, 0, 0], [1/2, 0, 0], [-1, 2, 0]]

def get_real_u(x):
    return (18 * x + numpy.exp(3 * x) * (3 * x - 1) - 8) * numpy.exp(-3 * x) / 9

def get_u(N):
    h = (end - begin) / (N - 1)
    x = numpy.linspace(begin, end, N)

    u = [u_0]
    for n in range(1, N):
        k = []
        for i in range(s):
            k.append(f(x[n - 1] + c[i] * h, u[n - 1] + h * numpy.sum([a[i][j] * k[j] for j in range(i)])))
        u.append(u[n - 1] + h * numpy.sum([b[j] * k[j] for j in range(s)]))

    return numpy.array(u)

x = numpy.linspace(begin, end, N_0)
print("delta =", numpy.linalg.norm(get_real_u(x) - get_u(N_0), numpy.inf))

plot.plot(x, get_u(N_0), label="estimated")
plot.plot(numpy.linspace(begin, end, 1000), get_real_u(numpy.linspace(begin, end, 1000)), label="real")
axis = plot.gca()
axis.set_xlabel("x")
axis.set_ylabel("u")
axis.legend()

steps = 10
deltas = numpy.array([numpy.linalg.norm(get_u(2 ** i * N_0) - get_real_u(numpy.linspace(begin, end, 2 ** i * N_0)), numpy.inf) for i in range(steps)])

print(deltas)

plot.figure()
plot.scatter(numpy.log([1 / (2 ** i * N_0) for i in range(steps)]), numpy.log(numpy.array(deltas)))
axis = plot.gca()
axis.set_xlabel("log(delta_i)")
axis.set_ylabel("log(h_i)")

A = numpy.matrix([[numpy.log(1 / (2 ** i * N_0)), 1] for i in range(steps)])
b = numpy.matrix([[numpy.log(delta)] for delta in deltas])

p = ((A.T * A).I * A.T * b)[0]
print("p =", numpy.array(p)[0][0])

plot.show()
