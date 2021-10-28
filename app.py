# Matplotlib widget for calculating Euler, Improved_euler, Runga-Kutta methods

import matplotlib
import numpy
import math
import matplotlib.pyplot as plt

#---------------------------------TextField---------------------------------#

class TextField(object):
    def __init__(self, val, dims, str):
        self.val = val
        self.dims = dims
        self.str = str

    def draw(self):
        self.axes = matplotlib.pyplot.axes(self.dims)
        self.type = matplotlib.widgets.TextBox(self.axes, self.str + ':', initial = str(self.val))
        self.type.on_submit(self.on)

    def on(self, text):
        self.val = eval(text)

#-------------------------------GUI & CALCULATIONS-------------------------------#

class App(object):
    def __init__(self, solution):
        self.solution = solution
        for s in self.solution:
            if isinstance(s, Exact_solution):
                self.exact_solution = s

        self.figure = matplotlib.pyplot.figure(figsize=(9, 8), num="y' = e^ay - 2/bx")

        self.x0_TextField = TextField(self.exact_solution.x_0, [0.06, 0.86, 0.1, 0.04], "x0")
        self.x0_TextField.draw()
        self.y0_TextField = TextField(self.exact_solution.y_0, [0.06, 0.80, 0.1, 0.04], "y0")
        self.y0_TextField.draw()
        self.xf_TextField = TextField(self.exact_solution.x_BIG, [0.06, 0.74, 0.1, 0.04], "X")
        self.xf_TextField.draw()
        self.n_TextField = TextField(self.exact_solution.n, [0.06, 0.68, 0.1, 0.04], "n")
        self.n_TextField.draw()
        self.n_min_TextField = TextField(self.exact_solution.n_min, [0.06, 0.62, 0.1, 0.04], "min_n")
        self.n_min_TextField.draw()
        self.n_max_TextField = TextField(self.exact_solution.n_max, [0.06, 0.56, 0.1, 0.04], "max_n")
        self.n_max_TextField.draw()
        self.a_TextField = TextField(self.exact_solution.a, [0.06, 0.50, 0.1, 0.04], "a")
        self.a_TextField.draw()
        self.b_TextField = TextField(self.exact_solution.b, [0.06, 0.44, 0.1, 0.04], "b")
        self.b_TextField.draw()
        self.button = matplotlib.widgets.Button(matplotlib.pyplot.axes([0.06, 0.38, 0.1, 0.03]), "Change")
        self.button.on_clicked(self.on)  

        self.axes = self.figure.add_subplot(3, 1, 1)
        self.axes.set_title("Solutions of IVP")
        self.axes.set_xlabel("x")
        self.axes.set_ylabel("y")
        self.axes.grid(True)

        self.rescale(self.axes, self.x0_TextField.val, self.xf_TextField.val)

        for i in self.solution:
            i.graph(self.axes)
            i.calculate_error(self.exact_solution.y)
        matplotlib.pyplot.legend(loc = "upper right")

        # ERRORS

        self.axes_errors = self.figure.add_subplot(3, 1, 2)
        self.axes_errors.set_title("Errors of Numerical Methods")
        self.axes_errors.set_xlabel("x")
        self.axes_errors.set_ylabel("y")
        self.axes_errors.grid(True)

        self.rescale(self.axes_errors, self.x0_TextField.val, self.xf_TextField.val)

        for i in self.solution:
            if not isinstance(i, Exact_solution):
                i.graph_error(self.axes_errors)
        matplotlib.pyplot.legend(loc = "upper right")

        self.axes_max_errors = self.figure.add_subplot(3, 1, 3)
        self.axes_max_errors.set_title("Max errors of Numerical Methods")
        self.axes_max_errors.set_xlabel("n")
        self.axes_max_errors.set_ylabel("Total error")
        self.axes_max_errors.grid(True)

        self.rescale(self.axes_max_errors, self.n_min_TextField.val, self.n_max_TextField.val)

        n_values = numpy.linspace(self.n_min_TextField.val, self.n_max_TextField.val, self.n_max_TextField.val - self.n_min_TextField.val + 1)
        e_total_errors = numpy.zeros([self.n_max_TextField.val - self.n_min_TextField.val + 1])
        i_total_errors = numpy.zeros([self.n_max_TextField.val - self.n_min_TextField.val + 1])
        r_total_errors = numpy.zeros([self.n_max_TextField.val - self.n_min_TextField.val + 1])
        for i in range(self.n_min_TextField.val, self.n_max_TextField.val + 1):
            exact_solution = Exact_solution(function, self.x0_TextField.val, self.y0_TextField.val, self.xf_TextField.val, self.a_TextField.val, self.b_TextField.val, i)
            euler = Euler(function, self.x0_TextField.val, self.y0_TextField.val, self.xf_TextField.val, self.a_TextField.val, self.b_TextField.val, i)
            i_euler = Improved_Euler(function, self.x0_TextField.val, self.y0_TextField.val, self.xf_TextField.val, self.a_TextField.val, self.b_TextField.val, i)
            r_k = Runge_Kutta(function, self.x0_TextField.val, self.y0_TextField.val, self.xf_TextField.val, self.a_TextField.val, self.b_TextField.val, i)
            euler.calculate_error(exact_solution.y)
            i_euler.calculate_error(exact_solution.y)
            r_k.calculate_error(exact_solution.y)
            e_total_errors[i - self.n_min_TextField.val] = max(numpy.amax(euler.err), abs(numpy.amin(euler.err)))
            i_total_errors[i - self.n_min_TextField.val] = max(numpy.amax(i_euler.err), abs(numpy.amin(i_euler.err)))
            r_total_errors[i - self.n_min_TextField.val] = max(numpy.amax(r_k.err), abs(numpy.amin(r_k.err)))
            del exact_solution
            del euler
            del i_euler
            del r_k
        self.ax_max_err_0, = self.axes_max_errors.plot(n_values, e_total_errors, 'y-', label='Euler')
        self.ax_max_err_1, = self.axes_max_errors.plot(n_values, i_total_errors, 'r-', label='Improved Euler')
        self.ax_max_err_2, = self.axes_max_errors.plot(n_values, r_total_errors, 'b-', label='Runge-Kutta')

        matplotlib.pyplot.legend(loc = "upper right")

        self.lines = []
        for i in self.solution:
            self.lines.append(i.ax)

        self.lines_err = []
        for i in self.solution:
            if not isinstance(i, Exact_solution):
                self.lines_err.append(i.ax_err)

        self.lines_max_err = []
        self.lines_max_err.append(self.ax_max_err_0)
        self.lines_max_err.append(self.ax_max_err_1)
        self.lines_max_err.append(self.ax_max_err_2)

        rax = matplotlib.pyplot.axes([0.02, 0.20, 0.16, 0.15])
        self.labels = [str(line.get_label()) for line in self.lines]
        visibility = [line.get_visible() for line in self.lines]
        check = matplotlib.widgets.CheckButtons(rax, self.labels, visibility)
        check.on_clicked(self.tick)


        self.show()
            
    def recalculate(self):
        for s in self.solution:
            s.x_0 = self.x0_TextField.val
            s.y_0 = self.y0_TextField.val
            s.x_BIG = self.xf_TextField.val
            s.n = self.n_TextField.val
            s.n_min = self.n_min_TextField.val
            s.n_max = self.n_max_TextField.val
            s.a = self.a_TextField.val
            s.b = self.b_TextField.val
            s.recalculate(self.axes)
            if not isinstance(s, Exact_solution):
                s.calculate_error(self.exact_solution.y)

        print(self.a_TextField.val)
        print(self.b_TextField.val)
        n_values = numpy.linspace(self.n_min_TextField.val, self.n_max_TextField.val, self.n_max_TextField.val - self.n_min_TextField.val + 1)
        e_total_errors = numpy.zeros([self.n_max_TextField.val - self.n_min_TextField.val + 1])
        i_total_errors = numpy.zeros([self.n_max_TextField.val - self.n_min_TextField.val + 1])
        r_total_errors = numpy.zeros([self.n_max_TextField.val - self.n_min_TextField.val + 1])
        for i in range(self.n_min_TextField.val, self.n_max_TextField.val + 1):
            exact_solution = Exact_solution(function, self.x0_TextField.val, self.y0_TextField.val, self.xf_TextField.val, self.a_TextField.val, self.b_TextField.val, i)
            euler = Euler(function, self.x0_TextField.val, self.y0_TextField.val, self.xf_TextField.val, self.a_TextField.val, self.b_TextField.val, i)
            i_euler = Improved_Euler(function, self.x0_TextField.val, self.y0_TextField.val, self.xf_TextField.val, self.a_TextField.val, self.b_TextField.val, i)
            r_k = Runge_Kutta(function, self.x0_TextField.val, self.y0_TextField.val, self.xf_TextField.val, self.a_TextField.val, self.b_TextField.val, i)
            euler.calculate_error(exact_solution.y)
            i_euler.calculate_error(exact_solution.y)
            r_k.calculate_error(exact_solution.y)
            e_total_errors[i - self.n_min_TextField.val] = max(numpy.amax(euler.err), abs(numpy.amin(euler.err)))
            i_total_errors[i - self.n_min_TextField.val] = max(numpy.amax(i_euler.err), abs(numpy.amin(i_euler.err)))
            r_total_errors[i - self.n_min_TextField.val] = max(numpy.amax(r_k.err), abs(numpy.amin(r_k.err)))
            del exact_solution
            del euler
            del i_euler
            del r_k

        self.ax_max_err_0.set_data(n_values, e_total_errors)
        self.ax_max_err_1.set_data(n_values, i_total_errors)
        self.ax_max_err_2.set_data(n_values, r_total_errors)

        self.rescale(self.axes, self.x0_TextField.val, self.xf_TextField.val)
        self.rescale(self.axes_errors, self.x0_TextField.val, self.xf_TextField.val)
        self.rescale(self.axes_max_errors, self.n_min_TextField.val, self.n_max_TextField.val)

    def rescale(self, axes, start, end):
        axes.set_xlim(start - 0.3, end + 0.3)
        axes.relim()
        axes.autoscale_view(True,True,True)

    def on(self, event):
        self.recalculate()

    def tick(self, label):
        index = self.labels.index(label)
        index_err = index - 1
        self.lines[index].set_visible(not self.lines[index].get_visible())
        if label != 'Exact_solution':
            self.lines_err[index_err].set_visible(not self.lines_err[index_err].get_visible())
            self.lines_max_err[index_err].set_visible(not self.lines_max_err[index_err].get_visible())
        matplotlib.pyplot.draw()

    def show(self):
        matplotlib.pyplot.subplots_adjust(left=0.25, right=0.95, bottom=0.1, top=0.9, wspace=None, hspace=1.0)
        matplotlib.pyplot.show()

class Solution(object):
    def __init__(self, func, x_0, y_0, x_BIG, a, b, n):
        self.a = a
        self.b = b
        self.x_0 = x_0
        self.y_0 = y_0
        self.x_BIG = x_BIG
        self.n = n
        self.n_min = n - 10
        self.n_max = n + 10
        self.func = func
        self.h = (x_BIG - x_0) / (n)
        self.x = numpy.linspace(x_0, x_BIG, n + 1)
        self.y = numpy.zeros([n + 1])
        self.err = numpy.zeros([n + 1])
        self.y[0] = y_0
        self.ax = matplotlib.lines.Line2D([], [])
        self.ax_err = matplotlib.lines.Line2D([], [])
        self.solve()

    def recalculate(self, axes):
        self.x = numpy.linspace(self.x_0, self.x_BIG, self.n + 1)
        self.y = numpy.zeros([self.n + 1])
        self.h = (self.x_BIG - self.x_0) / (self.n)
        self.y[0] = self.y_0

        self.solve()
        self.ax.set_data(self.x, self.y)

    def calculate_error(self, y_exact_solution):
        self.err = y_exact_solution - self.y
        self.ax_err.set_data(self.x, self.err)

    def solve(self):
        pass

    def graph(self, axes):
        pass


class Exact_solution(Solution):
    def get_const(self):
        return (self.y_0 + 2 * self.x_0 - 1) / math.exp(self.x_0)

    def solve(self):
        const = self.get_const()
        for i in range(1, self.n + 1):
            self.y[i] = (-2) * self.x[i] + 1 + const * math.exp(self.x[i])

    def graph(self, axes):
        self.ax, = axes.plot(self.x, self.y, 'o-', label='Exact_solution', markersize=3)


class Euler(Solution):
    def solve(self):
        for i in range(1, self.n + 1):
            self.y[i] = self.h * self.func(self.x[i - 1], self.y[i - 1], self.a, self.b) + self.y[i - 1]

    def graph(self, axes):
        self.ax, = axes.plot(self.x, self.y, 'y-', label='Euler')

    def graph_error(self, axes):
        self.ax_err, = axes.plot(self.x, self.err, 'y-', label='Euler')


class Improved_Euler(Solution):
    def solve(self):
        for i in range(1, self.n + 1):
            self.y[i] = self.h * self.func(self.x[i - 1], self.y[i - 1], self.a, self.b) + self.y[i - 1]
            self.y[i] = (self.func(self.x[i - 1], self.y[i - 1], self.a, self.b) + self.func(self.x[i], self.y[i], self.a, self.b))
            self.y[i] = self.h * self.y[i] / 2 + self.y[i - 1]

    def graph(self, axes):
        self.ax, = axes.plot(self.x, self.y, 'r-', label='Improved Euler')

    def graph_error(self, axes):
        self.ax_err, = axes.plot(self.x, self.err, 'r-', label='Improved Euler')


class Runge_Kutta(Solution):
    def solve(self):
        k = numpy.zeros([4])
        for i in range(1, self.n + 1):
            k[0] = self.func(self.x[i - 1], self.y[i - 1], self.a, self.b)
            k[1] = self.func(self.x[i - 1] + self.h / 2, self.y[i - 1] + self.h * k[0] / 2, self.a, self.b)
            k[2] = self.func(self.x[i - 1] + self.h / 2, self.y[i - 1] + self.h * k[1] / 2, self.a, self.b)
            k[3] = self.func(self.x[i - 1] + self.h, self.y[i - 1] + self.h * k[2], self.a, self.b)
            self.y[i] = (self.h / 6) * (k[0] + 2 * k[1] + 2 * k[2] + k[3]) + self.y[i - 1]

    def graph(self, axes):
        self.ax, = axes.plot(self.x, self.y, 'b-', label='Runge-Kutta')

    def graph_error(self, axes):
        self.ax_err, = axes.plot(self.x, self.err, 'b-', label='Runge-Kutta')


# ------------------------------------FUNCTION------------------------------------#

# def function(x, y):
#     return (2.7**y -2/x)

def function(x, y, a, b):
    return (2.7**(float(a)*y)) -2/(float(b)*x)


def initiate_process(func, x_0, y_0, x_BIG, n):
    a = Exact_solution(func, x_0, y_0, x_BIG, 1, 1, n)
    b = Euler(func, x_0, y_0, x_BIG, 1, 1, n)
    c = Improved_Euler(func, x_0, y_0, x_BIG, 1, 1, n)
    d = Runge_Kutta(func, x_0, y_0, x_BIG, 1, 1, n)
    app = App([a, b, c, d])


#--------------------------------RUNNING APPLICATION---------------------------------#

if __name__ == "__main__":
    initiate_process(function, 1, -2, 7, 40)
