import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from numpy import sqrt, pi, matmul

from abcd import waist_eq, waist2, thin_lens, s_mirror, prop, loopMat
from beam_prop import make_plots

w0 = 5.2E-6 # input waist size in meters
z0 = 0.0 # input location in meters (can't be zero because I get a div/0 error)
d1 = 0.009605 # progation distance from fiber to collimating lens
f1 = 9.6E-3 # focal length of lens 1
d2 = 0.8 # propagation distance from collimating lens to first spherical mirror
roc1 = 2 # radius of curvature of spherical mirror 1
d3 = 1.7435 # distance between spherical mirrors
roc2 = 2 # radius of curvature of spherical mirror 2
λ = 1550E-9

loop_len = 299792458*10E-9

start = 0
stop = 1

s = 'static'
v = 'variable'


ops_to_input = [[prop, d1, 'green', s],[thin_lens, f1, None, s]]
fit_ops = [item[0] for item in ops_to_input]
fit_params = [item[1] for item in ops_to_input]

ops_to_output = [[prop, d1, 'green', s],[thin_lens, f1, None, s],[prop, d2, 'red', s],[s_mirror, roc1, None, s],[prop, d3, 'green', s],[s_mirror, roc2, None, s],[prop, 1, 'green', s]]
ops = [item[0] for item in ops_to_output]
params = [item[1] for item in ops_to_output]

def op_params(color, *args):
    ret = []
    for arg in args:
        ret.append([arg[0], arg[1], color, s])
    return ret



#Experimental x and y data points   
def xData(start, stop, num_points=100):
    step = (stop-start)/num_points
    return(np.array([step*i for i in range(num_points+1)])) 


# waist2(λ, w0, z0, matmul(prop(x),loopMat(prog_ops, prog_params)))
def yData(op_params, start, stop, λ=1550E-9, w0=5.2E-6, z0=0.0):
    d = sum(item[1] for item in op_params if item[0].__name__ == 'prop')
    ops = [item[0] for item in op_params]
    params = [item[1] for item in op_params]
    return np.array([waist2(λ, w0, z0, matmul(prop(xVal-d),loopMat(ops, params))) for xVal in xData(start, stop)])




func_list = []

for i in range(20):
    d = 0.009595 + i*(2E-5)/20
    # Fit function
    def fit_func(x, d2, d3, d1=d, λ=1550E-9, w0=5.2E-6, z0=0.0, f1=9.6E-3, roc1=2, roc2=2):
        op_params = [[prop, d1, 'green', s],[thin_lens, f1, None, s],[prop, d2, 'red', s],[s_mirror, roc1, None, s],[prop, d3, 'green', s],[s_mirror, roc2, None, s]]
        ops = [item[0] for item in op_params]
        params = [item[1] for item in op_params]
        return waist2(λ, w0, z0, matmul(prop(x-(d1+d2+d3-loop_len)),loopMat(ops, params)))
    func_list.append([fit_func, d])

res = []

for function in func_list:
    #Plot experimental data points
    ops_to_input = [[prop, function[1], 'green', s],[thin_lens, f1, None, s]]
    # plt.plot(xData(start, stop), yData(ops_to_input, start, stop), 'bo', label='experimental-data')
    # Initial guess for the parameters
    initialGuess = [0.8, 1.7435]    
    #Perform the curve-fit
    popt, pcov = curve_fit(function[0], xData(start, stop), yData(ops_to_input, start, stop), initialGuess)
    print([function[1], *popt])
    res.append([function[1], *popt])
    #x values for the fitted function
    xFit = np.arange(0.0, 1.0, 0.01)
    #Plot the fitted function
    # plt.plot(xFit, function[0](xFit, *popt))

# plt.show()

for i, item in enumerate(res):
    d1, d2, d3 = item
    num = len(res)
    o_p = op_params((0.0+(1/num)*i, 0.0, 1.0-(1/num)*i), *[[prop, d1],[thin_lens, f1],[prop, d2],[s_mirror, roc1],[prop, d3],[s_mirror, roc2],[prop, 1]])
    make_plots(o_p, w0, z0, start_buffer=0, stop_buffer=0)

plt.show()