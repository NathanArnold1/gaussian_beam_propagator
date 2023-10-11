import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from numpy import sqrt, pi, matmul, inf

from abcd import waist_eq, waist2, thin_lens, s_mirror, prop, interface, loopMat
from beam_prop import make_plots, Op

w0 = 5.2E-6 # input waist size in meters
z0 = 0.0 # input location in meters (can't be zero because I get a div/0 error)
d1 = 0.008885 # progation distance from fiber to collimating lens
f1 = 9.6E-3 # focal length of lens 1
d2 = 0.78850762 # propagation distance from collimating lens to first spherical mirror
roc1 = 2 # radius of curvature of spherical mirror 1
d3 = 1.7435 # distance between spherical mirrors
roc2 = 2 # radius of curvature of spherical mirror 2
λ = 1550E-9
asphereIndex = 1.5677 # index of refraction for the collimating asphere
asphereTc = 2.493E-3 # center thickness of collimating asphere
asphereROC = -5.609928E-3 # Radius of curvature of asphere

loop_len = 299792458*10E-9

start = 0.1
stop = 1

s = 'static'
v = 'variable'


#Experimental x and y data points   
def xData(start, stop, num_points=100):
    step = (stop-start)/num_points
    return(np.array([step*i for i in range(num_points+1)])) 


# waist2(λ, w0, z0, matmul(prop(x),loopMat(prog_ops, prog_params)))
def yData(op_list, start, stop, λ=1550E-9, w0=5.2E-6, z0=0.0):
    d = sum(operation.args[0] for operation in op_list if operation.op.__name__ == 'prop')
    # d = sum(item[1] for item in op_params if item[0].__name__ == 'prop')
    ops = [operation.op for operation in op_list]
    args = [operation.args for operation in op_list]
    # ops = [item[0] for item in op_params]
    # params = [item[1] for item in op_params]
    # breakpoint()
    return np.array([waist2(λ, w0, z0, 1, matmul(prop(xVal-d),loopMat(op_list, args))) for xVal in xData(start, stop)])


# scipy optimize doesn't let you change parameters of a function, 
# so to determine optimal setups for various values of d1
# (length from fiber to collimating lens), I am creating a list of functions
# that have different values for d1
func_list = []
for i in range(20):
    d = 0.00887 + i*(3E-5)/20
    # Fit function
    def fit_func(x, d2, d3, d1=d, λ=1550E-9, w0=5.2E-6, z0=0.0, f1=9.6E-3, roc1=2, roc2=2):
        op_list = [Op(prop, d1, 1, color='Blue', static=False), 
                Op(interface, 1, asphereIndex, inf), 
                Op(prop, asphereTc, asphereIndex, color='Red'), 
                Op(interface, asphereIndex, 1, asphereROC),
                Op(prop, d2, 1, color='Blue'),
                Op(s_mirror, 2),
                Op(prop, d3, 1, color='Blue'),
                Op(s_mirror, 2)]
        args = [operation.args for operation in op_list]
        return waist2(λ, w0, z0, 1, matmul(prop(x-(d1+d2+d3-loop_len)),loopMat(op_list, args)))
    func_list.append([fit_func, d])


# for each function (i.e., each value of d1), find the optimal
# lengnths for the rest of the system to be what I want
res = []
for function in func_list:
    #Plot experimental data points
    func, d = function
    ops_to_input = [Op(prop, d, 1, color='Blue', static=False), 
                    Op(interface, 1, asphereIndex, inf), 
                    Op(prop, asphereTc, asphereIndex, color='Red'), 
                    Op(interface, asphereIndex, 1, asphereROC)]
    # Initial guess for the parameters
    initialGuess = [0.008885, 1.7435]    
    #Perform the curve-fit
    popt, pcov = curve_fit(func, xData(start, stop), yData(ops_to_input, start, stop), initialGuess)
    print([d, *popt])
    res.append([d, *popt])
    #x values for the fitted function
    xFit = np.arange(0.0, 1.0, 0.01)


# plot each optimized propagation to view differences
for i, item in enumerate(res):
    d1, d2, d3 = item
    num = len(res)
    col =(0.0+(1/num)*i, 0.0, 1.0-(1/num)*i)
    updated_op_list = [Op(prop, d1, 1, color=col), 
                        Op(interface, 1, asphereIndex, inf), 
                        Op(prop, asphereTc, asphereIndex, color=col), 
                        Op(interface, asphereIndex, 1, asphereROC),
                        Op(prop, d2, 1, color=col),
                        Op(s_mirror, 2),
                        Op(prop, d3, 1, color=col),
                        Op(s_mirror, 2),
                        Op(prop, 1, 1, color=col)]
    make_plots(updated_op_list, w0, z0, start_buffer=0, stop_buffer=0)

plt.show()