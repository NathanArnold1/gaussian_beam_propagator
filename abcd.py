import numpy as np
from numpy import sqrt, pi, matmul

# All of these are directly from the wikipedia pages for gaussian beams and for ABCD matrices, just defining them as functions
def waist_eq(wavelength, waist, z, z0):
    return waist*sqrt(1+(((z-z0)*wavelength)/(pi*(waist**2)))**2)

def inverseR(wavelength, waist, z0):
    return z0/(z0**2 + (pi*waist**2/wavelength)**2)

def q(wavelength, waist, z0):
    return 1/((-1j*wavelength)/(pi*(waist_eq(wavelength, waist, 0, z0)**2)) + inverseR(wavelength, waist, z0))

def q2(wavelength, waist, z0, matrix):
    return( (matrix[0,0]*q(wavelength, waist, z0) + matrix[0,1])/(matrix[1,0]*q(wavelength, waist, z0) + matrix[1,1]) )

def waist2(wavelength, waist, z0, matrix):
    return sqrt((-wavelength)/(pi*((1/q2(wavelength, waist, z0, matrix)).imag)))

def thin_lens(f):
    return np.array([[1,0],[-1/f,1]])

def thick_lens(n1, n2, r1, r2, t):
    """
    n1: index of refraction outside of the lens (~1)
    n2: index of refraction inside of the lens
    r1: Radius of curvature of first surface
    r1: Radius of curvature of second surface
    """
    return np.array([[1,0],[1/((n2-n1)/(r2*n1)),n2/n1]])*np.array([[1,t],[0,1]])*np.array([[1,0],[1/((n1-n2)/(r1*n2)),n1/n2]])

def s_mirror(R):
    return np.array([[1,0],[-2/R,1]], dtype=object)

def prop(d):
    return np.array([[1,d],[0,1]], dtype=object)

def loopMat(ops, args):
    """
    Multiply all the matrices together
    """
    assert len(ops) == len(args)
    matrix = np.array([[1,0],[0,1]])
    for op, arg in zip(ops,args):
        matrix = matmul(op(arg), matrix)
    return matrix