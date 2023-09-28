from raytracing import *

# define input beam
beam1 = GaussianBeam(w=5.2E-3,wavelength=1550E-6,z=0)
# beam2 = GaussianBeam(w=4.2E-3,wavelength=1550E-6,z=0)


# define optics for 10-ns loop
path = LaserPath()
path.append(Space(d=9.82556))
path.append(Lens(f=9.6, diameter = 12.7))
path.append(Space(d=915))
path.append(CurvedMirror(R=-1000, diameter = 12.7))
path.append(Space(d=1960))
path.append(CurvedMirror(R=-1000, diameter = 12.7))
path.append(Space(d=500))
path.display(beams=[beam1])

rayPath = ImagingPath()
rayPath.append(Space(d=9.82556))
rayPath.append(Lens(f=9.6, diameter = 12.7))
rayPath.append(Space(d=915))
rayPath.append(CurvedMirror(R=-1000, diameter = 12.7))
rayPath.append(Space(d=1960))
rayPath.append(CurvedMirror(R=-1000, diameter = 12.7))
rayPath.append(Space(d=500))
rayPath.display()