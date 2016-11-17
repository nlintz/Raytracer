from material import Finish

VerySoftDull = Finish(ambient=0.35, diffuse=0.3, specular=0.8, roughness=1./20., reflection=0.1)
SoftDull = Finish(ambient=0.3, diffuse=0.4, specular=0.7, roughness=1./60., reflection=0.25)
Medium = Finish(ambient=0.25, diffuse=0.5, specular=0.8, roughness=1./80., reflection=0.5)
HardPolished = Finish(ambient=0.15, diffuse=0.6, specular=0.8, roughness=1./100., reflection=0.65)
VeryHardPolished = Finish(ambient=0.1, diffuse=0.7, specular=0.8, roughness=1./120., reflection=0.8)
