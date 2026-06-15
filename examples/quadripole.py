from easy_builder import EasyGeom

def generate_geometry(mesh_size=0.15):
    builder = EasyGeom("quadrupole")

    builder.set_air_domain(-2, -2, 4, 4,boundary_voltage=0.0)
    
    builder.add_circle(0, 1.5, 0.1, voltage=50.0)
    builder.add_circle(0, -1.5, 0.1, voltage=50.0)
    
    builder.add_circle(-1.5, 0, 0.1, voltage=50.0)
    builder.add_circle(1.5, 0, 0.1, voltage=-20.0)
    
    builder.build_and_mesh(mesh_size=mesh_size)