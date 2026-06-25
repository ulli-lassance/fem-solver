from easy_builder import EasyGeom

def generate_geometry(mesh_size):
    builder = EasyGeom("transmission_line")

    # define the outer box
    builder.set_air_domain(0, 0, 0.4, 0.1, boundary_voltage=0.0)

    # left conductor (1000V)
    builder.add_circle(0.1,0.05,0.025, voltage=1000)

    # right conductor (-1000V)
    builder.add_circle(0.3,0.05,0.025, voltage=-1000)
    
    # generates the finite element mesh
    builder.build_and_mesh(mesh_size)