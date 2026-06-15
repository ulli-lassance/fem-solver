from easy_builder import EasyGeom

def generate_geometry(mesh_size):
    builder = EasyGeom("transmission_line")
    
    builder.set_air_domain(0, 0, 0.4, 0.1)

    builder.add_circle(0.1,0.05,0.025, 1000)

    builder.add_circle(0.3,0.05,0.025, -1000)
    
    builder.build_and_mesh(mesh_size)