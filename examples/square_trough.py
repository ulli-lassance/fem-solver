from easy_builder import EasyGeom

def generate_geometry(mesh_size):
    builder = EasyGeom("square_trough")
    
    builder.set_air_domain(-2, -2, 4, 4,boundary_voltage=0)
    builder.add_circle(0, 0, radius=0.3, voltage=100)
    
    builder.build_and_mesh(mesh_size)