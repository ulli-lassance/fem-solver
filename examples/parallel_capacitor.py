from easy_builder import EasyGeom

def generate_geometry(mesh_size):
    builder = EasyGeom("capacitor")
    
    builder.set_air_domain(-2, -2, 4, 4, boundary_voltage=0.0)
    builder.add_rectangle(-1, 0.5, 2, 0.2, voltage=30)
    builder.add_rectangle(-1, -0.7, 2, 0.2, voltage=-30)
    
    builder.build_and_mesh(mesh_size)