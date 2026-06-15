from easy_builder import EasyGeom

def generate_geometry(mesh_size=0.15):
    geom = EasyGeom("coaxial_cable")

    cx, cy = 0.0, 0.0
    
    geom.set_circular_domain(cx, cy, radius=5.0, boundary_voltage=0.0)
    geom.add_circle(cx, cy, radius=1.0, voltage=100.0)
    geom.build_and_mesh(mesh_size)