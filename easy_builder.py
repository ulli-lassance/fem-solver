import gmsh
from collections import defaultdict

class EasyGeom:
    def __init__(self, name="model"):
        gmsh.model.add(name)
        self.shapes = []
        self.domain_tag = None
        self.domain_boundary_voltage = None 

    def set_air_domain(self, x, y, width, height, boundary_voltage=0.0):
        self.domain_tag = gmsh.model.occ.addRectangle(x, y, 0, width, height)
        self.domain_boundary_voltage = boundary_voltage

    def set_circular_domain(self, x, y, radius, boundary_voltage=0.0):
        self.domain_tag = gmsh.model.occ.addDisk(x, y, 0, radius, radius)
        self.domain_boundary_voltage = boundary_voltage

    def add_circle(self, x, y, radius, voltage=100.0):
        tag = gmsh.model.occ.addDisk(x, y, 0, radius, radius)
        bbox = (x-radius, y-radius, x+radius, y+radius)
        self.shapes.append((tag, voltage, bbox))

    def add_rectangle(self, x, y, width, height, voltage=100.0):
        tag = gmsh.model.occ.addRectangle(x, y, 0, width, height)
        bbox = (x, y, x+width, y+height)
        self.shapes.append((tag, voltage, bbox))

    def add_triangle(self, x1, y1, x2, y2, x3, y3, voltage=100.0):
        p1 = gmsh.model.occ.addPoint(x1, y1, 0)
        p2 = gmsh.model.occ.addPoint(x2, y2, 0)
        p3 = gmsh.model.occ.addPoint(x3, y3, 0)
        
        l1 = gmsh.model.occ.addLine(p1, p2)
        l2 = gmsh.model.occ.addLine(p2, p3)
        l3 = gmsh.model.occ.addLine(p3, p1)
        
        cl = gmsh.model.occ.addCurveLoop([l1, l2, l3])
        tag = gmsh.model.occ.addPlaneSurface([cl])
        
        min_x, max_x = min(x1, x2, x3), max(x1, x2, x3)
        min_y, max_y = min(y1, y2, y3), max(y1, y2, y3)
        bbox = (min_x, min_y, max_x, max_y)
        
        self.shapes.append((tag, voltage, bbox))

    def build_and_mesh(self, mesh_size=0.15):
        if not self.domain_tag:
            raise ValueError("You must set an air domain first.")

        shape_tags = [(2, s[0]) for s in self.shapes]
        domain_dimtag = [(2, self.domain_tag)]
        
        if shape_tags:
            out, out_map = gmsh.model.occ.fragment(domain_dimtag, shape_tags)
        else:
            out_map = [[(2, self.domain_tag)]]
            
        gmsh.model.occ.synchronize()

        voltage_groups_2d = defaultdict(list)
        
        domain_fragments = [tag for dim, tag in out_map[0]]
        
        if shape_tags:
            for i, shape_info in enumerate(self.shapes):
                voltage = shape_info[1]
                shape_fragments = [tag for dim, tag in out_map[i + 1]]
                voltage_groups_2d[voltage].extend(shape_fragments)

        all_surfaces = gmsh.model.getEntities(2)
        curve_counts = defaultdict(int)
        
        for dim, tag in all_surfaces:
            boundaries = gmsh.model.getBoundary([(dim, tag)], oriented=False)
            for b_dim, b_tag in boundaries:
                curve_counts[abs(b_tag)] += 1
                
        outer_boundary_curves = [c_tag for c_tag, count in curve_counts.items() if count == 1]

        group_id = 1
        
        if self.domain_boundary_voltage is not None and outer_boundary_curves:
            gmsh.model.addPhysicalGroup(1, outer_boundary_curves, group_id)
            gmsh.model.setPhysicalName(1, group_id, f"VOLTAGE_{self.domain_boundary_voltage}")
            group_id += 1

        for voltage, surfs in voltage_groups_2d.items():
            if surfs:
                gmsh.model.addPhysicalGroup(2, list(set(surfs)), group_id)
                gmsh.model.setPhysicalName(2, group_id, f"VOLTAGE_{voltage}")
                group_id += 1
                
        if domain_fragments:
            air_only_fragments = [t for t in domain_fragments if not any(t in v_surfs for v_surfs in voltage_groups_2d.values())]
            if air_only_fragments:
                gmsh.model.addPhysicalGroup(2, air_only_fragments, group_id)
                gmsh.model.setPhysicalName(2, group_id, "DOMAIN")

        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.model.mesh.generate(2)