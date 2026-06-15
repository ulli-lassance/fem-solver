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
            out, _ = gmsh.model.occ.cut(domain_dimtag, shape_tags)
            final_domain_tags = [t[1] for t in out]
        else:
            final_domain_tags = [self.domain_tag]
            
        gmsh.model.occ.synchronize()

        final_boundaries = gmsh.model.getBoundary([(2, t) for t in final_domain_tags], oriented=False)
        voltage_groups = defaultdict(list)

        for c_dim, c_tag in final_boundaries:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(c_dim, c_tag)
            assigned = False
            
            for tag, voltage, bbox in self.shapes:
                tol = 1e-3  

                if (bbox[0] - tol <= xmin and xmax <= bbox[2] + tol) and \
                   (bbox[1] - tol <= ymin and ymax <= bbox[3] + tol):
                    voltage_groups[voltage].append(c_tag)
                    assigned = True
                    break 

            if not assigned and self.domain_boundary_voltage is not None:
                voltage_groups[self.domain_boundary_voltage].append(c_tag)

        group_id = 1
        for voltage, curves in voltage_groups.items():
            if curves:
                gmsh.model.addPhysicalGroup(1, list(set(curves)), group_id)
                gmsh.model.setPhysicalName(1, group_id, f"VOLTAGE_{voltage}")
                group_id += 1
                
        gmsh.model.addPhysicalGroup(2, final_domain_tags, group_id)
        gmsh.model.setPhysicalName(2, group_id, "DOMAIN")

        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.model.mesh.generate(2)