import sys
import importlib.util
import inspect
import numpy as np
from scipy.sparse import lil_matrix, linalg, coo_matrix
from scipy.interpolate import griddata
import gmsh

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QLabel,
    QDoubleSpinBox,
    QTabWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


def extract_mesh_data(geom_module, mesh_size):
    """
    handles all gmsh operations. generates the geometry and extracts 
    node coordinates, element connectivity, and boundary conditions.
    """
    if gmsh.isInitialized():
        gmsh.clear()
        gmsh.finalize()
    gmsh.initialize()

    sig = inspect.signature(geom_module.generate_geometry)
    if len(sig.parameters) > 0:
        geom_module.generate_geometry(mesh_size)
    else:
        geom_module.generate_geometry()

    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
    if len(nodeTags) == 0:
        raise ValueError("no mesh was generated. check your geometry file.")

    coords = nodeCoords.reshape(-1, 3)[:, :2]
    tag2idx = {int(tag): i for i, tag in enumerate(nodeTags)}

    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(2)
    if len(elemTags) == 0:
        raise ValueError("no triangle elements found. check your mesh settings.")
        
    tri_tags = elemNodeTags.reshape(-1, 3)

    expanded_elements = []

    for tri in tri_tags:
        inner_list = []
        for t in tri:
            mapped_value = tag2idx[int(t)]
            inner_list.append(mapped_value)
        expanded_elements.append(inner_list)
    elements = np.array(expanded_elements, dtype=np.int32)

    fixed_nodes = {}

    for target_dim in [1, 2]:
        physical_groups = gmsh.model.getPhysicalGroups(dim=target_dim)
        
        for dim, group_tag in physical_groups:
            name = gmsh.model.getPhysicalName(dim, group_tag)
            if name.startswith("VOLTAGE_"):
                voltage_val = float(name.split("_")[1])
                
                entities = gmsh.model.getEntitiesForPhysicalGroup(dim, group_tag)
                for entity_tag in entities:
                    nTags, _, _ = gmsh.model.mesh.getNodes(dim, entity_tag, includeBoundary=True)
                    for nt in nTags:
                        fixed_nodes[tag2idx[int(nt)]] = voltage_val

    if not fixed_nodes:
        raise ValueError("no voltage boundary conditions found.")

    gmsh.clear()
    gmsh.finalize()

    return coords, elements, fixed_nodes

def solve_fem(coords, elements, fixed_nodes):
    from scipy.linalg import solve_banded
    
    num_nodes = len(coords)

    # ensure counter-clockwise order for positive triangle area
    i, j, k = elements[:, 0], elements[:, 1], elements[:, 2]
    xi, yi = coords[i, 0], coords[i, 1]
    xj, yj = coords[j, 0], coords[j, 1]
    xk, yk = coords[k, 0], coords[k, 1]

    A_signed = 0.5 * ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
    needs_swap = A_signed < 0
    elements[needs_swap, 1], elements[needs_swap, 2] = elements[needs_swap, 2].copy(), elements[needs_swap, 1].copy()

    # re-extract coordinates after potential node swap
    i, j, k = elements[:, 0], elements[:, 1], elements[:, 2]
    xi, yi = coords[i, 0], coords[i, 1]
    xj, yj = coords[j, 0], coords[j, 1]
    xk, yk = coords[k, 0], coords[k, 1]

    # compute spatial derivatives of shape functions and element area
    P0, P1, P2 = yj - yk, yk - yi, yi - yj
    Q0, Q1, Q2 = xk - xj, xi - xk, xj - xi
    A = 0.5 * (P1 * Q2 - P2 * Q1)

    P = np.column_stack((P0, P1, P2))
    Q = np.column_stack((Q0, Q1, Q2))

    # setup sparse banded matrix to save memory
    
    # find maximum node index difference for half-bandwidth
    nb = int(np.max([np.abs(elements[:, r] - elements[:, s]).max() for r in range(3) for s in range(3)]))

    # initialize banded matrix and load vector
    ab = np.zeros((2 * nb + 1, num_nodes), dtype=np.float64)
    B = np.zeros(num_nodes, dtype=np.float64)

    # compute local stiffness coefficients and add to global banded matrix
    for r in range(3):
        for s in range(3):
            C_rs = (P[:, r] * P[:, s] + Q[:, r] * Q[:, s]) / (4.0 * A)
            row_indices = elements[:, r]
            col_indices = elements[:, s]
            
            # map standard row and col to banded structure index
            band_rows = nb + row_indices - col_indices
            np.add.at(ab, (band_rows, col_indices), C_rs)

    # apply dirichlet boundary conditions to banded matrix
    for idx, val in fixed_nodes.items():
        # find valid column range within the band
        j_min = max(0, idx - nb)
        j_max = min(num_nodes - 1, idx + nb)
        j_cols = np.arange(j_min, j_max + 1)
        
        # zero out row entries in the band
        ab[nb + idx - j_cols, j_cols] = 0.0
        
        # set main diagonal to 1 and assign target voltage
        ab[nb, idx] = 1.0
        B[idx] = val

    # solve system using gaussian elimination for banded matrices
    V = solve_banded((nb, nb), ab, B)
    V = np.asarray(V).flatten()

    # calculate electric field as negative gradient of potential inside elements
    Ex_centroids = -(V[i] * P0 + V[j] * P1 + V[k] * P2) / (2.0 * A)
    Ey_centroids = -(V[i] * Q0 + V[j] * Q1 + V[k] * Q2) / (2.0 * A)

    # compute element centroids
    cx = (xi + xj + xk) / 3.0
    cy = (yi + yj + yk) / 3.0

    return V, Ex_centroids, Ey_centroids, cx, cy, elements


class fem_simulator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("simple fem solver")
        self.resize(1200, 800)

        self.current_file_path = None

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        control_layout = QVBoxLayout()
        
        self.mesh_label = QLabel("max mesh size:")
        self.mesh_input = QDoubleSpinBox()
        self.mesh_input.setRange(0.01, 2.0)
        self.mesh_input.setSingleStep(0.05)
        self.mesh_input.setValue(0.15)

        self.load_btn = QPushButton("load new geometry")
        self.load_btn.clicked.connect(self.load_file)

        self.reload_btn = QPushButton("reload")
        self.reload_btn.clicked.connect(self.reload_file)
        self.reload_btn.setEnabled(False)

        self.status_label = QLabel("status: waiting for input...")

        control_layout.addWidget(self.mesh_label)
        control_layout.addWidget(self.mesh_input)
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.reload_btn)
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()

        self.tabs = QTabWidget()
        
        self.tab_2d = QWidget()
        layout_2d = QVBoxLayout(self.tab_2d)
        self.fig_2d = Figure()
        self.canvas_2d = FigureCanvas(self.fig_2d)
        self.toolbar_2d = NavigationToolbar(self.canvas_2d, self.tab_2d)
        layout_2d.addWidget(self.toolbar_2d)
        layout_2d.addWidget(self.canvas_2d)
        self.tabs.addTab(self.tab_2d, "2d cross-section")

        self.tab_3d = QWidget()
        layout_3d = QVBoxLayout(self.tab_3d)
        self.fig_3d = Figure()
        self.canvas_3d = FigureCanvas(self.fig_3d)
        self.toolbar_3d = NavigationToolbar(self.canvas_3d, self.tab_3d)
        layout_3d.addWidget(self.toolbar_3d)
        layout_3d.addWidget(self.canvas_3d)
        self.tabs.addTab(self.tab_3d, "3d voltage surface")

        layout.addLayout(control_layout, 1)
        layout.addWidget(self.tabs, 4)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "select geometry script", "", "python files (*.py)"
        )
        if not file_path:
            return

        self.current_file_path = file_path
        self.reload_btn.setEnabled(True)
        self.run_simulation()

    def reload_file(self):
        if self.current_file_path:
            self.run_simulation()
        else:
            QMessageBox.warning(self, "warning", "please load a geometry file first.")

    def run_simulation(self):
        try:
            # load the python geometry module
            spec = importlib.util.spec_from_file_location("geom_module", self.current_file_path)
            geom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(geom_module)

            if not hasattr(geom_module, "generate_geometry"):
                raise ValueError("the selected file does not contain a 'generate_geometry()' function.")

            gui_mesh_size = self.mesh_input.value()
            
            # extract mesh data
            self.status_label.setText("status: meshing and extracting data...")
            QApplication.processEvents()
            
            coords, elements, fixed_nodes = extract_mesh_data(geom_module, gui_mesh_size)

            # assemble and solve the system
            self.status_label.setText("status: assembling and solving system...")
            QApplication.processEvents()
            
            V, Ex_centroids, Ey_centroids, cx, cy, elements = solve_fem(coords, elements, fixed_nodes)
            
            # plot the results
            self.status_label.setText("status: plotting 2d and 3d results...")
            QApplication.processEvents()

            # call the separated plotting functions
            self.plot_2d_results(coords, elements, V, Ex_centroids, Ey_centroids, cx, cy, gui_mesh_size)
            self.plot_3d_results(coords, elements, V)

            self.status_label.setText("status: done.")

        except Exception as e:
            # clear gmsh on failure
            if gmsh.isInitialized():
                try:
                    gmsh.clear()
                    gmsh.finalize()
                except:
                    pass
            
            QMessageBox.critical(self, "error", f"failed to load or solve:\n\n{str(e)}")
            self.status_label.setText("status: error occurred. waiting for valid input...")

    def plot_2d_results(self, coords, elements, V, Ex_centroids, Ey_centroids, cx, cy, mesh_size):
        """
        handles drawing the 2d equipotential and field streamplot.
        """
        self.fig_2d.clear() 
        ax = self.fig_2d.add_subplot(111)
        ax.set_aspect("equal")


        surface_2d = ax.tripcolor(
            coords[:, 0], coords[:, 1], elements, V, 
            cmap="jet", shading='gouraud'
        )

        cbar = self.fig_2d.colorbar(surface_2d, ax=ax, shrink=0.7, pad=0.1)
        cbar.set_label("electric potential (v)")

        ax.triplot(
            coords[:, 0], coords[:, 1], elements, 
            color='black', linewidth=0.5, alpha=0.4
        )

        x_lin = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 200)
        y_lin = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 200)
        X, Y = np.meshgrid(x_lin, y_lin)

        Ex_grid = griddata((cx, cy), Ex_centroids, (X, Y), method='linear')
        Ey_grid = griddata((cx, cy), Ey_centroids, (X, Y), method='linear')

        E_mag = np.sqrt(Ex_grid**2 + Ey_grid**2)
        threshold = np.nanmax(E_mag) * 1e-4
        
        Ex_grid[E_mag < threshold] = np.nan
        Ey_grid[E_mag < threshold] = np.nan
        
        ax.streamplot(
            X, Y, Ex_grid, Ey_grid, 
            color="white", 
            density=1.2, 
            linewidth=1, 
            arrowsize=1.2
        )

        ax.set_title(f"equipotentials, mesh (size: {mesh_size:.2f}), and field lines")
        self.canvas_2d.draw()

    def plot_3d_results(self, coords, elements, V):
        """
        handles drawing the 3d voltage surface.
        """
        self.fig_3d.clear()
        ax = self.fig_3d.add_subplot(111, projection='3d')

        surf = ax.plot_trisurf(
            coords[:, 0], coords[:, 1], V, 
            triangles=elements, cmap='jet', linewidth=0.2, edgecolor='black', alpha=0.9
        )

        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        dx = x_max - x_min
        dy = y_max - y_min

        dz = max(dx, dy) * 0.4 
        
        ax.set_box_aspect((dx, dy, dz))

        cbar = self.fig_3d.colorbar(surf, ax=ax, shrink=0.7, pad=0.1)
        cbar.set_label("electric potential (v)")

        ax.set_title("3d surface plot of electric potential")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("voltage (v)")
        
        self.canvas_3d.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = fem_simulator()
    window.show()
    sys.exit(app.exec())