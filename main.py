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
    elements = np.array([[tag2idx[int(t)] for t in tri] for tri in tri_tags], dtype=np.int32)

    # get dirichlet boundary conditions from physical groups
    fixed_nodes = {}
    physical_groups = gmsh.model.getPhysicalGroups(dim=1)
    
    for dim, group_tag in physical_groups:
        name = gmsh.model.getPhysicalName(dim, group_tag)
        if name.startswith("VOLTAGE_"):
            voltage_val = float(name.split("_")[1])
            curve_tags = gmsh.model.getEntitiesForPhysicalGroup(1, group_tag)
            for curve_tag in curve_tags:
                nTags, _, _ = gmsh.model.mesh.getNodes(1, curve_tag)
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
        self.resize(1000, 800)

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

        right_panel_layout = QVBoxLayout()
        
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect("equal")
        self.ax.set_title("electric potential and field lines")

        right_panel_layout.addWidget(self.toolbar)
        right_panel_layout.addWidget(self.canvas)

        layout.addLayout(control_layout, 1)
        layout.addLayout(right_panel_layout, 4)

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
            self.status_label.setText("status: plotting results...")
            QApplication.processEvents()

            self.figure.clear() 
            self.ax = self.figure.add_subplot(111)
            self.ax.set_aspect("equal")

            contour = self.ax.tricontourf(
                coords[:, 0], coords[:, 1], elements, V, levels=30, cmap="jet"
            )

            self.cbar = self.figure.colorbar(contour, ax=self.ax)
            self.cbar.set_label("electric potential (v)")

            self.ax.triplot(
                coords[:, 0], coords[:, 1], elements, 
                color='black', linewidth=0.5, alpha=0.4
            )

            x_lin = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 200)
            y_lin = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 200)
            X, Y = np.meshgrid(x_lin, y_lin)

            Ex_grid = griddata((cx, cy), Ex_centroids, (X, Y), method='linear')
            Ey_grid = griddata((cx, cy), Ey_centroids, (X, Y), method='linear')

            self.ax.streamplot(
                X, Y, Ex_grid, Ey_grid, 
                color="white", 
                density=1.2, 
                linewidth=1, 
                arrowsize=1.2
            )

            self.ax.set_title(f"equipotentials, mesh (size: {gui_mesh_size:.2f}), and field lines")
            self.canvas.draw()
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = fem_simulator()
    window.show()
    sys.exit(app.exec())