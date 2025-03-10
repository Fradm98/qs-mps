import numpy as np

class FrancescosDualCylinder:
    def __init__(self, px, py):
        self.px = px
        self.py = py
        self.nodes = []
        self.edges = [] # Does not include edges to the extra degree of freedom
        for xi in range(px):
            for yi in range(py):
                self.nodes.append((xi, yi))
                # Horizontal dual edges
                if xi < px - 1:
                    self.edges.append((xi+0.5, yi))
                # Vertical dual edges
                if yi < py - 1 or py > 2:
                    self.edges.append((xi, yi+0.5))
    
        self.edges = sorted(self.edges)
        self.coords = sorted(self.nodes + self.edges)

    def is_vertical_edge(self, edge):
        if isinstance(edge, int):
            edge = self.links[edge]
        elif not isinstance(edge, tuple):
            raise ValueError("edge must be an int or a tuple.")
        if len(edge) != 2:
            raise ValueError("edge must be a tuple of length 2.")
        return type(edge[0]) == int

    def nodes_connected_to_edge(self, edge):
        if isinstance(edge, int):
            edge = self.edges[edge]
        elif not isinstance(edge, tuple):
            raise ValueError("edge must be an int or a tuple.")
        if len(edge) != 2:
            raise ValueError("edge must be a tuple of length 2.")
        
        if self.is_vertical_edge(edge):
            if edge[1]-0.5 < self.py-1:
                return [(edge[0], int(edge[1]-0.5)), (edge[0], int(edge[1]+0.5))]
            else:
                return [(edge[0], int(edge[1]-0.5)), (edge[0], 0)]
        else:
            return [(int(edge[0]-0.5), edge[1]), (int(edge[0]+0.5), edge[1])]
        
    def is_edge_coord(self, coords):
        if not isinstance(coords, tuple):
            raise ValueError("coords must be an int or a tuple.")
        if len(coords) != 2:
            raise ValueError("coords must be a tuple of length 2.")
        return not all(coords.is_integer() for coords in coords)
    
    def coords_to_index(self, coords):
        coords = np.array(coords)
        if len(coords.shape) == 1:
            if coords.shape[0] != 2:
                raise ValueError("not valid coords")
            coords = coords[None]
        elif len(coords.shape) == 2:
            if coords.shape[1] != 2:
                raise ValueError("not valid coords")
        else:
            raise ValueError("not valid coords")
        
        indices = np.zeros(ncoords := coords.shape[0], dtype=int)
        for i, coord in enumerate(coords):
            if self.is_edge_coord(tuple(coord)):
                raise ValueError("only node coords have an index assigned")
            else:
                indices[i] = self.nodes.index(tuple(coord))
        return indices if ncoords > 1 else int(indices[0])
    
    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def num_edges(self):
        return len(self.edges)

    def __len__(self):
        return len(self.coords)

    def __repr__(self):
        return f"FrancescosCylinder(px={self.px}, py={self.py})"