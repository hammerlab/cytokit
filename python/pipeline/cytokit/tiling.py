"""Tiling scheme implementations"""

class Tiling(object):

    def coordinates_from_index(self, index, w, h):
        """Get tile coordinates from index

        Args:
            index: 0-based tile index
            w: width of grid
            h: height of grid
        Returns:
            x, y - 0-based grid coordinates
        """
        if w <= 0:
            raise ValueError('Width must be >= 0')
        if h <= 0:
            raise ValueError('Height must be >= 0')
        return self._coordinates_from_index(index, w, h)

    def index_from_coordinates(self, x, y, w, h):
        """Get tile index from coordinates

        Args:
            x: 0-based X grid coordinate
            y: 0-based Y grid coordinate
            w: width of grid
            h: height of grid
        Returns:
            x, y - 0-based grid coordinates
        """
        if w <= 0:
            raise ValueError('Width must be >= 0')
        if h <= 0:
            raise ValueError('Height must be >= 0')
        return self._index_from_coordinates(x, y, w, h)

    def _coordinates_from_index(self, index, w, h):
        raise NotImplementedError()

    def _index_from_coordinates(self, x, y, w, h):
        raise NotImplementedError()

    def get_projection_map(self, src_dims, tgt_dims, origin):
        """Get 0-based projection map from a larger tiling to a smaller one

        Args:
            src_dims: Larger, original grid dimensions as 2-item tuple or list
            tget_dims: Smaller, target grid dimensions as 2-item tuple or list
            origin: 0-based Upper-Left location of smaller grid within larger one as
                2-item tuple or list
        Returns:
            Array where index in array corresponds to smaller grid index and value to index in larger grid (all 0-based)
        """
        import numpy as np
        n = tgt_dims[0] * tgt_dims[1]
        remap = []
        for i in range(n):
            # Get coordinates of index on target grid
            tgt_p = self.coordinates_from_index(i, w=tgt_dims[0], h=tgt_dims[1])
            
            # Add target point to point on source (i.e. larger) grid
            src_p = (origin[0] + tgt_p[0], origin[1] + tgt_p[1])
            
            # Map target grid index to index on original grid
            src_i = self.index_from_coordinates(x=src_p[0], y=src_p[1], w=src_dims[0], h=src_dims[1])
            remap.append(src_i)
        return np.array(remap)


class SnakeTiling(Tiling):
    """Snake tiling implies movements as left-to-right, down and then right-to-left, down, repeat"""

    def _coordinates_from_index(self, index, w, h):
        y = index // w
        x = index % w
        if y % 2 == 1:
            x = w - x - 1
        return x, y

    def _index_from_coordinates(self, x, y, w, h):
        i = y * w
        if y % 2 == 1:
            i += w - x - 1
        else:
            i += x
        return i


TILINGS = {
    'snake': SnakeTiling()
}

def get_tiling_by_name(name):
    if name not in TILINGS:
        raise ValueError('No tiling available for name "{}"'.format(name))
    return TILINGS[name]