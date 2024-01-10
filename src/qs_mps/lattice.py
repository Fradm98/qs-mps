"""This module implements the Lattice base class for the LGT children classes"""

import numpy as np
from itertools import chain, product
from typing import Tuple, List, Dict

# Type hintings
Site = Tuple[int, int]
Plaquette = Tuple[int, int, int, int]
Star = Tuple[int, int, int, int]


#
def cyclic_perm(lst, n=1):
    return lst[n:] + lst[:n]

def zip_nearest(lst, periodic=True, step=1):
    if periodic:
        return zip(lst, cyclic_perm(lst, n=step))
    else:
        return zip(lst[:-step], lst[step:])


class LatticeError(RuntimeError):
    pass


class Lattice(object):
    def __init__(self, size: Tuple[int, int], pbc: Tuple[bool, bool]):
        """
        Create an instance of the Lattice class for the given size and
        periodic boundary conditions.
        This class is intended to be used as a base class for all the LGT models.

        Parameters
        ----------
        size : tuple(int, int)
            Number of sites along x and y respectively.
        pbc : tuple(bool, bool)
            Periodic boundary conditions along x and y respectively.
        """
        self.size = size
        self.pbc = pbc
        self._label_links()
        self._lattice_drawer = LatticeDrawer(self)

    @property
    def Lx(self) -> int:
        """Returns the horizontal length"""
        return self.size[0]

    @property
    def Ly(self) -> int:
        """Returns the vertical length"""
        return self.size[1]

    @property
    def pbc_x(self) -> bool:
        """Returns if x-axis is period"""
        return self.pbc[0]

    @property
    def pbc_y(self) -> bool:
        """Returns if y-axis is period"""
        return self.pbc[1]

    @property
    def nlinks_x(self) -> int:
        """Returns the number of links along the x-axis"""
        return self.Lx - 1 * (not self.pbc_x)

    @property
    def nlinks_y(self) -> int:
        """Returns the number of links along the y-axis"""
        return self.Ly - 1 * (not self.pbc_y)

    @property
    def area(self) -> int:
        """Returns the total area of the lattice"""
        return self.Lx * self.Ly

    @property
    def nlinks(self) -> int:
        """Returns the total number of links"""
        return self.Lx * self.nlinks_y + self.nlinks_x * self.Ly

    def __repr__(self):
        return f"<Lattice: size={self.size}, pbc={self.pbc}>"

    def __str__(self):
        ret = repr(self) + "\n\n"
        ret += self._lattice_drawer.draw_lattice()
        return ret

    def _label_links(self):
        """Labels the links of the lattice"""
        Lx = self.Lx
        Ly = self.Ly
        lx = self.nlinks_x
        ly = self.nlinks_y
        # initialize the "adjancy matrix"
        self._link_matrix = np.zeros((self.area, self.area))
        # assign for each link an integer in a sequential order
        # Starting from the horizontal ones and then the vertical ones
        # if there is no link than it's a zero
        if Lx == 1:  # first along x
            offset = 0  # nothing to do
        elif Lx == 2:
            y = np.arange(0, Ly)
            self._link_matrix[0 + y * Lx, 1 + y * Lx] = y + 1
            self._link_matrix[1 + y * Lx, 0 + y * Lx] = y + 1 + self.pbc_x * Ly
            offset = (1 + self.pbc_x) * Ly + 1
        else:
            x = np.arange(0, lx)
            for y in np.arange(0, Ly):
                self._link_matrix[x + y * Lx, ((x + 1) % Lx) + y * Lx] = x + y * lx + 1
                self._link_matrix[((x + 1) % Lx) + y * Lx, x + y * Lx] = x + y * lx + 1
                # ^ the link matrix have to be bidirectional
            offset = Ly * (Lx - 1 * (not self.pbc_x)) + 1

        if Ly == 2:  # then along y
            x = np.arange(0, Lx)
            self._link_matrix[x + 0 * Lx, x + 1 * Lx] = x + offset
            self._link_matrix[x + 1 * Lx, x + 0 * Lx] = (
                x + offset + Lx * int(self.pbc_y)
            )
        elif Ly > 2:
            y = np.arange(0, ly)
            for x in np.arange(0, Lx):
                self._link_matrix[x + y * Lx, x + ((y + 1) % Ly) * Lx] = (
                    x + y * Lx + offset
                )
                self._link_matrix[x + ((y + 1) % Ly) * Lx, x + y * Lx] = (
                    x + y * Lx + offset
                )

    def link(self, site0: Site, site1: Site, flip=False, from_zero=False) -> int:
        """Return the link index between two sites

        Parameters
        ----------
        site0 : tuple[int,int]
        site1 : tuple[int,int]
        flip : bool (default=False)
            whether to flip the integer repr of the link indices.
        from_zero : bool (default=False)
            whether to start the indexing from zero or not.
            May cause problems.

        the coordinates are taken mod Lx (for x_i) or mod Ly (for y_i)
        The value 0 (with from_zero=False) is used to indicate an absent link.

        Returns
        ----------
        The index of the link
        """
        x0, y0 = site0
        x1, y1 = site1
        index = (
            self._link_matrix[
                (x0 % self.Lx) + (y0 % self.Ly) * self.Lx,
                (x1 % self.Lx) + (y1 % self.Ly) * self.Lx,
            ]
            - 1 * from_zero
        )
        if flip:
            index = self.nlinks - index - 1 * from_zero
        return int(index)

    def plaquette(self, site: Site, **kwargs) -> Plaquette:
        """
        Return the links of a plaquette. The links are oriented
        counterclok-wise starting from the bottom site.

        Return
        ----------
        plaq : tuple[int, int, int, int]
            A 4-element list of the indices of the links of the plaquette.
        """
        x, y = site
        if x >= self.nlinks_x or y >= self.nlinks_y:
            raise ValueError(f"The coordinates {x} and {y} exceeds the lattice")

        plaq = (
            self.link((x, y), (x + 1, y), **kwargs),
            self.link((x + 1, y), (x + 1, y + 1), **kwargs),
            self.link((x + 1, y + 1), (x, y + 1), **kwargs),
            self.link((x, y + 1), (x, y), **kwargs),
        )
        return plaq

    def star(self, site: Site, L, l, **kwargs) -> Star:
        """
        Return the links of a star. The links are oriented counterclock-wise
        starting from the right link.

        Return
        ----------
        star : tuple[int, int, int, int]
            A 4-element array
        """
        x, y = site
        if x >= self.Lx or y >= self.Ly:
            raise ValueError(f"The coordinates {x} and {y} exceeds the lattice")

        star = []
        if x + 1 >= L:
            star.append(0)
        else:
            star.append(self.link((x, y), (x + 1, y), **kwargs))
        if y + 1 >= l:
            star.append(0)
        else:
            star.append(self.link((x, y), (x, y + 1), **kwargs))
        if x - 1 < 0:
            star.append(0)
        else:
            star.append(self.link((x - 1, y), (x, y), **kwargs))
        if y - 1 < 0:
            star.append(0)
        else:
            star.append(self.link((x, y - 1), (x, y), **kwargs))
        return star

    @property
    def sites(self) -> List[Site]:
        return list(product(range(self.Lx), range(self.Ly)))

    # @property
    def plaquettes(self, **kwargs) -> List[Plaquette]:
        """
        Return a list of all the plaquettes.
        The indices of the plaquettes start from 0 by default.
        """
        kwargs.setdefault("from_zero", False)
        plaquettes = [
            self.plaquette(site, **kwargs)
            for site in self.sites
            if site[0] < self.nlinks_x and site[1] < self.nlinks_y
        ]
        return plaquettes

    # @property
    def stars(self, **kwargs) -> List[Star]:
        """
        Return an array of all the stars.
        The indices of the stars start from 0.
        """
        kwargs.setdefault("from_zero", False)
        stars = [self.star(site, **kwargs) for site in self.sites]
        return stars

    def path(self, site0: Site, site1: Site, **kwargs) -> List[int]:
        """
        Return the indices of links that connect two sites that are on a straigth line

        Parameters
        ----------
        site0, site1 : tuple(int,int)
            the coordinates of the start and end of the path

        Return
        ----------
        path : np.array(int)
            A list of the links that connects site0 and site1.
            The indexing of the links starts from 0.

        Raise:
        ----------
        RuntimeError
                if the sites are not aligned
        """
        x0, y0 = site0
        x1, y1 = site1
        kwargs.setdefault("from_zero", False)
        step = lambda a, b: +1 if a <= b else -1
        if y0 == y1:
            sites = [(x, y0) for x in range(x0, x1 + step(x0, x1), step(x0, x1))]
        elif x0 == x1:
            sites = [(x0, y) for y in range(y0, y1 + step(y0, y1), step(y0, y1))]
        else:
            raise RuntimeError(
                f"The sites ({x0}, {y0}) and ({x1}, {y1}) are not aligned"
            )
        path_ = [
            self.link(prev, next, **kwargs)
            for prev, next in zip_nearest(sites, periodic=False)
        ]
        path_ = np.array(path_)
        return path_

    def dual_path(self, plq0: Site, plq1: Site, **kwargs) -> List[int]:
        """
        Given two plaquettes plq0 and plq1, it returns the links that are cut from
        a dual path that start from plq0 and ends in plq1.

        Parameters
        ----------
        plq0, plq1 : tuple(int,int)
            A plaquette is specified by the coordinates of the bottom-left site.

        Returns
        ----------
        dual_path : np.array(int)
            A list of indices of the links that are cut by the dual path.
            Indices start from 0.

        Raises
        ----------
        Exception
            If the plaquettes are not aligned along a straigth line
        """
        x0, y0 = plq0
        x1, y1 = plq1
        kwargs.setdefault("from_zero", False)
        # +1 for positive direction, -1 for negative directions:
        step = lambda a, b: +1 if a <= b else -1
        # get the correct starting and ending points
        # depending if we are going in the positive or negative direction
        begin = lambda a, b: a + 1 if a <= b else a
        end = lambda a, b: b + 1 if a <= b else a

        # is the path horizontal or vertical?
        if y0 == y1:
            pairs = [
                ((x, y0), (x, y0 + 1))
                for x in range(begin(x0, x1), end(x0, x1), step(x0, x1))
            ]
        elif x0 == x1:
            pairs = [
                ((x0, y), (x0 + 1, y))
                for y in range(begin(y0, y1), end(y0, y1), step(y0, y1))
            ]
        else:
            raise RuntimeError(
                f"The sites ({x0}, {y0}) and ({x1}, {y1}) are not aligned"
            )
        dual_path_ = [self.link(prev, next, **kwargs) for prev, next in pairs]
        dual_path_ = np.array(dual_path_)
        return dual_path_

    def loop(self, sites: List[Site], **kwargs):
        """
        Return a list of indices of a loop given a list of the site to visits.
        The sites are specified by tuple of two integers.

        Parameters
        ----------
        sites : list(tuple(int,int))

        Return
        ----------
        loop : np.array(int)
            A list of the indices of all the link visited
        """
        loop_indx = [
            self.path(prev, next, **kwargs)
            for prev, next in zip_nearest(sites, periodic=True)
        ]
        loop_indx = list(chain.from_iterable(loop_indx))
        return np.array(loop_indx)

    def string(self, plqs: List[Site], **kwargs):
        """
        Return a list of indices of a loop given a list of the site to visit.

        Parameters
        ----------
        plqs : list(tuple(int,int))

        Return
        ----------
        plaquettes : np.array(int)
            A list of the indices of all the link visited
        """
        string_indx = [
            self.dual_path(prev, next, **kwargs)
            for prev, next in zip_nearest(plqs, periodic=False)
        ]
        string_indx = list(chain.from_iterable(string_indx))
        return np.array(string_indx)

    def vlinks(self, y, **kwargs):
        """Returns a list of all the vertical links at a given y"""
        if y >= self.nlinks_y or y < 0:
            raise ValueError("Given y parameter exceeds the lattice dimensions")
        return [self.link((x, y), (x, y + 1), **kwargs) for x in range(self.Lx)]

    def hlinks(self, y, **kwargs):
        """Returns a list of all the horizontal links at a given y"""
        if y >= self.Ly or y < 0:
            raise ValueError("Given y parameter exceeds the lattice dimensions")
        bulk = [self.link((x, y), (x + 1, y), **kwargs) for x in range(self.Lx - 1)]
        if self.pbc_x:
            edges = [
                self.link((self.Lx - 1, y), (0, y), **kwargs),
                self.link((self.Lx - 1, y), (self.Lx, y), **kwargs),
            ]
        else:
            edges = []
        return (bulk, edges)


class LatticeDrawer(object):
    default_char_table = dict(hlink="--", vlink="|", site="+")

    def __init__(self, lattice, char_table: Dict = None):
        self.lattice = lattice
        self.Lx = self.lattice.Lx
        self.pbc_x = self.lattice.pbc_x
        self.char_table = (
            char_table if char_table is not None else self.default_char_table
        )

    def draw_row_links(self, y: int) -> str:
        """Draw the horizontal links of a given row"""
        hlink = self.char_table["hlink"]
        site = self.char_table["site"]
        blank = " " * len(hlink)
        bigblank = blank * 2 + "  "
        bulk, edges = self.lattice.hlinks(y, from_zero=False)
        left = blank + f"{edges[0]:>2d}" + hlink if self.pbc_x else bigblank
        right = hlink + f"{edges[1]:<2d}" + blank if self.pbc_x else bigblank
        left = left + site + hlink
        right = hlink + site + right
        inner = (hlink + site + hlink).join(f"{link:2d}" for link in bulk)
        return left + inner + right + "\n"

    def draw_vert_empty(self) -> str:
        """Draw empty vertical spaces"""
        blank = " " * (2 * len(self.char_table["hlink"]) + 2)
        vlink = self.char_table["vlink"]
        return vlink.join(blank for _ in range(self.Lx + 1)) + "\n"

    def draw_vert_links(self, y: int) -> str:
        """Draw the vertical links at a given y"""
        links = self.lattice.vlinks(y, from_zero=False)
        blank = " " * (2 * len(self.char_table["hlink"]) + 2)
        return blank + blank[1:].join(f"{link:<2d}" for link in links) + "\n"

    def draw_lattice(self) -> str:
        """Draw the lattice"""
        picture = ""
        for y in range(self.lattice.Ly - 1, 0, -1):
            picture += (
                self.draw_row_links(y)
                + self.draw_vert_empty()
                + self.draw_vert_links(y - 1)
                + self.draw_vert_empty()
            )
        picture += self.draw_row_links(0)

        if self.lattice.pbc_y:
            picture = (
                self.draw_vert_links(self.lattice.Ly - 1)
                + self.draw_vert_empty()
                + picture
                + self.draw_vert_empty()
                + self.draw_vert_links(self.lattice.Ly - 1)
            )
        return picture
