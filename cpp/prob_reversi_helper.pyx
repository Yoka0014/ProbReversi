"""
prob_reversi_helperのラッパー.
"""

from libcpp cimport bool

cdef extern from "prob_reversi_helper.h":
    cdef bool __set_board_size(int size);
    cdef unsigned long long __calc_mobility(unsigned long long p, unsigned long long o)
    cdef unsigned long long __calc_flip_discs(unsigned long long p, unsigned long long o, int coord);


def set_board_size(size):
    return __set_board_size(size)

def calc_mobility(p, o):
    return __calc_mobility(p, o)

def calc_flip_discs(p, o, coord):
    return __calc_flip_discs(p, o, coord)