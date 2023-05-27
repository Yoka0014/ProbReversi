"""
prob_reversi_helperのラッパー.
"""
from libcpp cimport bool


cdef extern from "helper.h":
    cdef cppclass __Helper:
        __Helper(int)
        unsigned long long calc_mobility(unsigned long long p, unsigned long long o)
        unsigned long long calc_flip_discs(unsigned long long p, unsigned long long o, int coord);


cdef class Helper:
    cdef __Helper *__this_ptr

    def __cinit__(self, size):
        self.__this_ptr = new __Helper(size)

    def calc_mobility(self, p, o):
        return self.__this_ptr.calc_mobility(p, o)

    def calc_flip_discs(self, p, o, coord):
        return self.__this_ptr.calc_flip_discs(p, o, coord)