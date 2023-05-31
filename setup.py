"""
C++のコードをコンパイルしてPythonのモジュールとして吐き出すためのコード.
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

source = [
    "cpp/helper.cpp",
    "cpp/prob_reversi_helper.pyx"
    ]

setup(
    cmdclass = dict(build_ext = build_ext),
    ext_modules = [
        Extension(
            "prob_reversi_helper",             
            source,
            language='c++',    
        )
    ]
)