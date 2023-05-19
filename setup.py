from setuptools import setup
from cmake_build_extension import BuildExtension, CMakeExtension

setup(
    name="pystraightskeleton",
    description="Python wrapper for CGAL straight skeleton",
    url="https://github.com/pwuertz/pystraightskeleton",
    version="0.1",
    author="Peter Würtz",
    author_email="pwuertz@gmail.com",
    ext_modules=[CMakeExtension(
        name="pystraightskeleton",
        source_dir=".",
        install_prefix=".",
    )],
    cmdclass=dict(build_ext=BuildExtension),
)
