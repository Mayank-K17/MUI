#!/bin/bash

CC	= mpicxx -cxx=icpx -fsycl
CFLAGS	= -std=c++17 -O3 -fsycl-targets=nvptx64-nvidia-cuda
SYCL_PATH = /opt/intel/oneapi/compiler/2023.1.0/linux/include
SYCL_INC_PATH = /opt/intel/oneapi/compiler/2023.1.0/linux/include/sycl
EIGEN_PATH ?= ../../../../eigen-3.4.0/

SCR = $(wildcard *.cpp)
EXE = $(SCR:.cpp=)

default: $(EXE)

% : %.cpp
	$(CC) $(CFLAGS) -I$(EIGEN_PATH) $< -o $@

clean:
	rm -f $(EXE)
