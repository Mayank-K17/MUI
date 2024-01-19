#!/bin/bash

CC=mpicxx -cxx=icpx -fsycl
CFLAGS	= -std=c++17 -O3 
SYCL_PATH = /opt/intel/oneapi/compiler/2023.1.0/linux/include
SYCL_INC_PATH = /opt/intel/oneapi/compiler/2023.1.0/linux/include/sycl

SCR = $(wildcard *.cpp)
EXE = $(SCR:.cpp=)

default: $(EXE)

% : %.cpp
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(EXE)
