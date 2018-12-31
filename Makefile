### Allows better regexp support.
SHELL:=/bin/bash -O extglob

##### Compilers
#CC=clang++
CC=g++
NVCC=nvcc

HDF5FLAGS=-I/usr/include/hdf5/serial -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FORTIFY_SOURCE=2 -g -fstack-protector-strong -Wformat -Werror=format-security
NVCCHDF5FLAGS=-I/usr/include/hdf5/serial -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FORTIFY_SOURCE=2 -g
CUDAINCLUDES=-I/usr/local/cuda/include

WARNINGS=-Wall
NVCCWARNINGS=
CFLAGS = ${HDF5FLAGS} ${CUDAINCLUDES} -O2 -std=c++11 ${WARNINGS}
NVCCFLAGS= ${NVCCHDF5FLAGS} ${CUDAINCLUDES} -O2 -std=c++11 -arch=sm_30 ${NVCCWARNINGS}
LDFLAGS = 

### Libraries
COMMONLIBS=-lm
BOOSTLIBS=-lboost_program_options
HDF5LIBS=-L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_hl -lhdf5 -Wl,-z,relro -lpthread -lz -ldl -lm -Wl,-rpath -Wl,/usr/lib/x86_64-linux-gnu/hdf5/serial
CUDALIBS=-L/usr/local/cuda/lib64/ -lcudart
LIBS=${COMMONLIBS} ${BOOSTLIBS} ${HDF5LIBS} ${CUDALIBS}

### Sources and executable
CPPSOURCES=$(wildcard *.cpp)
CPPHEADERS=$(wildcard *.h)
CUSOURCES=$(wildcard *.cu)
CUHEADERS=$(wildcard *.cuh)

CUOBJECTS=$(CUSOURCES:%.cu=%.o)
OBJECTS=$(CPPSOURCES:%.cpp=%.o)

EXECUTABLE=ef.out
MAKE=make
TINYEXPR=./lib/tinyexpr
TINYEXPR_OBJ=./lib/tinyexpr/tinyexpr.o
SUBDIRS=doc

$(EXECUTABLE): $(OBJECTS) $(CUOBJECTS) $(TINYEXPR) 
	$(CC) $(LDFLAGS) $(OBJECTS) $(CUOBJECTS) $(TINYEXPR_OBJ) -o $@ $(LIBS)
$(CUOBJECTS):%.o:%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
$(OBJECTS):%.o:%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: allsubdirs $(SUBDIRS) $(TINYEXPR) clean cleansubdirs cleanall

allsubdirs: $(SUBDIRS)

$(TINYEXPR):
	$(MAKE) -C $@

$(SUBDIRS):
	$(MAKE) -C $@

all: $(EXECUTABLE) doc

clean: cleansublibs
	rm -f *.o *.out *.mod *.zip

cleansublibs:
	for X in $(TINYEXPR); do $(MAKE) clean -C $$X; done 

cleansubdirs:
	for X in $(SUBDIRS); do $(MAKE) clean -C $$X; done 

cleanall: clean cleansubdirs

