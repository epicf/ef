### Allows better regexp support.
SHELL:=/bin/bash -O extglob

##### Compilers
#CC=clang++
CC=g++
NVCC=nvcc

HDF5FLAGS=-I/usr/include/hdf5/serial -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FORTIFY_SOURCE=2 -g -fstack-protector-strong -Wformat -Werror=format-security
WARNINGS=-Wall
CFLAGS = ${HDF5FLAGS} -O2 -std=c++11 ${WARNINGS}
LDFLAGS = 

CUDAINCLUDES= -I/usr/local/cuda/include
CUDAFLAGS= ${CUDAINCLUDES} -std=c++11 -arch=sm_30

### Libraries
COMMONLIBS=-lm
BOOSTLIBS=-lboost_program_options
HDF5LIBS=-L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_hl -lhdf5 -Wl,-z,relro -lpthread -lz -ldl -lm -Wl,-rpath -Wl,/usr/local/hdf5/lib
CUDALIBS=-L/usr/local/cuda/lib64/ -lcudart
LIBS=${COMMONLIBS} ${BOOSTLIBS} ${HDF5LIBS}

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

$(EXECUTABLE): $(OBJECTS) $(TINYEXPR) $(CUOBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) $(TINYEXPR_OBJ) $(CUOBJECTS) -o $@ $(LIBS) $(CUDALIBS)
$(CUOBJECTS):%.o:%.cu $(CUHEADERS)
	$(NVCC) $(CUDAFLAGS) -I/usr/local/hdf5/include -c $< -o $@
$(OBJECTS):%.o:%.cpp $(CPPHEADERS)
	$(CC) $(CFLAGS) $(CUDAINCLUDES) -c $< -o $@

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

