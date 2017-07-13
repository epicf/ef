### Allows better regexp support.
SHELL:=/bin/bash -O extglob

##### Compilers
##### Prll
#export OMPI_CXX=clang++
CC = mpic++
#CC = mpiicpc 
HDF5FLAGS=-I/usr/include/hdf5/openmpi -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_BSD_SOURCE -D_FORTIFY_SOURCE=2 -g -fstack-protector-strong -Wformat -Werror=format-security
CFLAGS = ${HDF5FLAGS} -O2 -std=c++11
LDFLAGS = 

### Libraries
COMMONLIBS=-lm
BOOSTLIBS=-lboost_program_options
HDF5LIBS=-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi -lhdf5_hl -lhdf5 -Wl,-z,relro -lpthread -lz -ldl -lm -Wl,-rpath -Wl,/usr/lib/x86_64-linux-gnu/hdf5/openmpi
LIBS=${COMMONLIBS} ${BOOSTLIBS} ${HDF5LIBS}

### Sources and executable
CPPSOURCES=$(wildcard *.cpp)
CPPHEADERS=$(wildcard *.h)
OBJECTS=$(CPPSOURCES:%.cpp=%.o)
EXECUTABLE=ef.out
MAKE=make
SUBDIRS=doc

$(EXECUTABLE): $(OBJECTS) cuda.o
	$(CC) $(LDFLAGS) $(OBJECTS) cuda.o -o $@ $(LIBS) -L /usr/local/cuda/lib64/ -lcudart

$(OBJECTS):%.o:%.cpp $(CPPHEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

cuda.o: 
	nvcc  -I ~/cusplibrary -o $@ -c field_solver.cu 

.PHONY: allsubdirs $(SUBDIRS) clean cleansubdirs cleanall

allsubdirs: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

all: $(EXECUTABLE) doc

clean: 
	rm -f *.o *.out *.mod *.zip

cleansubdirs:
	for X in $(SUBDIRS); do $(MAKE) clean -C $$X; done 

cleanall: clean cleansubdirs

