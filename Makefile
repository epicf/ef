### Allows better regexp support.
SHELL:=/bin/bash -O extglob

##### Compilers
##### Prll
#export OMPI_CXX=clang++
CC = mpic++
HDF5FLAGS=-I/usr/include/hdf5/openmpi -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_BSD_SOURCE -D_FORTIFY_SOURCE=2 -g -fstack-protector-strong -Wformat -Werror=format-security
PETSCFLAGS=-isystem /usr/include/petsc
SUPPRESS_MPI_C11_WARNING=-Wno-literal-suffix
WARNINGS=-Wall -fbounds-check -Warray-bounds -fsanitize=address
CFLAGS = ${HDF5FLAGS} ${PETSCFLAGS} -O2 -std=c++11 ${WARNINGS} ${SUPPRESS_MPI_C11_WARNING}
LDFLAGS = 

### Libraries
COMMONLIBS=-lm -fsanitize=address
SANITIZER=-lasan
BOOSTLIBS=-lboost_program_options
PETSCLIBS=-lpetsc
HDF5LIBS=-L/usr/lib/x86_64-linux-gnu/hdf5/openmpi -lhdf5_hl -lhdf5 -Wl,-z,relro -lpthread -lz -ldl -lm -Wl,-rpath -Wl,/usr/lib/x86_64-linux-gnu/hdf5/openmpi
LIBS=${COMMONLIBS} ${BOOSTLIBS} ${PETSCLIBS} ${HDF5LIBS} ${SANITIZER}

### Sources and executable
CPPSOURCES=$(wildcard *.cpp)
CPPHEADERS=$(wildcard *.h)
OBJECTS=$(CPPSOURCES:%.cpp=%.o)
EXECUTABLE=ef.out
MAKE=make
SUBDIRS=doc

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@ $(LIBS)

$(OBJECTS):%.o:%.cpp $(CPPHEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

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

