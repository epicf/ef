### Allows better regexp support.
SHELL:=/bin/bash -O extglob

##### Compilers
##### Prll
#export OMPI_CXX=clang++
CC = mpic++
CFLAGS = -isystem /usr/include/petsc -isystem /usr/include/oce -O2 -std=c++11 -Wall -fbounds-check -Warray-bounds -fsanitize=address
LDFLAGS = 

### Libraries
COMMONLIBS=-lm
SANITIZER=-lasan
BOOSTLIBS=-lboost_program_options
PETSCLIBS=-lpetsc
OPENCASCADELIBS=-lTKXSBase -lTKernel -lTKBRep -lTKMath -lTKSTEP -lTKBool -lTKTopAlgo -lTKPrim
LIBS=${COMMONLIBS} ${BOOSTLIBS} ${PETSCLIBS} ${OPENCASCADELIBS} ${SANITIZER}

### Sources and executable
CPPSOURCES=$(wildcard *.cpp)
CPPHEADERS=$(wildcard *.h)
OBJECTS=$(CPPSOURCES:%.cpp=%.o)
EXECUTABLE=epicf.out
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

