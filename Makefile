### Allows better regexp support.
SHELL:=/bin/bash -O extglob

##### Compilers
#CC=clang++
CC=g++
HDF5FLAGS=-I/usr/include/hdf5/serial -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_BSD_SOURCE -D_FORTIFY_SOURCE=2 -g -fstack-protector-strong -Wformat -Werror=format-security
WARNINGS=-Wall
CFLAGS = ${HDF5FLAGS} -O2 -std=c++11 ${WARNINGS}
LDFLAGS = 

### Libraries
COMMONLIBS=-lm
BOOSTLIBS=-lboost_program_options
HDF5LIBS=-L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5_hl -lhdf5 -Wl,-z,relro -lpthread -lz -ldl -lm -Wl,-rpath -Wl,/usr/lib/x86_64-linux-gnu/hdf5/serial
LIBS=${COMMONLIBS} ${BOOSTLIBS} ${HDF5LIBS}

### Sources and executable
CPPSOURCES=$(wildcard *.cpp)
CPPHEADERS=$(wildcard *.h)
OBJECTS=$(CPPSOURCES:%.cpp=%.o)
EXECUTABLE=ef.out
MAKE=make
TINYEXPR=./lib/tinyexpr
TINYEXPR_OBJ=./lib/tinyexpr/tinyexpr.o
SUBDIRS=doc

$(EXECUTABLE): $(OBJECTS) $(TINYEXPR)
	$(CC) $(LDFLAGS) $(OBJECTS) $(TINYEXPR_OBJ) -o $@ $(LIBS)

$(OBJECTS):%.o:%.cpp $(CPPHEADERS)
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

