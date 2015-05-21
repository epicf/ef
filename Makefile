### Allows better regexp support.
SHELL:=/bin/bash -O extglob

##### Compilers
##### GNU debug
 CC = g++
# CFLAGS = -O2 -std=c++11 -Wall -fbounds-check -Warray-bounds -fsanitize=address
# LDFLAGS = -fsanitize=address
### GNU run
CFLAGS = -std=c++11 -O2
LDFLAGS =
##### Clang
#CC = clang++
#CFLAGS = -O1 -std=c++11 -g -Wall -fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls
#LDFLAGS = -g -fsanitize=address

### Libraries
LIBS=-lm -lgsl -lgslcblas -lboost_program_options

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

