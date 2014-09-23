### Allows better regexp support.
SHELL:=/bin/bash -O extglob

##### Compilers
##### GNU
##CC = g++
## Detect errors:
##CFLAGS = -O2 -std=c++11 -Wall -fbounds-check -Warray-bounds -fsanitize=address `pkg-config --cflags glib-2.0`
##LDFLAGS = -fsanitize=address
### Usual flags
#CFLAGS = -std=c99 -O2
#LDFLAGS =
##### Clang
CC = clang++
CFLAGS = -O1 -std=c++11 -g -Wall \
	-fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls `pkg-config --cflags glib-2.0`
LDFLAGS = -g -fsanitize=address

### Libraries
LIBS=-lm -L./fishpack -lfishpack_dbl -lboost_program_options `pkg-config --libs glib-2.0`

### Sources and executable
CPPSOURCES=$(wildcard *.cpp)
CPPHEADERS=$(wildcard *.h)
OBJECTS=$(CPPSOURCES:%.cpp=%.o)
EXECUTABLE=epicf.out
MAKE=make
SUBDIRS=fishpack

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) subdirs
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@ $(LIBS)

$(OBJECTS):%.o:%.cpp $(CHEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: subdirs $(SUBDIRS) clean cleansubdirs cleanall

subdirs: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

clean: 
	rm -f *.o *.out *.mod *.zip

cleansubdirs:
	for X in $(SUBDIRS); do $(MAKE) clean -C $$X; done 

cleanall: clean cleansubdirs
