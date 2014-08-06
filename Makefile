### Allows better regexp support.
SHELL:=/bin/bash -O extglob

##### Compilers
##### GNU
CC = gcc
## Detect errors:
CFLAGS = -std=c99 -O2 -Wall -fbounds-check -Warray-bounds -fsanitize=address `pkg-config --cflags glib-2.0`
LDFLAGS = -fsanitize=address
### Usual flags
#CFLAGS = -std=c99 -O2
#LDFLAGS =
##### Clang
#CC = clang
#CFLAGS = -O1 -g -Wall \
	-fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls
#LDFLAGS = -g -fsanitize=address

### Libraries
LIBS=-lm -lgsl -lgslcblas -L./fishpack -lfishpack_dbl `pkg-config --libs glib-2.0`

### Sources and executable
CSOURCES=$(wildcard *.c)
CHEADERS=$(wildcard *.h)
OBJECTS=$(CSOURCES:%.c=%.o)
EXECUTABLE=epicf.out
MAKE=make
SUBDIRS=fishpack

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) subdirs
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@ $(LIBS)

$(OBJECTS):%.o:%.c $(CHEADERS)
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
