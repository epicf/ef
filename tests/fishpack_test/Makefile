##### Compilers
##### GNU
CC = gcc
## Detect errors:
CFLAGS = -std=c99 -O2 -Wall -fbounds-check -Warray-bounds -fsanitize=address
LDFLAGS = -fsanitize=address

### Libraries
LIBS= -L../../fishpack/ -lfishpack_dbl -lm
CSOURCES=$(wildcard *.c)
OBJECTS=$(CSOURCES:%.c=%.o)
EXECUTABLE=main.out

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@ $(LIBS)

$(OBJECTS):%.o:%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o *.out *.mod *.zip
