CC=gcc
CCFLAGS = -ansi -Wall -Wshadow -O2
LFLAGS = -lm

.PHONY = all clean

all: tinyexpr.o

.c.o:
	$(CC) -c $(CCFLAGS) $< -o $@

clean:
	rm -f *.o
