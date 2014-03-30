FC=gfortran
FFLAGS= -O2 -fdefault-real-8
#FFLAGS= -O2 -freal-4-real-8
FSOURCES=$(wildcard *.f)
FOBJECTS=$(FSOURCES:%.f=%.o)
STATICLIB=libfishpack_dbl.a

$(STATICLIB):  $(FOBJECTS)
	ar r $@ $(FOBJECTS); ranlib $@

$(FOBJECTS):%.o:%.f
	$(FC) $(FFLAGS) -c $< -o $@

clean:
	rm -f *.o $(STATICLIB)
