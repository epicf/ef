cd
mkdir petsc
#mkdir petsc/debug
mkdir petsc/opt
### get source
#git clone -b maint https://bitbucket.org/petsc/petsc petsc
wget http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-lite-3.6.2.tar.gz

### compile debug-version
#tar -xvzf petsc-lite-3.6.2.tar.gz
#cd petsc-3.6.2
#PETSC_FLAGS="--with-cc=mpicc --with-cxx=mpic++ --with-fc=0 --with-clanguage=cxx \
#        --download-hypre=yes \
#        --download-metis=yes --download-parmetis=yes  --download-superlu_dist" 
#        #--download-mumps=yes --with-scalapack --with-fc=gfortran \
#./configure ${PETSC_FLAGS} --prefix=${HOME}/petsc/debug --with-debugging=1
#make PETSC_DIR=${HOME}/petsc-3.6.2 PETSC_ARCH=arch-linux2-cxx-debug all
#make PETSC_DIR=${HOME}/petsc-3.6.2 PETSC_ARCH=arch-linux2-cxx-debug test
#make install DESTDIR=${HOME}/petsc/debug
#make distclean
#cd ..
#rm -rf petsc-3.6.2

### compile production-version
tar -xvzf petsc-lite-3.6.2.tar.gz
cd petsc-3.6.2
PETSC_FLAGS="--with-cc=mpicc --with-cxx=mpic++ --with-fc=0 --with-clanguage=cxx \
        --download-hypre=yes \
        --download-metis=yes --download-parmetis=yes  --download-superlu_dist" 
./configure ${PETSC_FLAGS} --prefix=${HOME}/petsc/opt --with-debugging=0
make PETSC_DIR=${HOME}/petsc-3.6.2 PETSC_ARCH=arch-linux2-cxx-opt all
make PETSC_DIR=${HOME}/petsc-3.6.2 PETSC_ARCH=arch-linux2-cxx-opt test
make install DESTDIR=${HOME}/petsc/opt
cd ../
rm -rf petsc-3.6.2
rm petsc-lite-3.6.2.tar.gz
