#include <iostream>
#include <string>
#include "config.h"
#include "domain.h"
#include "parse_cmd_line.h"

void pic_simulation( Config &conf );

int main( int argc, char *argv[] )
{
    std::string config_file;

    // prepare everything
    PetscErrorCode ierr;
    PetscMPIInt mpi_comm_size;
    int mpi_process_rank;
    PetscInitialize( &argc, &argv, (char*)0, NULL );
    ierr = MPI_Comm_size( PETSC_COMM_WORLD, &mpi_comm_size); CHKERRXX(ierr);
    MPI_Comm_rank( PETSC_COMM_WORLD, &mpi_process_rank );


    //// Parse command line
    parse_cmd_line( argc, argv, config_file );
    //// Read config
    Config conf( config_file );
    if ( mpi_process_rank == 0 )
    	conf.print();
    // run simulation
    pic_simulation( conf );

    // finalize_whatever_left
    ierr = PetscFinalize(); CHKERRXX(ierr);
    return 0;
}

void pic_simulation( Config &conf )
{
  Domain dom( conf );

  // Echo inner_domains
  // dom.inner_regions.print();

  // fields in domain without any particles
  dom.eval_and_write_fields_without_particles( conf );
  // run simulation
  dom.run_pic( conf );

  return;
}
