#include <iostream>
#include <string>
#include "Config.hpp"
#include "Domain.hpp"
#include "parse_cmd_line.h"

template< int dim >
void pic_simulation( Config<dim> &conf );

int main( int argc, char *argv[] )
{
    std::string config_file;
    int dim;
    
    // prepare everything
    //// Parse command line
    parse_cmd_line( argc, argv, dim, config_file );
    //// Read config
    if ( dim == 1 ) {
	Config<1> conf( config_file );
	conf.print();
	// run simulation
	pic_simulation( conf );    
    } else if ( dim == 2 ) {
	Config<2> conf( config_file );
	conf.print();
	// run simulation
	pic_simulation( conf );    
    } else if ( dim == 3 ) {
	Config<3> conf( config_file );
	conf.print();
	// run simulation
	pic_simulation( conf );
    } else {
	std::cout << "Unsupported dim" << std::endl;
	exit( EXIT_FAILURE );
    }
    

    

    // finalize_whatever_left
    return 0;
}

template< int dim >
void pic_simulation( Config<dim> &conf )
{
    Domain<dim> dom( conf );
  
    // save domain state just after initialisation
    dom.write( conf );
    // run simulation
    dom.run_pic( conf );
    
    return;
}
