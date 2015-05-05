#include <iostream>
#include <string>
#include "Config.hpp"
#include "Domain.hpp"
#include "parse_cmd_line.h"

void pic_simulation( Config &conf );

int main( int argc, char *argv[] )
{
    std::string config_file;
    int dim;
    
    // prepare everything
    //// Parse command line
    parse_cmd_line( argc, argv, dim, config_file );
    //// Read config
    Config<dim> conf( config_file );
    conf.print();
    
    // run simulation
    pic_simulation( dim, conf );

    // finalize_whatever_left
    return 0;
}

void pic_simulation( int &dim, Config &conf )
{
    Domain<dim> dom( conf );
  
    // save domain state just after initialisation
    dom.write( conf );
    // run simulation
    dom.run_pic( conf );
    
    return;
}
