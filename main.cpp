#include <iostream>
#include <string>
#include "config.h"
#include "domain.h"
#include "parse_cmd_line.h"

void pic_simulation( Config *conf );

int main(int argc, char *argv[])
{
    std::string config_file;
    Config conf;

    // prepare everything
    //// Parse command line
    parse_cmd_line( argc, argv, config_file );
    std::cout << "main: CONFIG = " << config_file << std::endl;
    //// Read config
    config_read( config_file.c_str(), &conf );
    config_print( &conf );
    // run simulation
    pic_simulation( &conf );
    // finalize_whatever_left
    return 0;
}

void pic_simulation( Config *conf )
{
  Domain dom( conf );
  
  // save domain state just after initialisation
  dom.write( conf );
  // run simulation
  dom.run_pic( conf );

  return;
}
