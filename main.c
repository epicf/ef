#include "config.h"
#include "domain.h"
#include "parse_args.h"

void pic_simulation( Config *conf );
void write_nth_step( Domain *dom, Config *conf,
		     int current_step, int step_to_save );

int main(int argc, char *argv[])
{

    char *config_file = NULL;
    Config conf;

    // prepare everything
    //// Parse command line
    parse_args( argc, argv, &config_file );
    printf( "CONFIG = %s \n", config_file );
    //// Read config
    config_read( config_file, &conf );
    config_print( &conf );
    // run simulation
    pic_simulation( &conf );
    // finalize_whatever_left
    return 0;
}

void pic_simulation( Config *conf )
{
  Domain dom;
  int total_time_steps;
  int time_step_to_save;

  domain_prepare( &dom, conf );
  domain_write( &dom, conf );

  total_time_steps = dom.time_grid.total_nodes;
  time_step_to_save = total_time_steps/10;
  
  for (int i = 0; i < total_time_steps; i++){
      domain_run_pic( &dom );
      write_nth_step( &dom, conf, i, time_step_to_save );
  }
  domain_free( &dom );
  return;
}

void write_nth_step( Domain *dom, Config *conf, int current_step, int step_to_save )
{
    if ( !( current_step % step_to_save ) ){	
	domain_write( dom, conf );
    }
    return;
}
