#include "domain.h"

void pic_simulation();
void write_nth_step( Domain *dom, 
		     int current_step, int step_to_save );

int main(int argc, char *argv[])
{
  // prepare_everything();
  pic_simulation();
  // finalize_whatever_left();
  return 0;
}

void pic_simulation()
{
  Domain dom;
  int total_time_steps;
  int time_step_to_save;

  domain_prepare( &dom );

  total_time_steps = dom.time_grid.total_nodes;
  time_step_to_save = total_time_steps/10;
  
  for (int i = 0; i < total_time_steps; i++){
      domain_run_pic( &dom );
      write_nth_step( &dom, i, time_step_to_save );
  }
  domain_free( &dom );
  return;
}

void write_nth_step( Domain *dom, int current_step, int step_to_save )
{
    if ( !( current_step % step_to_save ) ){	
	domain_write( dom );
    }
    return;
}
