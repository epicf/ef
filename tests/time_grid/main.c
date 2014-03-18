#include "../../time_grid.h"

int main(int argc, char *argv[])
{
    Time_grid tg;
    double total_time = 10.0;
    double step_size = 0.01;
    tg = time_grid_init( total_time, step_size );
    time_grid_print(tg);

    return 0;
}
