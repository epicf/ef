#include "spatial_mesh.h"

int main(int argc, char *argv[])
{
    Spatial_mesh spm;
    double x_size = 10.0;
    double x_step = 1;
    double y_size = 10.0;
    double y_step = 1;
    spm = spatial_mesh_init( x_size, x_step, y_size, y_step );
    spatial_mesh_print( &spm );
    return 0;
}
