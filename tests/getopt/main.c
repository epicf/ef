#include "parse_args.h"

int main( int argc, char *argv[] ){
    char *config_file = NULL;

    parse_args( argc, argv, &config_file );
    printf( "CONFIG = %s \n", config_file );

    return 0;
}
