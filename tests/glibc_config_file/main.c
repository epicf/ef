#include "config.h"

int main(int argc, char *argv[])
{
    Config conf;
    
    config_read( "test.conf", &conf );
    config_check_correctness( &conf );
    config_print( &conf );

    return 0;
}

