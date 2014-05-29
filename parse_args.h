#ifndef _PARSE_ARGS_H_
#define _PARSE_ARGS_H_

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
    
void print_usage();
void print_help();
void parse_args( int argc, char *argv[], char *config_file[] );

#endif /* _PARSE_ARGS_H_ */
