#ifndef _PARSE_CMD_LINE_H_
#define _PARSE_CMD_LINE_H_

#include <boost/program_options.hpp>
#include <iostream>
#include <string>

//todo: create a structure for command line parameters?
void parse_cmd_line( int argc, char *argv[], int &dim, std::string &config_file );

#endif /* _PARSE_CMD_LINE_H_ */
