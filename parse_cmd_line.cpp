#include "parse_cmd_line.h"
namespace po = boost::program_options;

void parse_cmd_line( int argc, char *argv[], std::string &config_file )
{
    try {
        po::options_description cmd_line_options("Allowed options");
        cmd_line_options.add_options()
            ("help,h", "produce help message");
	
	po::options_description positional_parameters;
	positional_parameters.add_options()
	    ("config", po::value< std::vector<std::string> >(), "specify config file");
	po::positional_options_description p;	
	p.add("config", -1);

	po::options_description all_options;
	all_options.add( cmd_line_options ).add( positional_parameters );

	po::options_description visible_options;
	visible_options.add( cmd_line_options );
		
        po::variables_map vm;        
        po::store( 
	    po::command_line_parser(argc, argv).options(all_options).positional(p).allow_unregistered().run(),
	    vm);
        po::notify(vm);    

        if ( vm.count("help") ) {
	    std::cout << "Particle-in-cell simulation program." << std::endl;
	    std::cout << "Usage: ./ef [OPTIONS] config-file" << std::endl;
            std::cout << visible_options << "\n";
            exit( EXIT_FAILURE );
        }
        if ( vm.count("config" ) ) {
	    std::vector<std::string> all_positional_args = vm["config"].as< std::vector<std::string> >();
	    //if ( all_positional_args.size() > 1 ) {
	    //std::cout << "Single config file is expected. \n";
	    //	exit( EXIT_FAILURE );
	    //}
	    //config_file = all_positional_args.at(0);
	    config_file = all_positional_args.back();
            std::cout << "Config file is " << config_file << std::endl;
        } else {
	    std::cout << "Error: config file is not specified." << std::endl;
	    std::cout << "See './ef -h' for usage info." << std::endl;
            exit( EXIT_FAILURE );
        }
    }
    catch( std::exception& e ) {
        std::cerr << "error: " << e.what() << "\n";
        exit( EXIT_FAILURE);
    }
    catch( ... ) {
        std::cerr << "Exception of unknown type!\n";
    }

    return;
}
