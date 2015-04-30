#include "parse_cmd_line.h"
namespace po = boost::program_options;

void parse_cmd_line( int argc, char *argv[], int &dim, std::string &config_file )
{
    try {
        po::options_description cmd_line_options("Allowed options");
        cmd_line_options.add_options()
            ("help,h", "produce help message")
	    ("dimension,d", po::value<int>, "specify dimension. 1, 2, and 3 are supported.");
	
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
	    po::command_line_parser(argc, argv).options(all_options).positional(p).run(),
	    vm);
        po::notify(vm);    

        if ( vm.count("help") ) {
	    std::cout << "Particle-in-cell simulation program." << std::endl;
	    std::cout << "Usage: ./epicf [OPTIONS] config-file" << std::endl;
            std::cout << visible_options << "\n";
            exit( EXIT_FAILURE );
        }
	if ( vm.count("dimension") ) {
	    dim = vm["dimension"].as< int >();	    
            std::cout << "Dimension is " << dim << std::endl;
	    if( dim < 1 || dim > 3 ){
		std::cout << "Unsupported dimension" << std::endl;
		exit( EXIT_FAILURE );
	    }
        }
        if ( vm.count("config" ) ) {
	    std::vector<std::string> all_positional_args = vm["config"].as< std::vector<std::string> >();
	    if ( all_positional_args.size() > 1 ) {
		std::cout << "Single config file is expected. \n";
		exit( EXIT_FAILURE );
	    }
	    config_file = all_positional_args.at(0);
            std::cout << "Config file is " << config_file << std::endl;
        } else {
	    std::cout << "Error: config file is not specified." << std::endl;
	    std::cout << "See './epicf -h' for usage info." << std::endl;
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
