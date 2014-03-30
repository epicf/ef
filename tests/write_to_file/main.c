#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

void test_write( );
char *construct_output_filename( const char *output_filename_prefix, 
				 const int current_time_step,
				 const char *output_filename_suffix );
void hello_write_to_file( FILE *f );

int main(int argc, char *argv[])
{
    test_write();
    return 0;
}

void test_write()
{
    const char output_filename_prefix[] = "out";
    const char output_filename_suffix[] = ".dat";
    char *file_name_to_write;
    int step = 5;
    /* int number_len = ((CHAR_BIT * sizeof(int) - 1) / 3 + 2); // don't know how this works */
    /* int prefix_len = strlen(output_filename_prefix); */
    /* int suffix_len = strlen(output_filename_suffix); */
    /* int ENOUGH = prefix_len + number_len + suffix_len; */
    /* char file_name_to_write[ENOUGH]; */
    
    file_name_to_write = construct_output_filename( output_filename_prefix, step, output_filename_suffix  );
    FILE *f = fopen(file_name_to_write, "w");
    if (f == NULL) {
	printf("Error opening file!\n");
	exit( EXIT_FAILURE );
    }
    printf ("Writing step %d to file %s\n", step, file_name_to_write);

    hello_write_to_file( f );

    free( file_name_to_write );
    fclose(f);
    return;
}

char *construct_output_filename( const char *output_filename_prefix, 
				 const int current_time_step,
				 const char *output_filename_suffix )
{    
    int prefix_len = strlen(output_filename_prefix);
    int suffix_len = strlen(output_filename_suffix);
    int number_len = ((CHAR_BIT * sizeof(int) - 1) / 3 + 2); // don't know how this works
    int ENOUGH = prefix_len + number_len + suffix_len;
    char *filename;
    filename = (char *) malloc( ENOUGH * sizeof(char) );
    snprintf(filename, ENOUGH, "%s%.4d%s", 
	     output_filename_prefix, current_time_step, output_filename_suffix);
//    snprintf(filename, strlen(filename)+1, "%s%.4d%s", 
//	       output_filename_prefix, current_time_step, output_filename_suffix);
//    printf( "%d %d %d", prefix_len, number_len, ENOUGH );
    return filename;
}

void hello_write_to_file( FILE *f )
{
    fprintf(f, "%s", "Hello\n.");
    return;
}
