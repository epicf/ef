#!/usr/bin/Rscript 
rm( list=ls() )

necessary_packages <- c("optparse")
packages_to_intsall <-
  necessary_packages[ !( necessary_packages %in% installed.packages()[,"Package"] ) ]
if( length( packages_to_intsall ) ) {
  print( paste( "The following necessary packages are missing:", packages_to_intsall ) )
  print( paste( "Attempting to install." ) )
  install.packages( packages_to_intsall )
}
invisible( sapply( necessary_packages, library, character.only=T ) )

extract_domain_properties <- function( file ){
    sections <- c( "time", "grid", "particles" )
    domain_properties <- vector( "list", length(sections) )
    names( domain_properties ) <- sections
    time_subsections <- c( "total_time",                                        
                           "current_time",                                          
                           "time_step_size",
                           "total_nodes", 
                           "current_node" )
    grid_subsections <- c( "x_volume_size",
                           "y_volume_size",
                           "x_cell_size",
                           "y_cell_size",
                           "x_nodes",
                           "y_nodes" )
    particles_subsections <- c( "total_number_of_particles" )

    domain_properties$time <- vector( "list", length( time_subsections ) )
    domain_properties$grid <- vector( "list", length( grid_subsections ) )
    domain_properties$particles <- vector( "list", length( particles_subsections ) )
    names( domain_properties$time ) <- time_subsections
    names( domain_properties$grid ) <- grid_subsections
    names( domain_properties$particles ) <- particles_subsections

    section_name_to_pattern_to_search <- function( section_name ) {
        no_underscores <- gsub( "_", " ", section_name )
        capitalize_first_letter <- paste( toupper( substring( no_underscores, 1, 1 ) ),
                                          substring( no_underscores, 2 ),
                                          sep="" )
        in_the_beginning_of_the_line <- paste( "^", capitalize_first_letter, sep="" )
        return ( in_the_beginning_of_the_line )
    }

    search_pattern_in_file <- function( pattern, file ) {
        return( grep( pattern, file, value=T ) )
    }

    extract_values_from_found_pattern <- function( s ) {
        return( as.numeric( strsplit( s, "=")[[c(1,2)]] ) )
    }
    
    for ( i in seq_along( domain_properties ) ) {
        sec <- domain_properties[[i]]
        patterns_to_search <- section_name_to_pattern_to_search( names( sec ) )
        found_patterns <- lapply( patterns_to_search,
                                  function(x){ search_pattern_in_file( x, file ) } )
        names( found_patterns ) <- names( sec )
        extracted_values <- lapply( found_patterns, extract_values_from_found_pattern )
        domain_properties[[i]] <- extracted_values
    }

    return( domain_properties )

}

construct_output_filename <- function( filename, value_to_plot ) {
  return ( sub( "(.*)\\.(.*)",
                paste("\\1_", value_to_plot, ".png", sep=''),
                filename) )
}

grid_nodes_to_real_coords <- function( on_grid_values, domain_properties ) {
    x_starting_value <- 0
    on_grid_values$x <- x_starting_value + on_grid_values$nx * domain_properties$grid$x_cell_size
    y_starting_value <- 0
    on_grid_values$y <- y_starting_value + on_grid_values$ny * domain_properties$grid$y_cell_size
    return( on_grid_values )    
}

extract_data_for_potential_plot <- function( data ){
  grid_start <- grep( "^### Grid", data )
  grid_header <- 8
  grid_end <- grep( "^### Particles", data ) - 1
  cols_to_read <- c( "numeric", "numeric", "NULL", "numeric", "NULL", "NULL" )
  col_names=c("nx", "ny", NA, "phi", NA, NA )
  potential_data <- read.table(
                      textConnection( data[ (grid_start+grid_header) : grid_end ] ),
                      colClasses = cols_to_read, 
                      col.names = col_names )
  return( potential_data )
}

plot_potential <- function( dataframe_to_plot, domain_properties, filename ) {
    x <- unique( dataframe_to_plot$x )
    y <- unique( dataframe_to_plot$y )
    z <- xtabs( dataframe_to_plot$phi ~ dataframe_to_plot$x + dataframe_to_plot$y,
                dataframe_to_plot)
    as.data.frame.matrix( z )
###
    png( filename )
    xmin <- 0
    xmax <- max( x )
    ymin <- 0
    ymax <- max( y )
    axis_ticks_step <- 10
    filled.contour(x, y, z,
                   color = topo.colors,
                   xlim <- c( xmin, xmax ),
                   ylim <- c( ymin, ymax ),
                   plot.axes = { axis( 1 ) #, seq( xmin, xmax, by = axis_ticks_step ))
                                 axis( 2 ) },#, seq( ymin, ymax, by = axis_ticks_step ))},
                   plot.title = title(main="Potential", xlab = "nx", ylab = "ny" ),
                   )
    dev.off()
}


extract_data_for_density_plot <- function( data ){
  grid_start <- grep( "^### Grid", data )
  grid_header <- 8
  grid_end <- grep( "^### Particles", data ) - 1
  cols_to_read <- c( "numeric", "numeric", "numeric", "NULL", "NULL", "NULL" )
  col_names=c("nx", "ny", "charge_density", NA, NA, NA )
  density_data <- read.table(
                      textConnection( data[ (grid_start+grid_header) : grid_end ] ),
                      colClasses = cols_to_read, 
                      col.names = col_names )
  return( density_data )
}

plot_density <- function( dataframe_to_plot, domain_properties, filename ) {
    x <- unique( dataframe_to_plot$nx )
    y <- unique( dataframe_to_plot$ny )
    z <- xtabs(
          dataframe_to_plot$charge_density ~ dataframe_to_plot$nx + dataframe_to_plot$ny,
          dataframe_to_plot )
    as.data.frame.matrix( z )
###
    png( filename )
    xmin <- 0
    xmax <- domain_properties$grid$x_nodes - 1
    ymin <- 0
    ymax <- domain_properties$grid$y_nodes - 1
    axis_ticks_step <- 10
    filled.contour(x, y, z,
                   color = topo.colors,
                   xlim <- c( xmin, xmax ),
                   ylim <- c( ymin, ymax ),
                   plot.axes = { axis(1, seq( xmin, xmax, by = axis_ticks_step ))
                                 axis(2, seq( ymin, ymax, by = axis_ticks_step ))},
                   plot.title = title(main="Charge density", xlab = "nx", ylab = "ny" ),
                   )
    dev.off()
}


extract_data_for_particles_coords_plot <- function( data ){
  particles_start <- grep( "^### Particles", data )
  particles_header <- 2
  cols_to_read <- c( "NULL", "NULL", "NULL", "numeric", "numeric", "numeric", "numeric" )
  col_names <- c(NA, NA, NA, "x", "y", "px", "py" )
  particles_data <-
    read.table( textConnection( tail( data, -( particles_start + particles_header ) ) ),
                colClasses = cols_to_read, 
                col.names = col_names )               
  return( particles_data )
}

plot_particles_coords <- function( particles_data, domain_properties, outfile ) {
    png( outfile )
    xmin <- 0
    xmax <- domain_properties$grid$x_volume_size
    ymin <- 0
    ymax <- domain_properties$grid$y_volume_size  
    xlim <- c( xmin, xmax )
    ylim <- c( ymin, ymax )
    plot(NA, NA,
         xaxs="i", yaxs="i", axes = F,
         xlim = xlim,
         ylim = ylim,
         ##         las = 1,
         ##         cex.lab=2.5, cex.axis=1.5,
         main = "Particles", 
         xlab = "x", ylab = "y"
         )

    points( particles_data$x, particles_data$y,
            pch = 20, col = "black")

    mean_p <- mean( sqrt( particles_data$px^2 + particles_data$py^2 ) )
    max_p <- max( sqrt( particles_data$px^2 + particles_data$py^2 ) )
    arrows( particles_data$x,
            particles_data$y,
            particles_data$x + particles_data$px/mean_p,
            particles_data$y + particles_data$py/mean_p,
            length = 0.05, angle = 20, 
            col = "red" )
    
    box()
    axis(1, las = 1, lwd.ticks=4)
    axis(2, las = 1, lwd.ticks=4)         

    dev.off()
}




option_list <-
  list( make_option(c("-p", "--potential"), action="store_true",
                    help="Potential on grid"),
        make_option(c("-d", "--density"), action="store_true",
                    help="Charge density on grid"),
        make_option(c("-P", "--particles"), action="store_true",
                    help="Particles position and momentum" )
)

parser <- OptionParser( usage = "%prog [options] file", option_list = option_list )
arguments <- parse_args( parser, positional_arguments = TRUE )
opt <- arguments$options
files <- arguments$args
any_flag <- !is.null( opt$potential ) ||
            !is.null( opt$density ) ||
            !is.null( opt$particles )

###file <- "out0001.dat"

if ( any_flag ) {
  for ( file in files ){
    if ( !file.exists( file ) ) {
      print( paste( "Can't find file: ", file, ". Ignoring.", sep='' ) )
      next
    }

    data <- readLines( file )
    domain_properties <- extract_domain_properties( data )
    
    if ( !is.null( opt$potential ) ) {
      print( paste( "Plotting potential for", file ) )
      data_potential <- extract_data_for_potential_plot( data )
      data_potential <- grid_nodes_to_real_coords( data_potential, domain_properties )
      output_filename <- construct_output_filename( file, "potential" )
      plot_potential( data_potential, domain_properties, output_filename )
      rm( list = c( "data_potential" ) )
    }

    if ( !is.null( opt$density ) ) {
      print( paste( "Plotting density for", file ) )
      data_density <- extract_data_for_density_plot( data )
      output_filename <- construct_output_filename( file, "density" )
      plot_density( data_density, domain_properties, output_filename )
      rm( list = c( "data_density" ) )
    }

    if ( !is.null( opt$particles ) ) {
      print( paste( "Plotting particles for", file ) )
      data_particles <- extract_data_for_particles_coords_plot( data )
      output_filename <- construct_output_filename( file, "particles" )
      plot_particles_coords( data_particles, domain_properties, output_filename )
      rm( list = c( "data_particles" ) )
    }
      
    rm( list = c("data") )  
    
  }
} else {
  print_help( parser )
  stop("At least one flag should be specified.")
}

## Plotting 

if ( !is.null( opt$hist ) ) {
    toplot <- unlist( strsplit( opt$hist, "," ) )
    mch <- match( c( "x", "y", "px", "py" ), toplot )
    print( c( "plotting hist ", toplot, mch ) )
}
