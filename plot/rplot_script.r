#!/usr/bin/Rscript 
rm( list=ls() )

necessary_packages <- c("optparse", "scatterplot3d")
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
                           "z_volume_size",
                           "x_cell_size",
                           "y_cell_size",
                           "z_cell_size",
                           "x_nodes",
                           "y_nodes",
                           "z_nodes" )
    particles_subsections <- c("source_name",
                               "total_number_of_particles" )

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
        is_source_name_pattern <- any( grepl( "Source name", s ) )
        if( is_source_name_pattern ) {
            return( sapply( strsplit( s, "="), function(x){ x[2] } ) )
        } else {
            return( as.numeric( sapply( strsplit( s, "="), function(x){ x[2] } ) ) )
        }
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
    z_starting_value <- 0
    on_grid_values$z <- z_starting_value + on_grid_values$nz * domain_properties$grid$z_cell_size

    return( on_grid_values )    
}

extract_data_for_potential_plot <- function( data ){
  grid_start <- grep( "^### Grid", data )
  grid_header <- 10
  grid_end <- grep( "^### Particles", data ) - 1
  cols_to_read <- c( "numeric", "numeric", "NULL", "NULL", "numeric", "NULL", "NULL", "NULL" )
  col_names=c("nx", "ny", NA, NA, "phi", NA, NA, NA )
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
    xlim <- c( min(x), max(x) )
    xticks <- pretty( xlim )
    xtickslabels <- format( xticks, nsmall=1 )
    ylim <- c( min(y), max(y) )
    yticks <- pretty( ylim )
    ytickslabels <- format( yticks, nsmall=1 )
    filled.contour(x, y, z,
                   color = topo.colors,
                   xlim <- xlim,
                   ylim <- ylim,
                   plot.axes = { axis( 1, las = 1, lwd.ticks=2, 
                                       at = xticks, labels = xtickslabels )
                                 axis( 2, las = 1, lwd.ticks=2, 
                                       at = yticks, labels = ytickslabels ) },
                   plot.title = title(main="Potential", xlab = "X", ylab = "Y" ),
                   )
    dev.off()
}


extract_data_for_density_plot <- function( data ){
  grid_start <- grep( "^### Grid", data )
  grid_header <- 10
  grid_end <- grep( "^### Particles", data ) - 1
  cols_to_read <- c( "numeric", "numeric", "NULL", "numeric", "NULL", "NULL", "NULL", "NULL" )
  col_names=c("nx", "ny", NA, "charge_density", NA, NA, NA, NA )
  density_data <- read.table(
                      textConnection( data[ (grid_start+grid_header) : grid_end ] ),
                      colClasses = cols_to_read, 
                      col.names = col_names )
  return( density_data )
}

plot_density <- function( dataframe_to_plot, domain_properties, filename ) {
    x <- unique( dataframe_to_plot$x )
    y <- unique( dataframe_to_plot$y )
    z <- xtabs(
          dataframe_to_plot$charge_density ~ dataframe_to_plot$x + dataframe_to_plot$y,
          dataframe_to_plot )
    as.data.frame.matrix( z )
###
    png( filename )
    xlim <- c( min(x), max(x) )
    xticks <- pretty( xlim )
    xtickslabels <- format( xticks, nsmall=1 )
    ylim <- c( min(y), max(y) )
    yticks <- pretty( ylim )
    ytickslabels <- format( yticks, nsmall=1 )
    filled.contour(x, y, z,
                   color = topo.colors,
                   xlim <- xlim,
                   ylim <- ylim,
                   plot.axes = { axis( 1, las = 1, lwd.ticks=2, 
                                       at = xticks, labels = xtickslabels )
                                 axis( 2, las = 1, lwd.ticks=2, 
                                       at = yticks, labels = ytickslabels ) },
                   plot.title = title(main="Charge density", xlab = "X", ylab = "Y" ),
                   )
    dev.off()
}


extract_data_for_particles_coords_plot <- function( data ){
  particles_start <- grep( "^Source name", data )
  particles_end <- c( particles_start[-1]-1, length(data) )
  particles_header <- 3
  cols_to_read <- c( "NULL", "NULL", "NULL",
                    "numeric", "numeric", "numeric",
                    "numeric", "numeric", "numeric" )
  col_names <- c(NA, NA, NA, "x", "y", "z", "px", "py", "pz" )
  particles_data <- list()
  for ( i in seq_along( particles_start ) ) {
    particles_data[[i]] <-
      read.table(
        textConnection(
          data[ ( particles_start[i] + particles_header ) : particles_end[i] ] ),
        colClasses = cols_to_read, 
        col.names = col_names )
  }
  return( particles_data )
}

plot_particles_coords <- function( particles_data, domain_properties, outfile ) {
    png( outfile )
    xmin <- 0
    xmax <- domain_properties$grid$x_volume_size
    xlim <- c( xmin, xmax )
    xticks <- pretty( xlim )
    xtickslabels <- format( xticks, nsmall=1 )
    ymin <- 0
    ymax <- domain_properties$grid$y_volume_size  
    ylim <- c( ymin, ymax )    
    yticks <- pretty( ylim )
    ytickslabels <- format( yticks, nsmall=1 )
    zmin <- 0
    zmax <- domain_properties$grid$z_volume_size  
    zlim <- c( zmin, zmax )    
    zticks <- pretty( zlim )
    ztickslabels <- format( zticks, nsmall=1 )

    s3d <- scatterplot3d(NA, NA, NA,
                         angle=30,
                         xlim=xlim, ylim=ylim, zlim=zlim,
                         xlab = "X", ylab = "Y", zlab = "Z",
                         main = "Particles" )

    for ( i in seq_along( particles_data ) ) {
        s3d$points3d(particles_data[[i]]$x,
                     particles_data[[i]]$y,
                     particles_data[[i]]$z,
                     pch = i, col = i )
    }

    ## plot(NA, NA,
    ##      xaxs="i", yaxs="i", axes = F,
    ##      xlim = xlim,
    ##      ylim = ylim,
    ##      ##         las = 1,
    ##      ##         cex.lab=2.5, cex.axis=1.5,
    ##      main = "Particles", 
    ##      xlab = "X", ylab = "Y"
    ##      )
    
    ## for ( i in seq_along( particles_data ) ) {        
    ##     points(particles_data[[i]]$x, particles_data[[i]]$y,
    ##            pch = i, col = i )      
    ##   mean_p <- mean( sqrt( particles_data[[i]]$px^2 + particles_data[[i]]$py^2 ) )
    ## max_p <- max( sqrt( particles_data[[i]]$px^2 + particles_data[[i]]$py^2 ) )
    ##      arrows(particles_data[[i]]$x,
    ##             particles_data[[i]]$y,
    ##             particles_data[[i]]$x + particles_data[[i]]$px/mean_p,
    ##             particles_data[[i]]$y + particles_data[[i]]$py/mean_p,
    ##             length = 0.05, angle = 20, 
    ##             col = i )
    ## }

    ## box()
    ## axis( 1, las = 1, lwd.ticks=2, at = xticks, labels = xtickslabels )
    ## axis( 2, las = 1, lwd.ticks=2, at = yticks, labels = ytickslabels )         

    ## legentries <- domain_properties$particles$source_name
    ## legpch <- seq_along( legentries )
    ## legend(x="topright",
    ##        legend = legentries,
    ##        bty = "n",
    ##        pch = legpch,
    ##        seg.len = 2.7,
    ##        col = seq_along( legentries ) ,
    ##        cex = 1.5)
    
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
      data_density <- grid_nodes_to_real_coords( data_density, domain_properties )
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
