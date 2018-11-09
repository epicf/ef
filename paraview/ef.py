from paraview.simple import *
import h5py
from PySide import QtGui, QtCore

def main():
    h5file_name = get_filename()
    if h5file_name == "":
        print( "Failed to load file: file was not selected \n" )
    else:
        # Computation volume
        extract_boundary_box_properties_from_out_file( h5file_name )
        show_computation_volume()

        # Inner regions (currently only box-shaped are processed)
        h5file = h5py.File( h5file_name, driver="core", mode="r" )
        inner_reg_group = h5file["/Inner_regions/"]
        for inner_reg_name in inner_reg_group:
            region = create_inner_region( inner_reg_name, inner_reg_group )
            if region:
                region_repr = inner_region_repr( region )
        h5file.close()

        # Spatial mesh 
        spat_mesh = read_spat_mesh_data_as_table( h5file_name )
        spat_mesh_struct_grid = spat_mesh_table_to_struct_grid( spat_mesh )
        spat_mesh_struct_grid_rotated = rotate_spat_mesh_struct_grid( spat_mesh_struct_grid )

        # Particles
        h5file = h5py.File( h5file_name, driver="core", mode="r" )
        source_group = h5file["/Particle_sources/"]
        for source_name in source_group:
            particles_table = read_particles_as_table(
                source_name, h5file_name )
            particles_table_to_points = convert_particles_table_to_points(
                source_name, particles_table )
            particles_glyph = particles_glyph_representation(
                source_name, particles_table_to_points )
        h5file.close()

        
def get_filename():    
    default_dialog_path = "./"
    h5file_name, filename_filter = QtGui.QFileDialog.getOpenFileName(
        None, ".h5 file", default_dialog_path, "*.h5" )
    return( h5file_name )

def extract_boundary_box_properties_from_out_file( outfile_name ):
    outfile = h5py.File( outfile_name, driver="core", mode="r" )
    global x_cell_size, x_n_nodes, x_volume_size
    global y_cell_size, y_n_nodes, y_volume_size
    global z_cell_size, z_n_nodes, z_volume_size
    x_cell_size = outfile["/Spatial_mesh"].attrs["x_cell_size"][0]
    y_cell_size = outfile["/Spatial_mesh"].attrs["y_cell_size"][0]
    z_cell_size = outfile["/Spatial_mesh"].attrs["z_cell_size"][0]
    x_n_nodes = outfile["/Spatial_mesh"].attrs["x_n_nodes"][0]
    y_n_nodes = outfile["/Spatial_mesh"].attrs["y_n_nodes"][0]
    z_n_nodes = outfile["/Spatial_mesh"].attrs["z_n_nodes"][0]
    x_volume_size = outfile["/Spatial_mesh"].attrs["x_volume_size"][0]
    y_volume_size = outfile["/Spatial_mesh"].attrs["y_volume_size"][0]
    z_volume_size = outfile["/Spatial_mesh"].attrs["z_volume_size"][0]
    outfile.close()

def show_computation_volume():
    computation_volume = Box()
    computation_volume.XLength = x_volume_size
    computation_volume.YLength = y_volume_size
    computation_volume.ZLength = z_volume_size
    computation_volume.Center = [ x_volume_size / 2,
                                  y_volume_size / 2,
                                  z_volume_size / 2 ]
    computation_volume_repr = Show( computation_volume )
    computation_volume_repr.Visibility = 0
    computation_volume_repr.Representation = 'Wireframe'
    computation_volume_repr.AmbientColor = [1.0, 0.0, 0.0]
    computation_volume_repr.Visibility = 1
    RenameSource( "Computation Volume", computation_volume )


def create_inner_region( inner_region_name, inner_region_group ):
    reg = None
    if inner_region_group[inner_region_name].attrs["object_type"] == 'box':
        xleft, xright, ybottom, ytop, znear, zfar = extract_box_position(
            inner_region_name, inner_region_group )
        reg = Box()
        reg.XLength = xleft - xright
        reg.YLength = ytop - ybottom
        reg.ZLength = zfar - znear
        reg.Center = [ xright + reg.XLength / 2,
                       ybottom + reg.YLength / 2,
                       znear + reg.ZLength / 2 ]
        RenameSource( "inner_region_" + inner_region_name, reg )
    return reg

def inner_region_repr( region ):
    reg_repr = Show( region )
    reg_repr.Visibility = 0
    reg_repr.Representation = 'Surface'
    reg_repr.AmbientColor = [1.0, 1.0, 0.0]
    reg_repr.DiffuseColor = [1.0, 1.0, 0.0]
    reg_repr.Visibility = 1    
    return reg_repr    

def extract_box_position( inner_reg_name, inner_reg_group ):
    xleft = inner_reg_group[inner_reg_name].attrs["x_left"][0]
    xright = inner_reg_group[inner_reg_name].attrs["x_right"][0]
    ytop = inner_reg_group[inner_reg_name].attrs["y_top"][0]
    ybottom = inner_reg_group[inner_reg_name].attrs["y_bottom"][0]
    znear = inner_reg_group[inner_reg_name].attrs["z_near"][0]
    zfar = inner_reg_group[inner_reg_name].attrs["z_far"][0]
    return ( xleft, xright, ybottom, ytop, znear, zfar )


def read_spat_mesh_data_as_table( h5file_name ):
    spat_mesh = ProgrammableSource()
    spat_mesh.OutputDataSetType = 'vtkTable'
    script = gen_spat_mesh_script( h5file_name )
    spat_mesh.Script = script
    UpdatePipeline()
    RenameSource( "Spatial mesh", spat_mesh )    
    return spat_mesh


def spat_mesh_table_to_struct_grid( spat_mesh ):
    spat_mesh_struct_grid = TableToStructuredGrid( spat_mesh )
    # VTK_structured_grid expects a table with node points.
    # Nodes should be arranged in such a way, that points
    # with varying coord at dimenstion [0] come first and
    # fixed coordinates at dimensions [1] and [2] follow.
    # However in h5-file, nodes arranged in a way, that
    # nodes with varying coord at dim [2] come first 
    # and fixed coords at [1] and [0] follow.
    # Therefore spat_mesh_struct_grid.XColumn is set to Z
    # impossible to explain clearly. 
    spat_mesh_struct_grid.XColumn = 'Z'
    spat_mesh_struct_grid.YColumn = 'Y'
    spat_mesh_struct_grid.ZColumn = 'X'
    spat_mesh_struct_grid.WholeExtent = [0, z_n_nodes-1,
                                         0, y_n_nodes-1,
                                         0, x_n_nodes-1]
    #spat_mesh_struct_grid.SetUpdateExtentToWholeExtent()
    UpdatePipeline()
    RenameSource( "spat_mesh_grid_norotation", spat_mesh_struct_grid )
    return spat_mesh_struct_grid


def rotate_spat_mesh_struct_grid( spat_mesh_struct_grid ):
    spat_mesh_struct_grid_rotated = Transform( spat_mesh_struct_grid )
    spat_mesh_struct_grid_rotated.Transform.Translate = [x_volume_size, 0.0, 0.0]
    spat_mesh_struct_grid_rotated.Transform.Rotate = [0.0, 270.0, 0.0]
    spat_mesh_struct_grid_rotated_repr = Show( spat_mesh_struct_grid_rotated )
    spat_mesh_struct_grid_rotated_repr.Representation = 'Outline'
    RenameSource( "spat_mesh_grid", spat_mesh_struct_grid_rotated )
    return spat_mesh_struct_grid_rotated

def gen_spat_mesh_script( h5file_name ):
    spat_mesh_script = spat_mesh_script_template.format( filename = h5file_name )
    return spat_mesh_script

def read_particles_as_table( source_name, h5file_name ):
    particles = ProgrammableSource()
    particles.OutputDataSetType = 'vtkTable'
    # todo: move script code to separate file.
    # Use 
    # particles.Script =
    #  "execfile('/home/noway/progs/paraview/epicf_macros/particles_progr_filter.py')"
    # or 
    # particles.PythonPath =
    #  '"/home/noway/progs/paraview/epicf_macros/particles_progr_filter.py"'
    script = gen_source_script( source_name, h5file_name )
    particles.Script = script
    UpdatePipeline()
    RenameSource( source_name, particles )
    return particles

def gen_source_script( source_group_name, h5file_name ):
    particles_source_script = particles_source_script_template.format(
        filename = h5file_name,
        source_group_name = source_group_name )
    return particles_source_script


def convert_particles_table_to_points( source_name, particles_table ):
    particles_table_to_points = TableToPoints( particles_table )
    particles_table_to_points.XColumn = 'X'
    particles_table_to_points.YColumn = 'Y'
    particles_table_to_points.ZColumn = 'Z'
    UpdatePipeline()
    RenameSource( source_name + "_table", particles_table_to_points )
    return particles_table_to_points


def particles_glyph_representation( source_name, particles_table_to_points ):
    particles_glyph = Glyph( particles_table_to_points )
    particles_glyph.Scalars = ['POINTS', 'Pz']
    particles_glyph.GlyphType = "Sphere"
    particles_glyph.GlyphType.Radius = 0.005
    UpdatePipeline()
    particles_glyph_repr = Show( particles_glyph )
    particles_glyph_repr.ColorArrayName = ('POINT_DATA', 'Pz')
    #color_lookup_table = GetLookupTableForArray( "Pz", 1 )            
    RenameSource( source_name + "_particle_glyphs", particles_glyph )
    UpdatePipeline() 


spat_mesh_script_template = """
import h5py
import numpy as np

def extract_nodes_rho_potential_fields_from_out_file( outfile ):
    node_coords_x_hdf5 = outfile['/Spatial_mesh/node_coordinates_x']
    node_coords_y_hdf5 = outfile['/Spatial_mesh/node_coordinates_y']
    node_coords_z_hdf5 = outfile['/Spatial_mesh/node_coordinates_z']
    rho_hdf5 = outfile['/Spatial_mesh/charge_density']
    num_potential_hdf5 = outfile['/Spatial_mesh/potential']
    el_field_x_hdf5 = outfile['/Spatial_mesh/electric_field_x']
    el_field_y_hdf5 = outfile['/Spatial_mesh/electric_field_y']
    el_field_z_hdf5 = outfile['/Spatial_mesh/electric_field_z']
    node_coords_x = np.empty_like( node_coords_x_hdf5 )
    node_coords_y = np.empty_like( node_coords_y_hdf5 )
    node_coords_z = np.empty_like( node_coords_z_hdf5 )
    rho = np.empty_like( rho_hdf5 )
    num_potential = np.empty_like( num_potential_hdf5 )
    el_field_x = np.empty_like( el_field_x_hdf5 )
    el_field_y = np.empty_like( el_field_y_hdf5 )
    el_field_z = np.empty_like( el_field_z_hdf5 )
    node_coords_x_hdf5.read_direct( node_coords_x )
    node_coords_y_hdf5.read_direct( node_coords_y )
    node_coords_z_hdf5.read_direct( node_coords_z )
    rho_hdf5.read_direct( rho )
    num_potential_hdf5.read_direct( num_potential )
    el_field_x_hdf5.read_direct( el_field_x )
    el_field_y_hdf5.read_direct( el_field_y )
    el_field_z_hdf5.read_direct( el_field_z )
    return( node_coords_x, node_coords_y, node_coords_z, 
            rho, num_potential, 
            el_field_x, el_field_y, el_field_z )
    
h5file_name = "{filename}"
h5file = h5py.File( h5file_name, driver = "core", mode = "r" )
nc_x, nc_y, nc_z, rho, phi, el_f_x, el_f_y, el_f_z = extract_nodes_rho_potential_fields_from_out_file( h5file )

# todo: find out how to convert array to vtkArray
X = vtk.vtkDoubleArray()
X.SetName("X")
Y = vtk.vtkDoubleArray()
Y.SetName("Y")
Z = vtk.vtkDoubleArray()
Z.SetName("Z")
Rho = vtk.vtkDoubleArray()
Rho.SetName("Charge density")
Phi = vtk.vtkDoubleArray()
Phi.SetName("Potential")
Ex = vtk.vtkDoubleArray()
Ex.SetName("Ex")
Ey = vtk.vtkDoubleArray()
Ey.SetName("Ey")
Ez = vtk.vtkDoubleArray()
Ez.SetName("Ez")

for ( x, y, z, rho, phi, ex, ey, ez ) in zip( nc_x, nc_y, nc_z, rho, phi, el_f_x, el_f_y, el_f_z ):
    X.InsertNextTuple( [x] )
    Y.InsertNextTuple( [y] )
    Z.InsertNextTuple( [z] )
    Rho.InsertNextTuple( [rho] )
    Phi.InsertNextTuple( [phi] )
    Ex.InsertNextTuple( [ex] )
    Ey.InsertNextTuple( [ey] )
    Ez.InsertNextTuple( [ez] )

output = self.GetOutput()
output.AddColumn( X )
output.AddColumn( Y )
output.AddColumn( Z )
output.AddColumn( Rho )
output.AddColumn( Phi )
output.AddColumn( Ex )
output.AddColumn( Ey )
output.AddColumn( Ez )

h5file.close()
"""


    
particles_source_script_template = """
import h5py
import numpy as np

def read_particles( h5_file ):
    particles_h5_x = h5_file["/Particle_sources/{source_group_name}/position_x"]
    particles_h5_y = h5_file["/Particle_sources/{source_group_name}/position_y"]
    particles_h5_z = h5_file["/Particle_sources/{source_group_name}/position_z"]
    particles_h5_px = h5_file["/Particle_sources/{source_group_name}/momentum_x"]
    particles_h5_py = h5_file["/Particle_sources/{source_group_name}/momentum_y"]
    particles_h5_pz = h5_file["/Particle_sources/{source_group_name}/momentum_z"]
    particles_x = np.empty_like( particles_h5_x )
    particles_y = np.empty_like( particles_h5_y )
    particles_z = np.empty_like( particles_h5_z )
    particles_px = np.empty_like( particles_h5_px )
    particles_py = np.empty_like( particles_h5_py )
    particles_pz = np.empty_like( particles_h5_pz )
    particles_h5_x.read_direct( particles_x )
    particles_h5_y.read_direct( particles_y )
    particles_h5_z.read_direct( particles_z )
    particles_h5_px.read_direct( particles_px )
    particles_h5_py.read_direct( particles_py )
    particles_h5_pz.read_direct( particles_pz )
    mass = h5_file["/Particle_sources/{source_group_name}"].attrs["mass"][0]
    # todo: get id
    return (particles_x, particles_y, particles_z, 
            particles_px, particles_py, particles_pz,
            mass)

h5file = "{filename}"
h5 = h5py.File( h5file, driver = "core", mode = "r" )
particles_x, particles_y, particles_z, particles_px, particles_py, particles_pz, mass = read_particles( h5 )

# todo: find out how to convert array to vtkArray
id = vtk.vtkIntArray()
id.SetName("particle id")
X = vtk.vtkDoubleArray()
X.SetName("X")
Y = vtk.vtkDoubleArray()
Y.SetName("Y")
Z = vtk.vtkDoubleArray()
Z.SetName("Z")
Px = vtk.vtkDoubleArray()
Px.SetName("Px")
Py = vtk.vtkDoubleArray()
Py.SetName("Py")
Pz = vtk.vtkDoubleArray()
Pz.SetName("Pz")
#todo: id
#for (i,x,y,z,px,py,pz) in zip( particles_x, particles_y, particles_z, particles_px, particles_py, particles_pz ):
for (x,y,z,px,py,pz) in zip( particles_x, particles_y, particles_z, particles_px, particles_py, particles_pz ):
#    id.InsertNextTuple( [i] )
    X.InsertNextTuple( [x] )
    Y.InsertNextTuple( [y] )
    Z.InsertNextTuple( [z] )
    Px.InsertNextTuple( [px] )
    Py.InsertNextTuple( [py] )
    Pz.InsertNextTuple( [pz] )

#output = self.GetOutputDataObject(0)
output = self.GetOutput()
#output.AddColumn( id )
output.AddColumn( X )
output.AddColumn( Y )
output.AddColumn( Z )
output.AddColumn( Px )
output.AddColumn( Py )
output.AddColumn( Pz )

h5.close()
"""


main()
