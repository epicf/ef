import FreeCAD, FreeCADGui, Part
from FreeCAD import Base
from pivy import coin

import PySide
from PySide import QtGui, QtCore
from PySide.QtGui import *
from PySide.QtCore import *

class My_Command_Class():
    """My new command"""
 
    def GetResources(self):
        return {'Pixmap'  : 'My_Command_Icon',
                # the name of a svg file available in the resources
                'Accel' : "Shift+S", # a default shortcut (optional)
                'MenuText': "My New Command",
                'ToolTip' : "What my new command does"}
 
    def Activated(self):
        "Do something here"
        FreeCAD.Console.PrintMessage( "Can't touch me\n" )
        return
 
    def IsActive(self):
        """Here you can define if the command must 
        be active or not (greyed) if certain conditions
        are met or not. This function is optional."""
        return True



class CreateEpicfConfig():
    """Create new epicf config"""
 
    def GetResources(self):
        return {'Pixmap'  : 'My_Command_Icon',
                # the name of a svg file available in the resources
                'Accel' : "Shift+N", # a default shortcut (optional)
                'MenuText': "New epicf config",
                'ToolTip' : "Create New epicf config"}
 
    def Activated(self):
        self.epicf_conf_group = FreeCAD.ActiveDocument.addObject(
            "App::DocumentObjectGroup", "Epicf conf" )

        self.time_grid_conf = self.epicf_conf_group.newObject(
            "App::FeaturePython", "Time grid" )
        TimeGridConfigPart( self.time_grid_conf )
                
        self.outfile_conf = self.epicf_conf_group.newObject(
            "App::FeaturePython", "Output filename")
        OutputFilenameConfigPart( self.outfile_conf )
        
        self.spat_mesh_conf = self.epicf_conf_group.newObject(
            "App::FeaturePython", "Spatial mesh" )
        # self.spat_mesh_conf = self.epicf_conf_group.newObject(
        #     "Part::Feature", "Spatial mesh" )
        SpatialMeshConfigPart( self.spat_mesh_conf )
        
        self.boundary_cond_conf = self.epicf_conf_group.newObject(
            "App::FeaturePython", "Boundary conditions" )
        BoundaryConditionsConfigPart( self.boundary_cond_conf )

        self.magn_field_conf = self.epicf_conf_group.newObject(
            "App::FeaturePython", "Magnetic field" )
        MagneticFieldConfigPart( self.magn_field_conf )

        FreeCAD.ActiveDocument.recompute()
        
        FreeCAD.Console.PrintMessage( "Output filename section\n" )
        return
 
    def IsActive(self):
        """Here you can define if the command must be 
        active or not (greyed) if certain conditions
        are met or not. This function is optional."""
        return True


class AddSourceRegion():
    """Add source of particles"""
 
    def GetResources(self):
        return {'Pixmap'  : 'My_Command_Icon',
                # the name of a svg file available in the resources
                'Accel' : "Shift+S", # a default shortcut (optional)
                'MenuText': "Add particle source",
                'ToolTip' : "Add particle source-descr"}
 
    def Activated(self):
        # todo: add to selected group
        self.epicf_conf_group = FreeCAD.ActiveDocument.getObject("Epicf_conf")

        self.source_conf = self.epicf_conf_group.newObject(
            "App::FeaturePython", "Source" )
        ParticleSourceConfigPart( self.source_conf )

        FreeCAD.ActiveDocument.recompute()
                        
        return
 
    def IsActive(self):
        """Here you can define if the command must be 
        active or not (greyed) if certain conditions
        are met or not. This function is optional."""
        return True


class AddInnerRegionBox():
    """Add box inner region"""
 
    def GetResources(self):
        return {'Pixmap'  : 'My_Command_Icon',
                # the name of a svg file available in the resources
                'Accel' : "Shift+R", # a default shortcut (optional)
                'MenuText': "Add box region",
                'ToolTip' : "Add box region source-descr"}
 
    def Activated(self):
        # todo: add to selected group
        self.epicf_conf_group = FreeCAD.ActiveDocument.getObject("Epicf_conf")

        self.inner_reg_conf = self.epicf_conf_group.newObject(
            "App::FeaturePython", "Inner_region_box" )
        InnerRegionBoxConfigPart( self.inner_reg_conf )

        FreeCAD.ActiveDocument.recompute()
                        
        return
 
    def IsActive(self):
        """Here you can define if the command must be 
        active or not (greyed) if certain conditions
        are met or not. This function is optional."""
        return True

    

class GenerateConfFile():
    """Generate .conf file suitable for epicf """
 
    def GetResources(self):
        return {'Pixmap'  : 'My_Command_Icon',
                # the name of a svg file available in the resources
                'Accel' : "Shift+G", # a default shortcut (optional)
                'MenuText': "Generate .conf file",
                'ToolTip' : "Gen-conf description"}
 
    def Activated(self):
        "Do something here"
                #FreeCAD.Console.PrintMessage( "Output filename section\n" )
                # iterate over group and export to config
        return


    def Activated(self):
        config_text = self.generate_config_text()
        self.write_config( config_text )
                
    def IsActive(self):
        """Here you can define if the command must 
        be active or not (greyed) if certain conditions
        are met or not. This function is optional."""
        # check group "Epicf_conf" is selected
        return True

    def generate_config_text(self):
        config_text = []
        config_text.append( "; Generated by FreeCAD module\n" )
        config_text.append( "\n" )
        
        conf_objects_in_grp = FreeCAD.ActiveDocument.getObject("Epicf_conf").OutList
        for conf_part in conf_objects_in_grp:
            config_text.extend( conf_part.Proxy.generate_config_part() )

        return config_text
    
    def write_config( self, config_text ):
        #default_dialog_path = FreeCAD.ConfigGet("UserAppData")
        #default_conf_name = "test.conf"
        conf_filename, filename_filter = QFileDialog.getSaveFileName(
            None, "Generate epicf config", "./test.conf", "*.conf" )
        if conf_filename == "":
            FreeCAD.Console.PrintMessage("Process aborted" + "\n")
        else:                                               
            FreeCAD.Console.PrintMessage("Registration of " + conf_filename + "\n")
            with open( conf_filename, 'w') as f:
                # here your code
                f.writelines( config_text )
                f.write("FreeCAD the best")        

                
class TimeGridConfigPart:
    """ """

    def __init__( self, obj ):
        # self.total_time = None
        # self.time_step_size = None
        # self.time_save_step = None
        obj.addProperty(
            "App::PropertyString", "total_time",
            "Time grid", "Total simulation time" ).total_time = "10"
        obj.addProperty(
            "App::PropertyString", "time_step_size",
            "Time grid", "Time step" ).time_step_size = "1e-3"
        obj.addProperty(
            "App::PropertyString", "time_save_step",
            "Time grid", "Time step between checkpoints" ).time_save_step = "1e-5"
        obj.Proxy = self
        obj.ViewObject.Proxy = self
        self.doc_object = obj
        self.view_object = obj.ViewObject
            
    def execute(self, fp):
        ''' Print a short message when doing a recomputation, this method is mandatory '''
        # self.total_time = fp.total_time
        # self.time_step_size = fp.time_step_size
        # self.time_save_step = fp.time_save_step
        FreeCAD.Console.PrintMessage("Recompute TimeGridConfPart feature\n")

    def updateData(self, fp, prop):
        "If a property of the handled feature has changed we have the chance to handle this here"
        # fp is the handled feature, prop is the name of the property that has changed
        #setattr(self, prop, fp.getPropertyByName( prop ) )

    def attach(self, obj):
        ''' Setup the scene sub-graph of the view provider, this method is mandatory '''
        FreeCAD.Console.PrintMessage("Attach TimeGridConfPart\n")
        return

    def generate_config_part( self ):
        # todo: rewrite with less repeats
        conf_part = []
        conf_part.append( "[Time grid]\n" )
        # conf_part.append( "total_time = {}\n".format( self.total_time ) )
        # conf_part.append( "time_step_size = {}\n".format( self.time_step_size ) )
        # conf_part.append( "time_save_step = {}\n".format( self.time_save_step ) )
        conf_part.append( "total_time = {}\n".format( self.doc_object.getPropertyByName( "total_time" ) ) )
        conf_part.append( "time_step_size = {}\n".format( self.doc_object.getPropertyByName( "time_step_size" ) ) )
        conf_part.append( "time_save_step = {}\n".format( self.doc_object.getPropertyByName( "time_save_step" ) ) )
        conf_part.append("\n")
        return conf_part
    


class OutputFilenameConfigPart():
    """ """
    
    def __init__( self, obj ):
        obj.addProperty(
            "App::PropertyString", "output_filename_suffix",
            "Output filename", "Output filename extension").output_filename_suffix = ".h5"
        obj.addProperty(
            "App::PropertyString", "output_filename_prefix",
            "Output filename", "Output filename basename").output_filename_prefix = "out_"
        obj.Proxy = self
        obj.ViewObject.Proxy = self
        self.doc_object = obj
        self.view_object = obj.ViewObject

    def execute(self, fp):
        ''' Print a short message when doing a recomputation, this method is mandatory '''
        FreeCAD.Console.PrintMessage("Recompute OutputFilenameConfPart feature\n")

    def attach(self, obj):
        ''' Setup the scene sub-graph of the view provider, this method is mandatory '''
        FreeCAD.Console.PrintMessage("Attach OutputFilenameConfPart\n")
        return

    def updateData(self, fp, prop):
        "If a property of the handled feature has changed we have the chance to handle this here"
        # fp is the handled feature, prop is the name of the property that has changed
        #setattr(self, prop, fp.getPropertyByName( prop ) )
    
    def generate_config_part( self ):
        conf_part = []
        conf_part.append( "[Output filename]\n" )
        export_property_names = [ "output_filename_suffix", "output_filename_prefix" ]
        for x in export_property_names:
            conf_part.append( "{0} = {1}\n".format( x, self.doc_object.getPropertyByName( x ) ) )
        conf_part.append("\n")
        return conf_part


class SpatialMeshConfigPart():
    """ """

    def __init__( self, obj ):
        obj.addProperty(
            "App::PropertyString", "grid_x_size",
            "Spatial mesh", "x" ).grid_x_size = "1.0"
        obj.addProperty(
            "App::PropertyString", "grid_x_step",
            "Spatial mesh", "x-step" ).grid_x_step = "0.1"
        obj.addProperty(
            "App::PropertyString", "grid_y_size",
            "Spatial mesh", "y" ).grid_y_size = "1.0"
        obj.addProperty(
            "App::PropertyString", "grid_y_step",
            "Spatial mesh", "y-step" ).grid_y_step = "0.1"
        obj.addProperty(
            "App::PropertyString", "grid_z_size",
            "Spatial mesh", "z" ).grid_z_size = "1.0"
        obj.addProperty(
            "App::PropertyString", "grid_z_step",
            "Spatial mesh", "z-step" ).grid_z_step = "0.1"
        obj.addProperty("Part::PropertyPartShape","Shape", "Spatial mesh", "Volume box")
        obj.ViewObject.addProperty("App::PropertyColor", "Color",
                                   "Spatial mesh", "Volume box color").Color=(1.0,0.0,0.0)
        obj.Proxy = self
        obj.ViewObject.Proxy = self
        self.doc_object = obj
        self.view_object = obj.ViewObject

    def execute(self, fp):
        ''' Print a short message when doing a recomputation, this method is mandatory '''
        return 
                
    def attach(self, obj):
        ''' Setup the scene sub-graph of the view provider, this method is mandatory '''
        self.shaded = coin.SoGroup()
        self.wireframe = coin.SoGroup()
        self.color = coin.SoBaseColor()
        self.trans = coin.SoTranslation()
        self.box = coin.SoCube()

        self.shaded.addChild( self.color )
        self.shaded.addChild( self.trans )
        self.shaded.addChild( self.box )
        obj.addDisplayMode( self.shaded, "Shaded" );

        style = coin.SoDrawStyle()
        style.style = coin.SoDrawStyle.LINES
        self.wireframe.addChild( style )
        self.wireframe.addChild( self.color )
        self.wireframe.addChild( self.trans )
        self.wireframe.addChild( self.box )
        obj.addDisplayMode( self.wireframe, "Wireframe" );
        self.onChanged( obj, "Color" )
        return

    def updateData(self, fp, prop ):
        "Executed when propery in field 'data' is changed"
        # fp is the handled feature, prop is the name of the property that has changed
        FreeCAD.Console.PrintMessage("updateData: " + str(prop) + "\n")
        X = float( fp.getPropertyByName("grid_x_size") )
        Y = float( fp.getPropertyByName("grid_y_size") )
        Z = float( fp.getPropertyByName("grid_z_size") )
        #self.scale.scaleFactor.setValue( X, Y, Z )
        FreeCAD.Console.PrintMessage("Trans: " + str(X) + str(Y) + str(Z) + "\n")
        self.trans.translation.setValue( [ X/2, Y/2, Z/2 ] )
        self.box.width.setValue( X )
        self.box.height.setValue( Y )
        self.box.depth.setValue( Z )
    
    def getDisplayModes(self,obj):
        "Return a list of display modes."
        modes=[]
        modes.append("Shaded")
        modes.append("Wireframe")
        return modes
 
    def getDefaultDisplayMode(self):
        "Return the name of the default display mode. It must be defined in getDisplayModes."
        return "Wireframe"

    def setDisplayMode(self,mode):
        return mode

    def onChanged(self, vp, prop):
        "Executed if any property is changed"
        FreeCAD.Console.PrintMessage("Change property: " + str(prop) + "\n")
        if prop == "Color":
            c = vp.getPropertyByName("Color")
            self.color.rgb.setValue( c[0],c[1],c[2] )
    
    def __getstate__(self):
        return None
 
    def __setstate__(self, state):
        return None
    
    def generate_config_part( self ):
        conf_part = []
        conf_part.append( "[Spatial mesh]\n" )
        export_property_names = [ "grid_x_size", "grid_x_step",
                                  "grid_y_size", "grid_y_step",
                                  "grid_z_size", "grid_z_step" ]
        for x in export_property_names:
            conf_part.append( "{0} = {1}\n".format( x, self.doc_object.getPropertyByName( x ) ) )
        conf_part.append("\n")
        return conf_part


class BoundaryConditionsConfigPart():
    """ """
    
    def __init__( self, obj ):
        obj.addProperty(
            "App::PropertyString",
            "boundary_phi_left",
            "Boundary conditions",
            "Potential boundary conditions").boundary_phi_left = "0.0"
        obj.addProperty(
            "App::PropertyString",
            "boundary_phi_right",
            "Boundary conditions",
            "Potential boundary conditions").boundary_phi_right = "0.0"
        obj.addProperty(
            "App::PropertyString",
            "boundary_phi_top",
            "Boundary conditions",
            "Potential boundary conditions").boundary_phi_top = "0.0"
        obj.addProperty(
            "App::PropertyString",
            "boundary_phi_bottom",
            "Boundary conditions",
            "Potential boundary conditions").boundary_phi_bottom = "0.0"
        obj.addProperty(
            "App::PropertyString",
            "boundary_phi_near",
            "Boundary conditions",
            "Potential boundary conditions").boundary_phi_near = "0.0"
        obj.addProperty(
            "App::PropertyString",
            "boundary_phi_far",
            "Boundary conditions",
            "Potential boundary conditions").boundary_phi_far = "0.0"
        obj.Proxy = self
        obj.ViewObject.Proxy = self
        self.doc_object = obj
        self.view_object = obj.ViewObject

    def execute(self, fp):
        ''' Print a short message when doing a recomputation, this method is mandatory '''
        FreeCAD.Console.PrintMessage("Recompute BoundaryConfPart feature\n")

    def attach(self, obj):
        ''' Setup the scene sub-graph of the view provider, this method is mandatory '''
        FreeCAD.Console.PrintMessage("Attach BoundaryConfPart\n")
        return

    def updateData(self, fp, prop):
        "If a property of the handled feature has changed we have the chance to handle this here"
        # fp is the handled feature, prop is the name of the property that has changed
        #setattr(self, prop, fp.getPropertyByName( prop ) )
        
    def generate_config_part( self ):
        conf_part = []
        conf_part.append( "[Boundary conditions]\n" )
        export_property_names = [ "boundary_phi_left", "boundary_phi_right",
                                  "boundary_phi_top" , "boundary_phi_bottom",
                                  "boundary_phi_near", "boundary_phi_far" ]
        for x in export_property_names:
            conf_part.append(
                "{0} = {1}\n".format( x, self.doc_object.getPropertyByName( x ) ) )
        conf_part.append("\n")
        return conf_part


class MagneticFieldConfigPart():
    """ """
    
    def __init__( self, obj ):
        obj.addProperty(
            "App::PropertyString", "magnetic_field_x",
            "External magnetic field", "Field magnitude along X axis").magnetic_field_x = "0.0"
        obj.addProperty(
            "App::PropertyString", "magnetic_field_y",
            "External magnetic field", "Field magnitude along Y axis").magnetic_field_y = "0.0"
        obj.addProperty(
            "App::PropertyString", "magnetic_field_z",
            "External magnetic field", "Field magnitude along Z axis").magnetic_field_z = "0.0"
        obj.addProperty(
            "App::PropertyString", "speed_of_light",
            "External magnetic field", "Speed of light").speed_of_light = "3.0e10"
        obj.Proxy = self
        obj.ViewObject.Proxy = self
        self.doc_object = obj
        self.view_object = obj.ViewObject

    def execute(self, fp):
        ''' Print a short message when doing a recomputation, this method is mandatory '''
        FreeCAD.Console.PrintMessage("Recompute OutputFilenameConfPart feature\n")

    def attach(self, obj):
        ''' Setup the scene sub-graph of the view provider, this method is mandatory '''
        FreeCAD.Console.PrintMessage("Attach OutputFilenameConfPart\n")
        return

    def updateData(self, fp, prop):
        "If a property of the handled feature has changed we have the chance to handle this here"
        # fp is the handled feature, prop is the name of the property that has changed
        #setattr(self, prop, fp.getPropertyByName( prop ) )
    
    def generate_config_part( self ):
        conf_part = []
        conf_part.append( "[External magnetic field]\n" )
        export_property_names = [ "magnetic_field_x", "magnetic_field_y",
                                  "magnetic_field_z", "speed_of_light" ]
        for x in export_property_names:
            conf_part.append( "{0} = {1}\n".format( x, self.doc_object.getPropertyByName( x ) ) )
        conf_part.append("\n")
        return conf_part


class ParticleSourceConfigPart():
    """Particle source region"""

    def __init__( self, obj ):
        obj.addProperty(
            "App::PropertyString", "particle_source_initial_number_of_particles",
            "Number of particles", "Number of particles at start" ).particle_source_initial_number_of_particles = "0"
        obj.addProperty(
            "App::PropertyString", "particle_source_particles_to_generate_each_step",
            "Number of particles", "zzz" ).particle_source_particles_to_generate_each_step = "0"
        obj.addProperty(
            "App::PropertyString", "particle_source_x_left",
            "Position", "zzz" ).particle_source_x_left = "0.4"
        obj.addProperty(
            "App::PropertyString", "particle_source_x_right",
            "Position", "zzz" ).particle_source_x_right = "0.6"
        obj.addProperty(
            "App::PropertyString", "particle_source_y_bottom",
            "Position", "zzz" ).particle_source_y_bottom = "0.4"
        obj.addProperty(
            "App::PropertyString", "particle_source_y_top",
            "Position", "zzz" ).particle_source_y_top = "0.6"
        obj.addProperty(
            "App::PropertyString", "particle_source_z_near",
            "Position", "zzz" ).particle_source_z_near = "0.4"
        obj.addProperty(
            "App::PropertyString", "particle_source_z_far",
            "Position", "zzz" ).particle_source_z_far = "0.6"
        obj.addProperty(
            "App::PropertyString", "particle_source_mean_momentum_x",
            "Momentum", "zzz" ).particle_source_mean_momentum_x = "1.0"
        obj.addProperty(
            "App::PropertyString", "particle_source_mean_momentum_y",
            "Momentum", "zzz" ).particle_source_mean_momentum_y = "1.0"
        obj.addProperty(
            "App::PropertyString", "particle_source_mean_momentum_z",
            "Momentum", "zzz" ).particle_source_mean_momentum_z = "1.0"
        obj.addProperty(
            "App::PropertyString", "particle_source_temperature",
            "Particle properties", "zzz" ).particle_source_temperature = "1.0"
        obj.addProperty(
            "App::PropertyString", "particle_source_charge",
            "Particle properties", "zzz" ).particle_source_charge = "1.0"
        obj.addProperty(
            "App::PropertyString", "particle_source_mass",
            "Particle properties", "zzz" ).particle_source_mass = "1.0"
        obj.ViewObject.addProperty("App::PropertyColor", "Color",
                                   "Spatial mesh", "Volume box color").Color=(0.0, 0.0, 1.0)
        
        obj.Proxy = self
        obj.ViewObject.Proxy = self
        self.doc_object = obj
        self.view_object = obj.ViewObject

    def execute(self, fp):
        ''' Print a short message when doing a recomputation, this method is mandatory '''
        FreeCAD.Console.PrintMessage("Recompute SpatMeshConfPart feature\n")

    def attach(self, obj):
        ''' Setup the scene sub-graph of the view provider, this method is mandatory '''
        self.shaded = coin.SoGroup()
        self.wireframe = coin.SoGroup()
        self.trans = coin.SoTranslation()
        self.color = coin.SoBaseColor()

        self.box = coin.SoCube()

        self.shaded.addChild( self.color )
        self.shaded.addChild( self.trans )
        self.shaded.addChild( self.box )
        obj.addDisplayMode( self.shaded, "Shaded" )

        style = coin.SoDrawStyle()
        style.style = coin.SoDrawStyle.LINES
        self.wireframe.addChild( style )
        self.wireframe.addChild( self.color )                
        self.wireframe.addChild( self.trans )
        self.wireframe.addChild( self.box )
        obj.addDisplayMode( self.wireframe, "Wireframe" )
        
        self.onChanged( obj, "Color" )
        return

    def updateData(self, fp, prop ):
        "Executed when propery in field 'data' is changed"
        # fp is the handled feature, prop is the name of the property that has changed
        FreeCAD.Console.PrintMessage("updateData: " + str(prop) + "\n")        
        x0 = float( fp.getPropertyByName("particle_source_x_left") )
        y0 = float( fp.getPropertyByName("particle_source_y_bottom") )
        z0 = float( fp.getPropertyByName("particle_source_z_near") )
        xlen = float( fp.getPropertyByName("particle_source_x_right") ) - x0
        ylen = float( fp.getPropertyByName("particle_source_y_top") ) - y0
        zlen = float( fp.getPropertyByName("particle_source_z_far") ) - z0
        self.trans.translation.setValue( [ x0 + xlen / 2,
                                           y0 + ylen / 2,
                                           z0 + zlen / 2 ] )
        self.box.width.setValue( xlen )
        self.box.height.setValue( ylen )
        self.box.depth.setValue( zlen )
        

    
    def getDisplayModes(self,obj):
        "Return a list of display modes."
        modes=[]
        modes.append("Shaded")
        modes.append("Wireframe")
        return modes
 
    def getDefaultDisplayMode(self):
        "Return the name of the default display mode. It must be defined in getDisplayModes."
        return "Wireframe"

    def setDisplayMode(self,mode):
        return mode

    def onChanged(self, vp, prop):
        "Executed if any property is changed"
        FreeCAD.Console.PrintMessage("Change property: " + str(prop) + "\n")
        if prop == "Color":
            c = vp.getPropertyByName("Color")
            self.color.rgb.setValue( c[0],c[1],c[2] )
    
    def __getstate__(self):
        return None
 
    def __setstate__(self, state):
        return None

    
    def generate_config_part( self ):
        conf_part = []
        conf_part.append( "[Source.{0}]\n".format( self.doc_object.getPropertyByName( "Label" ) ) )
        export_property_names = [ "particle_source_initial_number_of_particles",
                                  "particle_source_particles_to_generate_each_step",
                                  "particle_source_x_left", "particle_source_x_right",
                                  "particle_source_y_bottom", "particle_source_y_top",
                                  "particle_source_z_near", "particle_source_z_far",
                                  "particle_source_mean_momentum_x",
                                  "particle_source_mean_momentum_y",
                                  "particle_source_mean_momentum_z",
                                  "particle_source_temperature",
                                  "particle_source_charge",
                                  "particle_source_mass" ]
        for x in export_property_names:
            conf_part.append( "{0} = {1}\n".format( x, self.doc_object.getPropertyByName( x ) ) )
        conf_part.append("\n")
        return conf_part



class InnerRegionBoxConfigPart():
    """Box inner region"""

    def __init__( self, obj ):
        obj.addProperty(
            "App::PropertyString", "inner_region_box_x_left",
            "Position", "zzz" ).inner_region_box_x_left = "0.1"
        obj.addProperty(
            "App::PropertyString", "inner_region_box_x_right",
            "Position", "zzz" ).inner_region_box_x_right = "0.9"
        obj.addProperty(
            "App::PropertyString", "inner_region_box_y_bottom",
            "Position", "zzz" ).inner_region_box_y_bottom = "0.1"
        obj.addProperty(
            "App::PropertyString", "inner_region_box_y_top",
            "Position", "zzz" ).inner_region_box_y_top = "0.9"
        obj.addProperty(
            "App::PropertyString", "inner_region_box_z_near",
            "Position", "zzz" ).inner_region_box_z_near = "0.1"
        obj.addProperty(
            "App::PropertyString", "inner_region_box_z_far",
            "Position", "zzz" ).inner_region_box_z_far = "0.2"
        obj.addProperty(
            "App::PropertyString", "inner_region_box_potential",
            "Particle properties", "zzz" ).inner_region_box_potential = "0.0"
        obj.ViewObject.addProperty("App::PropertyColor", "Color",
                                   "Inner region color", "Inner region color").Color=(0.5, 0.5, 0.0)
        
        obj.Proxy = self
        obj.ViewObject.Proxy = self
        self.doc_object = obj
        self.view_object = obj.ViewObject

    def execute(self, fp):
        ''' Print a short message when doing a recomputation, this method is mandatory '''
        FreeCAD.Console.PrintMessage("Recompute SpatMeshConfPart feature\n")

    def attach(self, obj):
        ''' Setup the scene sub-graph of the view provider, this method is mandatory '''

        self.shaded = coin.SoGroup()
        self.wireframe = coin.SoGroup()
        self.trans = coin.SoTranslation()
        self.color = coin.SoBaseColor()

        self.box = coin.SoCube()

        self.shaded.addChild( self.color )
        self.shaded.addChild( self.trans )
        self.shaded.addChild( self.box )
        obj.addDisplayMode( self.shaded, "Shaded" )

        style = coin.SoDrawStyle()
        style.style = coin.SoDrawStyle.LINES
        self.wireframe.addChild( style )
        self.wireframe.addChild( self.color )
        self.wireframe.addChild( self.trans )
        self.wireframe.addChild( self.box )
        obj.addDisplayMode( self.wireframe, "Wireframe" )
        
        self.onChanged( obj, "Color" )
        return

    def updateData(self, fp, prop ):
        "Executed when propery in field 'data' is changed"
        # fp is the handled feature, prop is the name of the property that has changed
        FreeCAD.Console.PrintMessage("updateData: " + str(prop) + "\n")        
        x0 = float( fp.getPropertyByName("inner_region_box_x_left") )
        y0 = float( fp.getPropertyByName("inner_region_box_y_bottom") )
        z0 = float( fp.getPropertyByName("inner_region_box_z_near") )
        xlen = float( fp.getPropertyByName("inner_region_box_x_right") ) - x0
        ylen = float( fp.getPropertyByName("inner_region_box_y_top") ) - y0
        zlen = float( fp.getPropertyByName("inner_region_box_z_far") ) - z0
        self.trans.translation.setValue( [ x0 + xlen / 2,
                                           y0 + ylen / 2,
                                           z0 + zlen / 2 ] )
        self.box.width.setValue( xlen )
        self.box.height.setValue( ylen )
        self.box.depth.setValue( zlen )
    
    def getDisplayModes(self,obj):
        "Return a list of display modes."
        modes=[]
        modes.append("Shaded")
        modes.append("Wireframe")
        return modes
 
    def getDefaultDisplayMode(self):
        "Return the name of the default display mode. It must be defined in getDisplayModes."
        return "Shaded"

    def setDisplayMode(self,mode):
        return mode

    def onChanged(self, vp, prop):
        "Executed if any property is changed"
        FreeCAD.Console.PrintMessage("Change property: " + str(prop) + "\n")
        if prop == "Color":
            c = vp.getPropertyByName("Color")
            self.color.rgb.setValue( c[0],c[1],c[2] )
    
    def __getstate__(self):
        return None
 
    def __setstate__(self, state):
        return None

    
    def generate_config_part( self ):
        conf_part = []
        conf_part.append( "[Inner_region_box.{0}]\n".format( self.doc_object.getPropertyByName( "Label" ) ) )
        export_property_names = [ "inner_region_box_x_left", "inner_region_box_x_right",
                                  "inner_region_box_y_bottom", "inner_region_box_y_top",
                                  "inner_region_box_z_near", "inner_region_box_z_far",
                                  "inner_region_box_potential" ]
        for x in export_property_names:
            conf_part.append( "{0} = {1}\n".format( x, self.doc_object.getPropertyByName( x ) ) )
        conf_part.append("\n")
        return conf_part


    
FreeCADGui.addCommand( 'My_Command', My_Command_Class() )
FreeCADGui.addCommand( 'CreateEpicfConfig', CreateEpicfConfig() )
FreeCADGui.addCommand( 'AddSourceRegion', AddSourceRegion() )
FreeCADGui.addCommand( 'AddInnerRegionBox', AddInnerRegionBox() )
FreeCADGui.addCommand( 'GenerateConfFile', GenerateConfFile() )
