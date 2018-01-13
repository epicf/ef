import os
import FreeCAD, FreeCADGui
import Part
from math import *
from pivy import coin
from PySide import QtGui, QtCore

import subprocess

class CreateEfConfig():
    """Create objects for new ef config"""
 
    def GetResources(self):
        moddir = os.path.expanduser("~") + "/.FreeCAD/Mod/ef"
        return {'Pixmap'  : moddir + '/icons/new_conf_template.svg',
                'Accel' : "Shift+N", # a default shortcut (optional)
                'MenuText': "New minimal ef config",
                'ToolTip' : "New minimal ef config"}
 
    def Activated(self):
        ef_conf_group = FreeCAD.ActiveDocument.addObject(
            "App::DocumentObjectGroup", "ef_conf" )

        time_grid_conf = ef_conf_group.newObject(
            "App::FeaturePython", "Time grid" )
        TimeGridConfigPart( time_grid_conf )
                
        outfile_conf = ef_conf_group.newObject(
            "App::FeaturePython", "Output filename")
        OutputFilenameConfigPart( outfile_conf )

        particle_interaction_conf = ef_conf_group.newObject(
            "App::FeaturePython", "Particle interaction model")
        ParticleInteractionModelConfigPart( particle_interaction_conf )
        
        spat_mesh_conf = ef_conf_group.newObject(
            "App::FeaturePython", "Spatial mesh" )
        SpatialMeshConfigPart( spat_mesh_conf )
        
        boundary_cond_conf = ef_conf_group.newObject(
            "App::FeaturePython", "Boundary conditions" )
        BoundaryConditionsConfigPart( boundary_cond_conf )

        magn_field_conf = ef_conf_group.newObject(
            "App::FeaturePython", "Magnetic field" )
        MagneticFieldConfigPart( magn_field_conf )

        run_ef = ef_conf_group.newObject(
            "App::FeaturePython", "Run_Ef" )
        RunEfConfig( run_ef )
        
        FreeCAD.ActiveDocument.recompute()
        return
 
    def IsActive(self):        
        return (FreeCAD.ActiveDocument is not None)
            

class AddSourceRegion():
    """Add box-shaped source of particles"""
 
    def GetResources(self):
        moddir = os.path.expanduser("~") + "/.FreeCAD/Mod/ef"
        return {'Pixmap'  : moddir + '/icons/add_box_source.svg',
                'Accel' : "Shift+S", # a default shortcut (optional)
                'MenuText': "Add box-shaped source of particles",
                'ToolTip' : "Add box-shaped source of particles"}
 
    def Activated(self):
        for ef_conf_group in self.selected_ef_conf_groups:
            source_conf = ef_conf_group.newObject(
                "App::FeaturePython", "Source" )
            ParticleSourceConfigPart( source_conf )
        FreeCAD.ActiveDocument.recompute()                        
        return
 
    def IsActive(self):
        # Add source only if ef-group is selected
        # todo: check if selected object is ef-conf group
        # or directly belongs to ef-conf group
        sel = FreeCADGui.Selection.getSelection()
        self.selected_ef_conf_groups = []
        active = False
        for obj in sel:
            if "ef" in obj.Name:
                self.selected_ef_conf_groups.append( obj )
                active = True
            else:
                for parent_obj in obj.InList:
                    if "ef" in parent_obj.Name:
                        self.selected_ef_conf_groups.append( parent_obj )
                        active = True            
        return active



class AddCylindricalSource():
    """Add cylindrical-shaped source of particles"""
 
    def GetResources(self):
        moddir = os.path.expanduser("~") + "/.FreeCAD/Mod/ef"
        return {'Pixmap'  : moddir + '/icons/add_cylindrical_source.ico',
                'Accel' : "Shift+C", # a default shortcut (optional)
                'MenuText': "Add cylindrical-shaped source of particles",
                'ToolTip' : "Add cylindrical-shaped source of particles"}
 
    def Activated(self):
        for ef_conf_group in self.selected_ef_conf_groups:
            source_conf = ef_conf_group.newObject(
                "App::FeaturePython", "Cylindrical Source" )
            ParticleCylindricalSourceConfigPart( source_conf )
        FreeCAD.ActiveDocument.recompute()                        
        return
 
    def IsActive(self):
        # Add source only if ef-group is selected
        # todo: check if selected object is ef-conf group
        # or directly belongs to ef-conf group
        sel = FreeCADGui.Selection.getSelection()
        self.selected_ef_conf_groups = []
        active = False
        for obj in sel:
            if "ef" in obj.Name:
                self.selected_ef_conf_groups.append( obj )
                active = True
            else:
                for parent_obj in obj.InList:
                    if "ef" in parent_obj.Name:
                        self.selected_ef_conf_groups.append( parent_obj )
                        active = True            
        return active
    

class AddInnerRegionBox():
    """Add box inner region"""
 
    def GetResources(self):
        moddir = os.path.expanduser("~") + "/.FreeCAD/Mod/ef"
        return {'Pixmap'  : moddir + '/icons/add_box_inner_region.svg',
                'Accel' : "Shift+R", # a default shortcut (optional)
                'MenuText': "Add box-shaped inner region",
                'ToolTip' : "Add box-shaped inner region"}
 
    def Activated(self):
        for ef_conf_group in self.selected_ef_conf_groups:
            inner_reg_conf = ef_conf_group.newObject(
                "App::FeaturePython", "Inner_region_box" )
            InnerRegionBoxConfigPart( inner_reg_conf )
        FreeCAD.ActiveDocument.recompute()
        return
 
    def IsActive(self):
        # Add source only if ef-group is selected
        # todo: check if selected object is ef-conf group
        # or directly belongs to ef-conf group
        sel = FreeCADGui.Selection.getSelection()
        self.selected_ef_conf_groups = []
        active = False
        for obj in sel:
            if "ef" in obj.Name:
                self.selected_ef_conf_groups.append( obj )
                active = True
            else:
                for parent_obj in obj.InList:
                    if "ef" in parent_obj.Name:
                        self.selected_ef_conf_groups.append( parent_obj )
                        active = True            
        return active
    

class GenerateConfFile():
    """Generate .conf file suitable for ef"""
 
    def GetResources(self):
        moddir = os.path.expanduser("~") + "/.FreeCAD/Mod/ef"
        return {'Pixmap'  : moddir + '/icons/generate_config.svg',
                'Accel' : "Shift+G", # a default shortcut (optional)
                'MenuText': "Generate .conf file",
                'ToolTip' : "Generate .conf file"}
                
    def IsActive(self):
        # Add source only if ef-group is selected
        # todo: check if selected object is ef-conf group
        # or directly belongs to ef-conf group
        sel = FreeCADGui.Selection.getSelection()
        self.selected_ef_conf_groups = []
        active = False
        for obj in sel:
            if "ef" in obj.Name:
                self.selected_ef_conf_groups.append( obj )
                active = True
            else:
                for parent_obj in obj.InList:
                    if "ef" in parent_obj.Name:
                        self.selected_ef_conf_groups.append( parent_obj )
                        active = True            
        return active

    def Activated(self):
        for ef_grp in self.selected_ef_conf_groups:
            ### Generate and write config
            config_text = self.generate_config_text( ef_grp )            
            config_filename = self.write_config( config_text, ef_grp.Name )            
            
    def generate_config_text( self, ef_group ):
        config_text = []
        config_text.append( "; Generated by FreeCAD module\n" )
        config_text.append( "\n" )
        
        objects_in_grp = ef_group.Group
        for obj in objects_in_grp:
            config_text.extend( obj.Proxy.generate_config_part() )

        return config_text
    
    def write_config( self, config_text, ef_group_name ):
        default_dialog_path = "./"
        default_conf_name = ef_group_name + ".conf"
        conf_filename, filename_filter = QtGui.QFileDialog.getSaveFileName(
            None, "Generate ef config",
            default_dialog_path + default_conf_name,
            "*.conf" )
        if conf_filename == "":
            FreeCAD.Console.PrintMessage( "Config generation aborted: "
                                          "file to write was not selected" + "\n" )
        else:                                               
            with open( conf_filename, 'w') as f:
                f.writelines( config_text )
        return conf_filename        



class RunEf():
    """Run Ef"""
 
    def GetResources(self):
        moddir = os.path.expanduser("~") + "/.FreeCAD/Mod/ef"
        return {'Pixmap'  : moddir + '/icons/run_ef.svg',
                'Accel' : "Shift+S", # a default shortcut (optional)
                'MenuText': "Run Ef",
                'ToolTip' : "Run Ef"}
 
    def Activated(self):        
        for ef_conf_group in self.selected_ef_conf_groups:
            # todo: generate config to temp file in temp directory, run ef on this config
            # Rename 'command' to 'ef_command' 
            run_ef = ef_conf_group.getObject("Run_Ef")
            freecad_workdir = os.getcwd()
            os.chdir( run_ef.change_workdir_to )
            stdout = subprocess.Popen( run_ef.command, shell = True,
                                       stdout = subprocess.PIPE ).stdout.read()
            FreeCAD.Console.PrintMessage( stdout )
            # https://stackoverflow.com/questions/803265/getting-realtime-output-using-subprocess
            # realtime output for subprocess
            os.chdir( freecad_workdir )
        FreeCAD.ActiveDocument.recompute()
        return
 
    def IsActive(self):
        # Add source only if ef-group is selected
        # todo: check if selected object is ef-conf group
        # or directly belongs to ef-conf group
        sel = FreeCADGui.Selection.getSelection()
        self.selected_ef_conf_groups = []
        active = False
        for obj in sel:
            if "ef" in obj.Name:
                self.selected_ef_conf_groups.append( obj )
                active = True
            else:
                for parent_obj in obj.InList:
                    if "ef" in parent_obj.Name:
                        self.selected_ef_conf_groups.append( parent_obj )
                        active = True            
        return active


    

###


    
class TimeGridConfigPart:
    """Properties and representation of time_grid config part"""

    def __init__( self, obj ):
        obj.addProperty(
            "App::PropertyString", "total_time",
            "Time grid", "Total simulation time" ).total_time = "1.0"
        obj.addProperty(
            "App::PropertyString", "time_save_step",
            "Time grid", "Time step between checkpoints" ).time_save_step = "1e-3"
        obj.addProperty(
            "App::PropertyString", "time_step_size",
            "Time grid", "Time step" ).time_step_size = "1e-5"
        obj.Proxy = self
        obj.ViewObject.Proxy = self
        self.doc_object = obj
        self.view_object = obj.ViewObject

        
    def execute(self, fp):
        '''Executed when document is recomputated. This method is mandatory'''
        return

    
    def updateData(self, fp, prop):
        '''If a property of the handled feature has changed 
        we have the chance to handle this here'''
        return

    def attach(self, obj):
        ''' Setup the scene sub-graph of the view provider, this method is mandatory '''
        # todo: represent time grid as text on 3d-screen
        # self.text = coin.SoGroup()
        # self.t1 = coin.SoAsciiText()
        # self.t1.string = "arghk"
        # self.text.addChild( self.t1 )
        return

    def generate_config_part( self ):
        conf_part = []
        conf_part.append( "[Time grid]\n" )
        export_property_names = [ "total_time",
                                  "time_save_step",
                                  "time_step_size" ]
        for x in export_property_names:
            conf_part.append(
                "{0} = {1}\n".format( x, self.doc_object.getPropertyByName( x ) ) )
        conf_part.append("\n")
        return conf_part
    
    def __getstate__(self):
        '''When saving the document this object gets stored using Python's json module.
        Since we have some un-serializable parts 
        here -- the Coin stuff -- we must define this method
        to return a tuple of all serializable objects or None.'''
        doc_object_name = self.doc_object.Name
        return { "doc_object_name": doc_object_name }
 
    def __setstate__(self, state):
        '''When restoring the serialized object from document 
        we have the chance to set some internals here.
        Since no data were serialized nothing needs to be done here.'''
        doc_object_name = state[ "doc_object_name" ]
        self.doc_object = FreeCAD.ActiveDocument.getObject( doc_object_name )
        self.view_object = self.doc_object.ViewObject
        return None

    
class OutputFilenameConfigPart():
    """Properties and representation of output_filename config part"""
    
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
        '''Executed when document is recomputated. This method is mandatory'''
        return

    def updateData(self, fp, prop):
        '''If a property of the handled feature has changed 
        we have the chance to handle this here'''
        return

    def attach(self, obj):
        '''Setup the scene sub-graph of the view provider, this method is mandatory'''
        # todo: represent output_filename as text on 3d-screen
        return
    
    def generate_config_part( self ):
        conf_part = []
        conf_part.append( "[Output filename]\n" )
        export_property_names = [ "output_filename_suffix", "output_filename_prefix" ]
        for x in export_property_names:
            conf_part.append(
                "{0} = {1}\n".format( x, self.doc_object.getPropertyByName( x ) ) )
        conf_part.append("\n")
        return conf_part

    def __getstate__(self):
        '''When saving the document this object gets stored using Python's json module.
        Since we have some un-serializable parts 
        here -- the Coin stuff -- we must define this method
        to return a tuple of all serializable objects or None.'''
        doc_object_name = self.doc_object.Name
        return { "doc_object_name": doc_object_name }
 
    def __setstate__(self, state):
        '''When restoring the serialized object from document 
        we have the chance to set some internals here.
        Since no data were serialized nothing needs to be done here.'''
        doc_object_name = state[ "doc_object_name" ]
        self.doc_object = FreeCAD.ActiveDocument.getObject( doc_object_name )
        self.view_object = self.doc_object.ViewObject
        return None



class ParticleInteractionModelConfigPart():
    """Properties and representation of output_filename config part"""
    
    def __init__( self, obj ):
        obj.addProperty(
            "App::PropertyEnumeration",
            "particle_interaction_model",
            "Base",
            "Interaction of particles").particle_interaction_model = ["noninteracting", "PIC"]

        obj.Proxy = self
        obj.ViewObject.Proxy = self
        self.doc_object = obj
        self.view_object = obj.ViewObject

    def execute(self, fp):
        '''Executed when document is recomputated. This method is mandatory'''
        return

    def updateData(self, fp, prop):
        '''If a property of the handled feature has changed 
        we have the chance to handle this here'''
        return

    def attach(self, obj):
        '''Setup the scene sub-graph of the view provider, this method is mandatory'''
        return
    
    def generate_config_part( self ):
        conf_part = []
        conf_part.append( "[Particle interaction model]\n" )
        export_property_names = [ "particle_interaction_model" ]
        for x in export_property_names:
            conf_part.append(
                "{0} = {1}\n".format( x, self.doc_object.getPropertyByName( x ) ) )
        conf_part.append("\n")
        return conf_part

    def __getstate__(self):
        '''When saving the document this object gets stored using Python's json module.
        Since we have some un-serializable parts 
        here -- the Coin stuff -- we must define this method
        to return a tuple of all serializable objects or None.'''
        doc_object_name = self.doc_object.Name
        return { "doc_object_name": doc_object_name }
 
    def __setstate__(self, state):
        '''When restoring the serialized object from document 
        we have the chance to set some internals here.
        Since no data were serialized nothing needs to be done here.'''
        doc_object_name = state[ "doc_object_name" ]
        self.doc_object = FreeCAD.ActiveDocument.getObject( doc_object_name )
        self.view_object = self.doc_object.ViewObject
        return None

    

    
class SpatialMeshConfigPart():
    """Properties and representation of spatial_mesh config part"""

    def __init__( self, obj ):
        obj.addProperty(
            "App::PropertyString", "grid_x_size",
            "Spatial mesh", "Computational volume X-size" ).grid_x_size = "1.0"
        obj.addProperty(
            "App::PropertyString", "grid_x_step",
            "Spatial mesh", "X-step size" ).grid_x_step = "0.1"
        obj.addProperty(
            "App::PropertyString", "grid_y_size",
            "Spatial mesh", "Computational volume Y-size" ).grid_y_size = "1.0"
        obj.addProperty(
            "App::PropertyString", "grid_y_step",
            "Spatial mesh", "Y-step size" ).grid_y_step = "0.1"
        obj.addProperty(
            "App::PropertyString", "grid_z_size",
            "Spatial mesh", "Computational volume Z-size" ).grid_z_size = "1.0"
        obj.addProperty(
            "App::PropertyString", "grid_z_step",
            "Spatial mesh", "Z-step size" ).grid_z_step = "0.1"
        obj.addProperty("Part::PropertyPartShape", "Shape",
                        "Spatial mesh", "Computational volume box")
        obj.ViewObject.addProperty("App::PropertyColor", "Color",
                                   "Spatial mesh", "Volume box color").Color=(1.0,0.0,0.0)
        obj.Proxy = self
        obj.ViewObject.Proxy = self
        self.doc_object = obj
        self.view_object = obj.ViewObject

    def execute(self, fp):
        '''Executed when document is recomputated. This method is mandatory'''
        return
                
    def attach(self, obj):
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

    def updateData(self, obj, prop ):
        "Executed when propery in field 'data' is changed"
        # todo: recompute only 'prop'
        x_size = float( obj.getPropertyByName("grid_x_size") )
        y_size = float( obj.getPropertyByName("grid_y_size") )
        z_size = float( obj.getPropertyByName("grid_z_size") )
        self.trans.translation.setValue( [ x_size/2, y_size/2, z_size/2 ] )
        self.box.width.setValue( x_size )
        self.box.height.setValue( y_size )
        self.box.depth.setValue( z_size )
    
    def getDisplayModes(self,obj):
        "Return a list of display modes."
        modes=[]
        modes.append("Shaded")
        modes.append("Wireframe")
        return modes
 
    def getDefaultDisplayMode(self):
        '''Return the name of the default display mode. 
        It must be defined in getDisplayModes.'''
        return "Wireframe"

    def setDisplayMode(self,mode):
        return mode

    def onChanged(self, vp, prop):
        "Executed if any property is changed"
        if prop == "Color":
            c = vp.getPropertyByName("Color")
            self.color.rgb.setValue( c[0], c[1], c[2] )
    
    def __getstate__(self):
        doc_object_name = self.doc_object.Name
        return { "doc_object_name": doc_object_name }
 
    def __setstate__(self, state):
        doc_object_name = state[ "doc_object_name" ]
        self.doc_object = FreeCAD.ActiveDocument.getObject( doc_object_name )
        self.view_object = self.doc_object.ViewObject
        return None
    
    def generate_config_part( self ):
        conf_part = []
        conf_part.append( "[Spatial mesh]\n" )
        export_property_names = [ "grid_x_size", "grid_x_step",
                                  "grid_y_size", "grid_y_step",
                                  "grid_z_size", "grid_z_step" ]
        for x in export_property_names:
            conf_part.append(
                "{0} = {1}\n".format( x, self.doc_object.getPropertyByName( x ) ) )
        conf_part.append("\n")
        return conf_part


class BoundaryConditionsConfigPart():
    """Properties and representation of boundary_conditions config part"""
    
    def __init__( self, obj ):
        obj.addProperty(
            "App::PropertyString",
            "boundary_phi_left",
            "Boundary conditions",
            "Potential on left boundary").boundary_phi_left = "0.0"
        obj.addProperty(
            "App::PropertyString",
            "boundary_phi_right",
            "Boundary conditions",
            "Potential on right boundary").boundary_phi_right = "0.0"
        obj.addProperty(
            "App::PropertyString",
            "boundary_phi_top",
            "Boundary conditions",
            "Potential on top boundary").boundary_phi_top = "0.0"
        obj.addProperty(
            "App::PropertyString",
            "boundary_phi_bottom",
            "Boundary conditions",
            "Potential on bottom boundary").boundary_phi_bottom = "0.0"
        obj.addProperty(
            "App::PropertyString",
            "boundary_phi_near",
            "Boundary conditions",
            "Potential on near boundary").boundary_phi_near = "0.0"
        obj.addProperty(
            "App::PropertyString",
            "boundary_phi_far",
            "Boundary conditions",
            "Potential on far boundary").boundary_phi_far = "0.0"
        obj.Proxy = self
        obj.ViewObject.Proxy = self
        self.doc_object = obj
        self.view_object = obj.ViewObject

    def execute(self, fp):
        '''Executed when document is recomputated. This method is mandatory'''
        return

    def updateData(self, fp, prop):
        '''If a property of the handled feature has changed 
        we have the chance to handle this here'''
        return

    def attach(self, obj):
        '''Setup the scene sub-graph of the view provider, this method is mandatory'''
        return
        
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

    def __getstate__(self):
        doc_object_name = self.doc_object.Name
        return { "doc_object_name": doc_object_name }
 
    def __setstate__(self, state):
        doc_object_name = state[ "doc_object_name" ]
        self.doc_object = FreeCAD.ActiveDocument.getObject( doc_object_name )
        self.view_object = self.doc_object.ViewObject
        return None

    
class MagneticFieldConfigPart():
    """Properties and representation of magnetic_field config part"""
    
    def __init__( self, obj ):
        obj.addProperty(
            "App::PropertyString",
            "magnetic_field_x",
            "External magnetic field",
            "Field magnitude along X axis").magnetic_field_x = "0.0"
        obj.addProperty(
            "App::PropertyString",
            "magnetic_field_y",
            "External magnetic field",
            "Field magnitude along Y axis").magnetic_field_y = "0.0"
        obj.addProperty(
            "App::PropertyString",
            "magnetic_field_z",
            "External magnetic field",
            "Field magnitude along Z axis").magnetic_field_z = "0.0"
        obj.addProperty(
            "App::PropertyString",
            "speed_of_light",
            "External magnetic field",
            "Speed of light").speed_of_light = "3.0e10"
        obj.Proxy = self
        obj.ViewObject.Proxy = self
        self.doc_object = obj
        self.view_object = obj.ViewObject

    def execute(self, fp):
        '''Executed when document is recomputated. This method is mandatory'''
        return

    def updateData(self, fp, prop):
        '''If a property of the handled feature has changed we 
        have the chance to handle this here'''
        return

    def attach(self, obj):
        '''Setup the scene sub-graph of the view provider, this method is mandatory'''
        return
    
    def generate_config_part( self ):
        conf_part = []
        conf_part.append( "[External magnetic field]\n" )
        export_property_names = [ "magnetic_field_x", "magnetic_field_y",
                                  "magnetic_field_z", "speed_of_light" ]
        for x in export_property_names:
            conf_part.append(
                "{0} = {1}\n".format( x, self.doc_object.getPropertyByName( x ) ) )
        conf_part.append("\n")
        return conf_part

    def __getstate__(self):
        doc_object_name = self.doc_object.Name
        return { "doc_object_name": doc_object_name }
 
    def __setstate__(self, state):
        doc_object_name = state[ "doc_object_name" ]
        self.doc_object = FreeCAD.ActiveDocument.getObject( doc_object_name )
        self.view_object = self.doc_object.ViewObject
        return None

    
class ParticleSourceConfigPart():
    """Particle source region"""

    def __init__( self, obj ):
        obj.addProperty(
            "App::PropertyEnumeration",
            "individual_charge_or_total_current",
            "Base",
            "Specify particles' charge or total source current")
        obj.individual_charge_or_total_current = ["Particles' charge", "Source current"]
        obj.addProperty(
            "App::PropertyEnumeration",
            "mass_or_charge_to_mass",
            "Base",
            "Specify particles' mass or charge-to-mass ratio")
        obj.mass_or_charge_to_mass = [ "Mass", "Charge-to-mass" ]
        obj.addProperty(
            "App::PropertyString",
            "initial_number_of_particles",
            "Number of particles",
            "Initial number of particles" ).initial_number_of_particles = "1000"
        obj.addProperty(
            "App::PropertyString",
            "particles_to_generate_each_step",
            "Number of particles",
            "Number of particles to add at each time step" ).particles_to_generate_each_step = "1000"
        obj.addProperty(
            "App::PropertyString",
            "current",
            "Number of particles",
            "I = q * N / dt" ).current = "10" # default value is unimportant; it will be recalculated.
        obj.addProperty(
            "App::PropertyString",
            "box_x_left",
            "Position",
            "Position of the left side of the source" ).box_x_left = "0.6"
        obj.addProperty(
            "App::PropertyString",
            "box_x_right",
            "Position",
            "Position of the right side of the source" ).box_x_right = "0.4"
        obj.addProperty(
            "App::PropertyString",
            "box_y_bottom",
            "Position",
            "Position of the bottom side of the source" ).box_y_bottom = "0.4"
        obj.addProperty(
            "App::PropertyString",
            "box_y_top",
            "Position",
            "Position of the top side of the source" ).box_y_top = "0.6"
        obj.addProperty(
            "App::PropertyString",
            "box_z_near",
            "Position",
            "Position of the near side of the source" ).box_z_near = "0.4"
        obj.addProperty(
            "App::PropertyString",
            "box_z_far",
            "Position",
            "Position of the far side of the source" ).box_z_far = "0.6"
        obj.addProperty(
            "App::PropertyString",
            "mean_momentum_x",
            "Momentum",
            "Mean momentum in X direction" ).mean_momentum_x = "1.0"
        obj.addProperty(
            "App::PropertyString",
            "mean_momentum_y",
            "Momentum",
            "Mean momentum in Y direction" ).mean_momentum_y = "1.0"
        obj.addProperty(
            "App::PropertyString",
            "mean_momentum_z",
            "Momentum",
            "Mean momentum in Z direction" ).mean_momentum_z = "1.0"
        obj.addProperty(
            "App::PropertyString",
            "temperature",
            "Momentum",
            "Temperature" ).temperature = "1.0"
        obj.addProperty(
            "App::PropertyString",
            "charge",
            "Particle properties",
            "Particles' charge" ).charge = "1.0"
        obj.addProperty(
            "App::PropertyString",
            "mass",
            "Particle properties",
            "Particles' mass (calculated automatically from q and q/m)" ).mass = "1.0"
        obj.addProperty(
            "App::PropertyString",
            "charge_to_mass_ratio",
            "Particle properties",
            "Particles' charge to mass ratio" ).charge_to_mass_ratio = "1.0"
        obj.ViewObject.addProperty(
            "App::PropertyColor", "Color",
            "Spatial mesh", "Volume box color").Color=(0.0, 0.0, 1.0)

        obj.Proxy = self
        obj.ViewObject.Proxy = self
        self.doc_object = obj
        self.view_object = obj.ViewObject

    def execute( self, obj ):
        '''Executed when document is recomputated. This method is mandatory'''
        dt = self.get_time_step( obj )
        N = int( obj.particles_to_generate_each_step )
        individual_charge_or_total_current = obj.getPropertyByName(
            "individual_charge_or_total_current")
        mass_or_charge_to_mass = obj.getPropertyByName(
            "mass_or_charge_to_mass")
        
        if individual_charge_or_total_current == "Particles' charge":
            # todo: make certain fields read-only
            # e.g., obj.setEditorMode( "current", readonly )
            q = float( obj.charge )
            I = q * N / dt
            obj.current = str( I )
        elif individual_charge_or_total_current == "Source current":
            I = float( obj.current )
            q = I * dt / N        
            obj.charge = str( q )

        if mass_or_charge_to_mass == "Mass":
            # todo: make certain fields read-only
            # e.g., obj.setEditorMode( "current", readonly )
            m = float( obj.mass )
            q_to_m = q / m
            obj.charge_to_mass_ratio = str( q_to_m )
        elif mass_or_charge_to_mass == "Charge-to-mass":
            q_to_m = float( obj.charge_to_mass_ratio )
            m = abs( 1 / q_to_m * q )
            obj.mass = str( m )            

        return

    def attach(self, obj):
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

    def updateData( self, obj, prop ):
        "Executed when propery in field 'data' is changed"
        # todo: move charge-current recomputation here from 'execute'
        x0 = float( obj.getPropertyByName("box_x_right") )
        y0 = float( obj.getPropertyByName("box_y_bottom") )
        z0 = float( obj.getPropertyByName("box_z_near") )
        xlen = float( obj.getPropertyByName("box_x_left") ) - x0
        ylen = float( obj.getPropertyByName("box_y_top") ) - y0
        zlen = float( obj.getPropertyByName("box_z_far") ) - z0
        self.trans.translation.setValue( [ x0 + xlen / 2,
                                           y0 + ylen / 2,
                                           z0 + zlen / 2 ] )
        self.box.width.setValue( xlen )
        self.box.height.setValue( ylen )
        self.box.depth.setValue( zlen )
        
        return

    
    def getDisplayModes(self,obj):
        "Return a list of display modes."
        modes=[]
        modes.append("Shaded")
        modes.append("Wireframe")
        return modes
 
    def getDefaultDisplayMode(self):
        '''Return the name of the default display mode. 
        It must be defined in getDisplayModes.'''
        return "Wireframe"

    def setDisplayMode(self,mode):
        return mode

    def onChanged(self, vp, prop):
        "Executed if any property is changed"
        if prop == "Color":
            c = vp.getPropertyByName("Color")
            self.color.rgb.setValue( c[0], c[1], c[2] )

    def __getstate__(self):
        doc_object_name = self.doc_object.Name
        return { "doc_object_name": doc_object_name }
 
    def __setstate__(self, state):
        doc_object_name = state[ "doc_object_name" ]
        self.doc_object = FreeCAD.ActiveDocument.getObject( doc_object_name )
        self.view_object = self.doc_object.ViewObject
        return None

    def get_time_step( self, obj ):
        # todo: to get dt, get object named "Time_grid" from the first
        # group which the source belongs to.
        # Instead, pass reference to "Time_grid" object in the source-constructor.
        # todo: source properties are not recomputed when dt is changed. do something about it.
        dt = float( obj.InList[0].getObject("Time_grid").getPropertyByName("time_step_size") )
        return dt
    
    def generate_config_part( self ):
        conf_part = []
        source_name = self.doc_object.getPropertyByName( "Label" )
        conf_part.append( "[Particle_source_box.{0}]\n".format( source_name ) )
        export_property_names = [ "initial_number_of_particles",
                                  "particles_to_generate_each_step",
                                  "box_x_left",   "box_x_right",
                                  "box_y_bottom", "box_y_top",
                                  "box_z_near",   "box_z_far",
                                  "mean_momentum_x",
                                  "mean_momentum_y",
                                  "mean_momentum_z",
                                  "temperature",
                                  "charge",
                                  "mass" ]
        for x in export_property_names:
            conf_part.append(
                "{0} = {1}\n".format( x, self.doc_object.getPropertyByName( x ) ) )

        comments = [ "individual_charge_or_total_current",
                     "mass_or_charge_to_mass",
                     "charge_to_mass_ratio",
                     "current" ]
        for x in comments:
            conf_part.append(
                ";{0} = {1}\n".format( x, self.doc_object.getPropertyByName( x ) ) )

        conf_part.append("\n")
        return conf_part





class ParticleCylindricalSourceConfigPart():
    """Particle cylindrical source region"""

    def __init__( self, obj ):
        obj.addProperty(
            "App::PropertyEnumeration",
            "individual_charge_or_total_current",
            "Base",
            "Specify particles' charge or total source current")
        obj.individual_charge_or_total_current = ["Particles' charge", "Source current"]
        obj.addProperty(
            "App::PropertyEnumeration",
            "mass_or_charge_to_mass",
            "Base",
            "Specify particles' mass or charge-to-mass ratio")
        obj.mass_or_charge_to_mass = [ "Mass", "Charge-to-mass" ]
        obj.addProperty(
            "App::PropertyString",
            "initial_number_of_particles",
            "Number of particles",
            "Initial number of particles" ).initial_number_of_particles = "1000"
        obj.addProperty(
            "App::PropertyString",
            "particles_to_generate_each_step",
            "Number of particles",
            "Number of particles to add at each time step" ).particles_to_generate_each_step = "1000"
        obj.addProperty(
            "App::PropertyString",
            "current",
            "Number of particles",
            "I = q * N / dt" ).current = "10" # default value is unimportant;
        # it will be recalculated.
        ### Size
        obj.addProperty(
            "App::PropertyEnumeration",
            "cylinder_axis_direction",
            "Size",
            "Cylinder along axis").cylinder_axis_direction = ["X", "Y", "Z"]
        obj.addProperty(
            "App::PropertyString",
            "cylinder_length",
            "Size",
            "Cylinder axis length").cylinder_length = "0.5"
        obj.addProperty(
            "App::PropertyString",
            "cylinder_radius",
            "Size",
            "Cylinder radius" ).cylinder_radius = "0.05"
        ###
        obj.addProperty(
            "App::PropertyString",
            "cylinder_axis_start_x",
            "Size",
            "Position of the left side of the source" ).cylinder_axis_start_x = "0.0"
        obj.addProperty(
            "App::PropertyString",
            "cylinder_axis_start_y",
            "Size",
            "Position of the right side of the source" ).cylinder_axis_start_y = "0.0"
        obj.addProperty(
            "App::PropertyString",
            "cylinder_axis_start_z",
            "Size",
            "Position of the bottom side of the source" ).cylinder_axis_start_z = "0.0"
        obj.addProperty(
            "App::PropertyString",
            "cylinder_axis_end_x",
            "Size",
            "Position of the top side of the source" ).cylinder_axis_end_x = "0.2"
        obj.addProperty(
            "App::PropertyString",
            "cylinder_axis_end_y",
            "Size",
            "Position of the near side of the source" ).cylinder_axis_end_y = "0.2"
        obj.addProperty(
            "App::PropertyString",
            "cylinder_axis_end_z",
            "Size",
            "Position of the far side of the source" ).cylinder_axis_end_z = "0.3"
        ###
        obj.addProperty(
            "App::PropertyString",
            "mean_momentum_x",
            "Momentum",
            "Mean momentum in X direction" ).mean_momentum_x = "1.0"
        obj.addProperty(
            "App::PropertyString",
            "mean_momentum_y",
            "Momentum",
            "Mean momentum in Y direction" ).mean_momentum_y = "1.0"
        obj.addProperty(
            "App::PropertyString",
            "mean_momentum_z",
            "Momentum",
            "Mean momentum in Z direction" ).mean_momentum_z = "1.0"
        obj.addProperty(
            "App::PropertyString",
            "temperature",
            "Momentum",
            "Temperature" ).temperature = "1.0"
        obj.addProperty(
            "App::PropertyString",
            "charge",
            "Particle properties",
            "Particles' charge" ).charge = "1.0"
        obj.addProperty(
            "App::PropertyString",
            "mass",
            "Particle properties",
            "Particles' mass (calculated automatically from q and q/m)" ).mass = "1.0"
        obj.addProperty(
            "App::PropertyString",
            "charge_to_mass_ratio",
            "Particle properties",
            "Particles' charge to mass ratio" ).charge_to_mass_ratio = "1.0"
        obj.ViewObject.addProperty(
            "App::PropertyColor", "Color",
            "Spatial mesh", "Volume box color").Color=(0.0, 0.0, 1.0)

        # hide axis-end
        obj.setEditorMode( "cylinder_axis_end_x", 1 )
        obj.setEditorMode( "cylinder_axis_end_y", 1 )
        obj.setEditorMode( "cylinder_axis_end_z", 1 )
        
        obj.Proxy = self
        obj.ViewObject.Proxy = self
        self.doc_object = obj
        self.view_object = obj.ViewObject

    def execute( self, obj ):
        '''Executed when document is recomputated. This method is mandatory'''
        dt = self.get_time_step( obj )
        N = int( obj.particles_to_generate_each_step )
        individual_charge_or_total_current = obj.getPropertyByName(
            "individual_charge_or_total_current")
        mass_or_charge_to_mass = obj.getPropertyByName(
            "mass_or_charge_to_mass")

        
        if individual_charge_or_total_current == "Particles' charge":
            # todo: make certain fields read-only
            # e.g., obj.setEditorMode( "current", readonly )
            q = float( obj.charge )
            I = q * N / dt
            obj.current = str( I )
        elif individual_charge_or_total_current == "Source current":
            I = float( obj.current )
            q = I * dt / N        
            obj.charge = str( q )

        if mass_or_charge_to_mass == "Mass":
            # todo: make certain fields read-only
            # e.g., obj.setEditorMode( "current", readonly )
            m = float( obj.mass )
            q_to_m = q / m
            obj.charge_to_mass_ratio = str( q_to_m )
        elif mass_or_charge_to_mass == "Charge-to-mass":
            q_to_m = float( obj.charge_to_mass_ratio )
            m = abs( 1 / q_to_m * q )
            obj.mass = str( m )            

        cylinder_axis_direction = obj.cylinder_axis_direction
        cylinder_length = float( obj.cylinder_length )
        cylinder_axis_start_x = float( obj.cylinder_axis_start_x )
        cylinder_axis_start_y = float( obj.cylinder_axis_start_y )
        cylinder_axis_start_z = float( obj.cylinder_axis_start_z )
        if cylinder_axis_direction == 'X':
            obj.cylinder_axis_end_x = str( cylinder_axis_start_x + cylinder_length )
            obj.cylinder_axis_end_y = str( cylinder_axis_start_y )
            obj.cylinder_axis_end_z = str( cylinder_axis_start_z )
        elif cylinder_axis_direction == 'Y':
            obj.cylinder_axis_end_x = str( cylinder_axis_start_x )
            obj.cylinder_axis_end_y = str( cylinder_axis_start_y + cylinder_length )
            obj.cylinder_axis_end_z = str( cylinder_axis_start_z )
        elif cylinder_axis_direction == 'Z':
            obj.cylinder_axis_end_x = str( cylinder_axis_start_x )
            obj.cylinder_axis_end_y = str( cylinder_axis_start_y )
            obj.cylinder_axis_end_z = str( cylinder_axis_start_z + cylinder_length )

        return

    def attach(self, obj):    
        self.trans = coin.SoTranslation()
        self.rot_xyz = coin.SoRotationXYZ()
        self.color = coin.SoBaseColor()
        self.cyl = coin.SoCylinder()

        self.shaded = coin.SoGroup()
        self.shaded.addChild( self.color )
        self.shaded.addChild( self.trans )
        self.shaded.addChild( self.rot_xyz )
        self.shaded.addChild( self.cyl )
        obj.addDisplayMode( self.shaded, "Shaded" )
        
        style = coin.SoDrawStyle()
        style.style = coin.SoDrawStyle.LINES
        self.wireframe = coin.SoGroup()
        self.wireframe.addChild( style )
        self.wireframe.addChild( self.color )                
        self.wireframe.addChild( self.trans )
        self.wireframe.addChild( self.rot_xyz )
        self.wireframe.addChild( self.cyl )
        obj.addDisplayMode( self.wireframe, "Wireframe" )
        
        self.onChanged( obj, "Color" )
        return

    def updateData( self, obj, prop ):
        "Executed when propery in field 'data' is changed"        
        cylinder_axis_direction = obj.getPropertyByName("cylinder_axis_direction")
        cylinder_length = float( obj.getPropertyByName("cylinder_length") )
        cylinder_radius = float( obj.getPropertyByName("cylinder_radius") )
        cylinder_axis_start_x = float( obj.getPropertyByName("cylinder_axis_start_x") )
        cylinder_axis_start_y = float( obj.getPropertyByName("cylinder_axis_start_y") )
        cylinder_axis_start_z = float( obj.getPropertyByName("cylinder_axis_start_z") )
        
        if cylinder_axis_direction == 'X':
            self.rot_xyz.axis.setValue( 2 )
            self.rot_xyz.angle.setValue( pi / 2 )
            self.trans.translation.setValue(
                [ cylinder_axis_start_x + cylinder_length / 2,
                  cylinder_axis_start_y,
                  cylinder_axis_start_z ] )            
        elif cylinder_axis_direction == 'Y':
            self.rot_xyz.axis.setValue( 1 )
            self.rot_xyz.angle.setValue( 0 )
            self.trans.translation.setValue(
                [ cylinder_axis_start_x,
                  cylinder_axis_start_y + cylinder_length / 2,
                  cylinder_axis_start_z ] )
        elif cylinder_axis_direction == 'Z':
            self.rot_xyz.axis.setValue( 0 )
            self.rot_xyz.angle.setValue( pi / 2 )
            self.trans.translation.setValue(
                [ cylinder_axis_start_x,
                  cylinder_axis_start_y,
                  cylinder_axis_start_z + cylinder_length / 2 ] )
            
        self.cyl.radius.setValue( cylinder_radius )
        self.cyl.height.setValue( cylinder_length )
        return

    def getDisplayModes(self,obj):
        "Return a list of display modes."
        modes=[]
        modes.append("Shaded")
        modes.append("Wireframe")
        return modes
 
    def getDefaultDisplayMode(self):
        '''Return the name of the default display mode. 
        It must be defined in getDisplayModes.'''
        return "Wireframe"

    def setDisplayMode(self,mode):
        return mode

    def onChanged(self, vp, prop):
        "Executed if any property is changed"
        if prop == "Color":
            c = vp.getPropertyByName("Color")
            self.color.rgb.setValue( c[0], c[1], c[2] )

    def __getstate__(self):
        doc_object_name = self.doc_object.Name
        return { "doc_object_name": doc_object_name }
 
    def __setstate__(self, state):
        doc_object_name = state[ "doc_object_name" ]
        self.doc_object = FreeCAD.ActiveDocument.getObject( doc_object_name )
        self.view_object = self.doc_object.ViewObject
        return None

    def get_time_step( self, obj ):
        # todo: to get dt, get object named "Time_grid" from the first
        # group which the source belongs to.
        # Instead, pass reference to "Time_grid" object in the source-constructor.
        # todo: source properties are not recomputed when dt is changed.
        #       do something about it.
        dt = float(
            obj.InList[0].getObject("Time_grid").getPropertyByName("time_step_size") )
        return dt
    
    def generate_config_part( self ):
        conf_part = []
        source_name = self.doc_object.getPropertyByName( "Label" )
        conf_part.append( "[Particle_source_cylinder.{0}]\n".format( source_name ) )
        export_property_names = [ "initial_number_of_particles",
                                  "particles_to_generate_each_step",
                                  "cylinder_axis_start_x",
                                  "cylinder_axis_start_y",
                                  "cylinder_axis_start_z",
                                  "cylinder_axis_end_x",
                                  "cylinder_axis_end_y",
                                  "cylinder_axis_end_z",
                                  "cylinder_radius",
                                  "mean_momentum_x",
                                  "mean_momentum_y",
                                  "mean_momentum_z",
                                  "temperature",
                                  "charge",
                                  "mass" ]
        for x in export_property_names:
            conf_part.append(
                "{0} = {1}\n".format( x, self.doc_object.getPropertyByName( x ) ) )

        comments = [ "individual_charge_or_total_current",
                     "mass_or_charge_to_mass",
                     "charge_to_mass_ratio",
                     "current",
                     "cylinder_axis_direction",
                     "cylinder_length" ]
        for x in comments:
            conf_part.append(
                ";{0} = {1}\n".format( x, self.doc_object.getPropertyByName( x ) ) )

        conf_part.append("\n")
        return conf_part


    


class InnerRegionBoxConfigPart():
    """Box inner region"""

    def __init__( self, obj ):
        obj.addProperty(
            "App::PropertyString",
            "box_x_right",
            "Position",
            "Box right side position" ).box_x_right = "0.1"
        obj.addProperty(
            "App::PropertyString",
            "box_x_left",
            "Position",
            "Box left side position" ).box_x_left = "0.9"
        obj.addProperty(
            "App::PropertyString",
            "box_y_bottom",
            "Position",
            "Box bottom side position" ).box_y_bottom = "0.1"
        obj.addProperty(
            "App::PropertyString",
            "box_y_top",
            "Position",
            "Box top side position" ).box_y_top = "0.9"
        obj.addProperty(
            "App::PropertyString",
            "box_z_near",
            "Position",
            "Box near side position" ).box_z_near = "0.1"
        obj.addProperty(
            "App::PropertyString",
            "box_z_far",
            "Position",
            "Box far side position" ).box_z_far = "0.2"
        obj.addProperty(
            "App::PropertyString",
            "potential",
            "Potential",
            "Inner region potential" ).potential = "0.0"                
        obj.ViewObject.addProperty(
            "App::PropertyColor", "Color",
            "Inner region color", "Inner region color").Color=(0.5, 0.5, 0.0)        

        obj.Proxy = self
        obj.ViewObject.Proxy = self
        self.doc_object = obj
        self.view_object = obj.ViewObject

    def execute(self, fp):
        return

    def attach(self, obj):
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

    def updateData(self, obj, prop ):
        "Executed when propery in field 'data' is changed"
        x0 = float( obj.getPropertyByName("box_x_right") )
        y0 = float( obj.getPropertyByName("box_y_bottom") )
        z0 = float( obj.getPropertyByName("box_z_near") )
        xlen = float( obj.getPropertyByName("box_x_left") ) - x0
        ylen = float( obj.getPropertyByName("box_y_top") ) - y0
        zlen = float( obj.getPropertyByName("box_z_far") ) - z0
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
        '''Return the name of the default display mode. 
        It must be defined in getDisplayModes.'''
        return "Shaded"

    def setDisplayMode(self,mode):
        return mode

    def onChanged(self, vp, prop):
        "Executed if any property is changed"
        if prop == "Color":
            c = vp.getPropertyByName("Color")
            self.color.rgb.setValue( c[0],c[1],c[2] )
    
    def __getstate__(self):
        doc_object_name = self.doc_object.Name
        return { "doc_object_name": doc_object_name }
 
    def __setstate__(self, state):
        doc_object_name = state[ "doc_object_name" ]
        self.doc_object = FreeCAD.ActiveDocument.getObject( doc_object_name )
        self.view_object = self.doc_object.ViewObject
        return None

    
    def generate_config_part( self ):
        conf_part = []
        conf_part.append( "[Inner_region_box.{0}]\n".format(
            self.doc_object.getPropertyByName( "Label" ) ) )
        export_property_names = [ "box_x_left",   "box_x_right",
                                  "box_y_bottom", "box_y_top",
                                  "box_z_near",   "box_z_far",
                                  "potential" ]
        for x in export_property_names:
            conf_part.append(
                "{0} = {1}\n".format( x, self.doc_object.getPropertyByName( x ) ) )
        conf_part.append("\n")
        return conf_part



class RunEfConfig:
    """Parameters to run computation"""

    def __init__( self, obj ):
        obj.addProperty(
            "App::PropertyString", "current_workdir",
            "Run Parameters", "Path to working directory" ).current_workdir = os.getcwd()
        obj.setEditorMode( "current_workdir", 1 )
        obj.addProperty(
            "App::PropertyString", "change_workdir_to",
            "Run Parameters", "Path to working directory" ).change_workdir_to = "/tmp/"
        obj.addProperty(
            "App::PropertyString", "command",
            "Run Parameters", "Command to execute" ).command = "./ef.out test.conf"
        obj.Proxy = self
        obj.ViewObject.Proxy = self
        self.doc_object = obj
        self.view_object = obj.ViewObject
        
    def execute(self, fp):
        '''Executed when document is recomputated. This method is mandatory'''
        return

    def updateData(self, fp, prop):
        '''If a property of the handled feature has changed 
        we have the chance to handle this here'''
        return

    def attach(self, obj):
        ''' Setup the scene sub-graph of the view provider, this method is mandatory '''
        return
    
    def __getstate__(self):
        '''When saving the document this object gets stored using Python's json module.
        Since we have some un-serializable parts 
        here -- the Coin stuff -- we must define this method
        to return a tuple of all serializable objects or None.'''
        doc_object_name = self.doc_object.Name
        return { "doc_object_name": doc_object_name }
 
    def __setstate__(self, state):
        '''When restoring the serialized object from document 
        we have the chance to set some internals here.
        Since no data were serialized nothing needs to be done here.'''
        doc_object_name = state[ "doc_object_name" ]
        self.doc_object = FreeCAD.ActiveDocument.getObject( doc_object_name )
        self.view_object = self.doc_object.ViewObject
        return None

    def generate_config_part( self ):
        # no need to add something to config; return empty list
        # todo: avoid calling this method for this class
        return []

    
        
FreeCADGui.addCommand( 'CreateEfConfig', CreateEfConfig() )
FreeCADGui.addCommand( 'AddSourceRegion', AddSourceRegion() )
FreeCADGui.addCommand( 'AddCylindricalSource', AddCylindricalSource() )
FreeCADGui.addCommand( 'AddInnerRegionBox', AddInnerRegionBox() )
FreeCADGui.addCommand( 'GenerateConfFile', GenerateConfFile() )
FreeCADGui.addCommand( 'RunEf', RunEf() )
