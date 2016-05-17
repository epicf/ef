import os
import FreeCAD, FreeCADGui
import Part
from pivy import coin
from PySide import QtGui, QtCore

class CreateEpicfConfig():
    """Create objects for new epicf config"""
 
    def GetResources(self):
        moddir = os.path.expanduser("~") + "/.FreeCAD/Mod/epicf"
        return {'Pixmap'  : moddir + '/icons/new_conf_template.svg',
                'Accel' : "Shift+N", # a default shortcut (optional)
                'MenuText': "New epicf config",
                'ToolTip' : "Minimal new epicf config"}
 
    def Activated(self):
        epicf_conf_group = FreeCAD.ActiveDocument.addObject(
            "App::DocumentObjectGroup", "epicf_conf" )

        time_grid_conf = epicf_conf_group.newObject(
            "App::FeaturePython", "Time grid" )
        TimeGridConfigPart( time_grid_conf )
                
        outfile_conf = epicf_conf_group.newObject(
            "App::FeaturePython", "Output filename")
        OutputFilenameConfigPart( outfile_conf )
        
        spat_mesh_conf = epicf_conf_group.newObject(
            "App::FeaturePython", "Spatial mesh" )
        SpatialMeshConfigPart( spat_mesh_conf )
        
        boundary_cond_conf = epicf_conf_group.newObject(
            "App::FeaturePython", "Boundary conditions" )
        BoundaryConditionsConfigPart( boundary_cond_conf )

        magn_field_conf = epicf_conf_group.newObject(
            "App::FeaturePython", "Magnetic field" )
        MagneticFieldConfigPart( magn_field_conf )

        FreeCAD.ActiveDocument.recompute()
        return
 
    def IsActive(self):        
        return (FreeCAD.ActiveDocument is not None)
            


class AddSourceRegion():
    """Add box-shaped source of particles"""
 
    def GetResources(self):
        moddir = os.path.expanduser("~") + "/.FreeCAD/Mod/epicf"
        return {'Pixmap'  : moddir + '/icons/add_box_source.svg',
                'Accel' : "Shift+S", # a default shortcut (optional)
                'MenuText': "Add source of particles",
                'ToolTip' : "Add box-shaped source of particles"}
 
    def Activated(self):
        for epicf_conf_group in self.selected_epicf_conf_groups:
            source_conf = epicf_conf_group.newObject(
                "App::FeaturePython", "Source" )
            ParticleSourceConfigPart( source_conf )
        FreeCAD.ActiveDocument.recompute()                        
        return
 
    def IsActive(self):
        # Add source only if epicf-group is selected
        # todo: check if selected object is epicf-conf group
        # or directly belongs to epicf-conf group
        sel = FreeCADGui.Selection.getSelection()
        self.selected_epicf_conf_groups = []
        active = False
        for obj in sel:
            if "epicf" in obj.Name:
                self.selected_epicf_conf_groups.append( obj )
                active = True
            else:
                for parent_obj in obj.InList:
                    if "epicf" in parent_obj.Name:
                        self.selected_epicf_conf_groups.append( parent_obj )
                        active = True            
        return active


class AddInnerRegionBox():
    """Add box inner region"""
 
    def GetResources(self):
        moddir = os.path.expanduser("~") + "/.FreeCAD/Mod/epicf"
        return {'Pixmap'  : moddir + '/icons/add_box_inner_region.svg',
                'Accel' : "Shift+R", # a default shortcut (optional)
                'MenuText': "Add box-shaped inner region",
                'ToolTip' : "Add box-shaped inner region - tooltip"}
 
    def Activated(self):
        for epicf_conf_group in self.selected_epicf_conf_groups:
            inner_reg_conf = epicf_conf_group.newObject(
                "App::FeaturePython", "Inner_region_box" )
            InnerRegionBoxConfigPart( inner_reg_conf )
        FreeCAD.ActiveDocument.recompute()
        return
 
    def IsActive(self):
        # Add source only if epicf-group is selected
        # todo: check if selected object is epicf-conf group
        # or directly belongs to epicf-conf group
        sel = FreeCADGui.Selection.getSelection()
        self.selected_epicf_conf_groups = []
        active = False
        for obj in sel:
            if "epicf" in obj.Name:
                self.selected_epicf_conf_groups.append( obj )
                active = True
            else:
                for parent_obj in obj.InList:
                    if "epicf" in parent_obj.Name:
                        self.selected_epicf_conf_groups.append( parent_obj )
                        active = True            
        return active


class AddInnerRegionSTEP():
    """Add STEP inner region"""
 
    def GetResources(self):
        moddir = os.path.expanduser("~") + "/.FreeCAD/Mod/epicf"
        return {'Pixmap'  : moddir + '/icons/add_STEP_inner_region.svg',
                'Accel' : "Shift+T", # a default shortcut (optional)
                'MenuText': "Add STEP-defined inner region",
                'ToolTip' : "Add STEP-defined inner region - tooltip"}
 
    def Activated(self):
        for epicf_conf_group in self.selected_epicf_conf_groups:
            inner_reg_conf = epicf_conf_group.newObject(
                "App::FeaturePython", "Inner_region_STEP" )
            InnerRegionSTEPConfigPart( inner_reg_conf )
        FreeCAD.ActiveDocument.recompute()
        return
 
    def IsActive(self):
        # Add source only if epicf-group is selected
        # todo: check if selected object is epicf-conf group
        # or directly belongs to epicf-conf group
        sel = FreeCADGui.Selection.getSelection()
        self.selected_epicf_conf_groups = []
        active = False
        for obj in sel:
            if "epicf" in obj.Name:
                self.selected_epicf_conf_groups.append( obj )
                active = True
            else:
                for parent_obj in obj.InList:
                    if "epicf" in parent_obj.Name:
                        self.selected_epicf_conf_groups.append( parent_obj )
                        active = True            
        return active
    

class GenerateConfFile():
    """Generate .conf file suitable for epicf """
 
    def GetResources(self):
        moddir = os.path.expanduser("~") + "/.FreeCAD/Mod/epicf"
        return {'Pixmap'  : moddir + '/icons/generate_config.svg',
                'Accel' : "Shift+G", # a default shortcut (optional)
                'MenuText': "Generate .conf file",
                'ToolTip' : "Generate .conf file tooltip"}            
                
    def IsActive(self):
        # Add source only if epicf-group is selected
        # todo: check if selected object is epicf-conf group
        # or directly belongs to epicf-conf group
        sel = FreeCADGui.Selection.getSelection()
        self.selected_epicf_conf_groups = []
        active = False
        for obj in sel:
            if "epicf" in obj.Name:
                self.selected_epicf_conf_groups.append( obj )
                active = True
            else:
                for parent_obj in obj.InList:
                    if "epicf" in parent_obj.Name:
                        self.selected_epicf_conf_groups.append( parent_obj )
                        active = True            
        return active

    def Activated(self):
        for epicf_grp in self.selected_epicf_conf_groups:
            ### Generate and write config
            config_text = self.generate_config_text( epicf_grp )            
            config_filename = self.write_config( config_text, epicf_grp.Name )
            ### Save STEP models in the same directory
            config_dir = os.path.dirname( config_filename )
            self.export_step_models( epicf_grp, config_dir )
            
            
    def generate_config_text( self, epicf_group ):
        config_text = []
        config_text.append( "; Generated by FreeCAD module\n" )
        config_text.append( "\n" )
        
        objects_in_grp = epicf_group.Group
        for obj in objects_in_grp:
            config_text.extend( obj.Proxy.generate_config_part() )

        return config_text
    
    def write_config( self, config_text, epicf_group_name ):
        default_dialog_path = "./"
        default_conf_name = epicf_group_name + ".conf"
        conf_filename, filename_filter = QtGui.QFileDialog.getSaveFileName(
            None, "Generate epicf config",
            default_dialog_path + default_conf_name,
            "*.conf" )
        if conf_filename == "":
            FreeCAD.Console.PrintMessage( "Config generation aborted: "
                                          "file to write was not selected" + "\n" )
        else:                                               
            with open( conf_filename, 'w') as f:
                f.writelines( config_text )
        return conf_filename        

    def export_step_models( self, epicf_group, conf_dir ):
        objects_in_grp = epicf_group.Group
        for obj in objects_in_grp:
            if isinstance( obj.Proxy, InnerRegionSTEPConfigPart ):
                obj.Proxy.export_step_model( conf_dir )

    
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
        "If a property of the handled feature has changed we have the chance to handle this here"
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
        export_property_names = [ "total_time", "time_save_step",
                                  "time_step_size" ]
        for x in export_property_names:
            conf_part.append( "{0} = {1}\n".format( x, self.doc_object.getPropertyByName( x ) ) )
        conf_part.append("\n")
        return conf_part
    
    def __getstate__(self):
        '''When saving the document this object gets stored using Python's json module.\
        Since we have some un-serializable parts here -- the Coin stuff -- we must define this method\
        to return a tuple of all serializable objects or None.'''
        doc_object_name = self.doc_object.Name
        return { "doc_object_name": doc_object_name }
 
    def __setstate__(self, state):
        '''When restoring the serialized object from document we have the chance to set some internals here.\
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
        "If a property of the handled feature has changed we have the chance to handle this here"
        return

    def attach(self, obj):
        ''' Setup the scene sub-graph of the view provider, this method is mandatory '''
        # todo: represent output_filename as text on 3d-screen
        return
    
    def generate_config_part( self ):
        conf_part = []
        conf_part.append( "[Output filename]\n" )
        export_property_names = [ "output_filename_suffix", "output_filename_prefix" ]
        for x in export_property_names:
            conf_part.append( "{0} = {1}\n".format( x, self.doc_object.getPropertyByName( x ) ) )
        conf_part.append("\n")
        return conf_part

    def __getstate__(self):
        '''When saving the document this object gets stored using Python's json module.\
        Since we have some un-serializable parts here -- the Coin stuff -- we must define this method\
        to return a tuple of all serializable objects or None.'''
        doc_object_name = self.doc_object.Name
        return { "doc_object_name": doc_object_name }
 
    def __setstate__(self, state):
        '''When restoring the serialized object from document we have the chance to set some internals here.\
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
        "Return the name of the default display mode. It must be defined in getDisplayModes."
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
            conf_part.append( "{0} = {1}\n".format( x, self.doc_object.getPropertyByName( x ) ) )
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
        "If a property of the handled feature has changed we have the chance to handle this here"
        return

    def attach(self, obj):
        ''' Setup the scene sub-graph of the view provider, this method is mandatory '''
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
        '''Executed when document is recomputated. This method is mandatory'''
        return

    def updateData(self, fp, prop):
        "If a property of the handled feature has changed we have the chance to handle this here"
        return

    def attach(self, obj):
        ''' Setup the scene sub-graph of the view provider, this method is mandatory '''
        return
    
    def generate_config_part( self ):
        conf_part = []
        conf_part.append( "[External magnetic field]\n" )
        export_property_names = [ "magnetic_field_x", "magnetic_field_y",
                                  "magnetic_field_z", "speed_of_light" ]
        for x in export_property_names:
            conf_part.append( "{0} = {1}\n".format( x, self.doc_object.getPropertyByName( x ) ) )
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
            "set_of_parameters",
            "Base",
            "Specify particles' charge or total source current").set_of_parameters = ["Particles' charge",
                                                                                      "Source current"]
        obj.addProperty(
            "App::PropertyString",
            "particle_source_initial_number_of_particles",
            "Number of particles",
            "Initial number of particles" ).particle_source_initial_number_of_particles = "1000"
        obj.addProperty(
            "App::PropertyString",
            "particle_source_particles_to_generate_each_step",
            "Number of particles",
            "Number of particles to add at each time step" ).particle_source_particles_to_generate_each_step = "1000"
        obj.addProperty(
            "App::PropertyString",
            "current",
            "Number of particles",
            "I = q * N / dt" ).current = "10" # default value is unimportant; it will be recalculated.
        obj.addProperty(
            "App::PropertyString",
            "particle_source_x_left",
            "Position",
            "Position of the left side of the source" ).particle_source_x_left = "0.4"
        obj.addProperty(
            "App::PropertyString",
            "particle_source_x_right",
            "Position",
            "Position of the right side of the source" ).particle_source_x_right = "0.6"
        obj.addProperty(
            "App::PropertyString",
            "particle_source_y_bottom",
            "Position",
            "Position of the bottom side of the source" ).particle_source_y_bottom = "0.4"
        obj.addProperty(
            "App::PropertyString",
            "particle_source_y_top",
            "Position",
            "Position of the top side of the source" ).particle_source_y_top = "0.6"
        obj.addProperty(
            "App::PropertyString",
            "particle_source_z_near",
            "Position",
            "Position of the near side of the source" ).particle_source_z_near = "0.4"
        obj.addProperty(
            "App::PropertyString",
            "particle_source_z_far",
            "Position",
            "Position of the far side of the source" ).particle_source_z_far = "0.6"
        obj.addProperty(
            "App::PropertyString",
            "particle_source_mean_momentum_x",
            "Momentum",
            "Mean momentum in X direction" ).particle_source_mean_momentum_x = "1.0"
        obj.addProperty(
            "App::PropertyString",
            "particle_source_mean_momentum_y",
            "Momentum",
            "Mean momentum in Y direction" ).particle_source_mean_momentum_y = "1.0"
        obj.addProperty(
            "App::PropertyString",
            "particle_source_mean_momentum_z",
            "Momentum",
            "Mean momentum in Z direction" ).particle_source_mean_momentum_z = "1.0"
        obj.addProperty(
            "App::PropertyString",
            "particle_source_temperature",
            "Momentum",
            "Temperature" ).particle_source_temperature = "1.0"
        obj.addProperty(
            "App::PropertyString",
            "particle_source_charge",
            "Particle properties",
            "Particles' charge" ).particle_source_charge = "1.0"
        obj.addProperty(
            "App::PropertyString",
            "particle_source_mass",
            "Particle properties",
            "Particles' mass (calculated automatically from q and q/m)" ).particle_source_mass = "1.0"
        obj.addProperty(
            "App::PropertyString",
            "charge_to_mass_ratio",
            "Particle properties",
            "Particles' charge to mass ratio" ).charge_to_mass_ratio = "1.0"
        obj.ViewObject.addProperty("App::PropertyColor", "Color",
                                   "Spatial mesh", "Volume box color").Color=(0.0, 0.0, 1.0)
        obj.Proxy = self
        obj.ViewObject.Proxy = self
        self.doc_object = obj
        self.view_object = obj.ViewObject

    def execute( self, obj ):
        '''Executed when document is recomputated. This method is mandatory'''
        dt = self.get_time_step( obj )
        N = int( obj.particle_source_particles_to_generate_each_step )
        parameters_set = obj.getPropertyByName("set_of_parameters")
        
        if parameters_set == "Particles' charge":
            # todo: make certain fields read-only
            # e.g., obj.setEditorMode( "current", readonly )
            q = float( obj.particle_source_charge )
            I = q * N / dt
            obj.current = str( I )
        elif parameters_set == "Source current":
            I = float( obj.current )
            q = I * dt / N        
            obj.particle_source_charge = str( q )
            
        q_to_m = float( obj.charge_to_mass_ratio )
        m = 1 / q_to_m * q
        obj.particle_source_mass = str( m )
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
        x0 = float( obj.getPropertyByName("particle_source_x_left") )
        y0 = float( obj.getPropertyByName("particle_source_y_bottom") )
        z0 = float( obj.getPropertyByName("particle_source_z_near") )
        xlen = float( obj.getPropertyByName("particle_source_x_right") ) - x0
        ylen = float( obj.getPropertyByName("particle_source_y_top") ) - y0
        zlen = float( obj.getPropertyByName("particle_source_z_far") ) - z0
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
        "Return the name of the default display mode. It must be defined in getDisplayModes."
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
        conf_part.append( "[Source.{0}]\n".format( source_name ) )
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

        comments = [ "set_of_parameters", "charge_to_mass_ratio",
                     "current" ]
        for x in comments:
            conf_part.append( ";{0} = {1}\n".format( x, self.doc_object.getPropertyByName( x ) ) )

        conf_part.append("\n")
        return conf_part



class InnerRegionBoxConfigPart():
    """Box inner region"""

    def __init__( self, obj ):
        obj.addProperty(
            "App::PropertyString", "inner_region_box_x_left",
            "Position", "Box left side position" ).inner_region_box_x_left = "0.1"
        obj.addProperty(
            "App::PropertyString", "inner_region_box_x_right",
            "Position", "Box right side position" ).inner_region_box_x_right = "0.9"
        obj.addProperty(
            "App::PropertyString", "inner_region_box_y_bottom",
            "Position", "Box bottom side position" ).inner_region_box_y_bottom = "0.1"
        obj.addProperty(
            "App::PropertyString", "inner_region_box_y_top",
            "Position", "Box top side position" ).inner_region_box_y_top = "0.9"
        obj.addProperty(
            "App::PropertyString", "inner_region_box_z_near",
            "Position", "Box near side position" ).inner_region_box_z_near = "0.1"
        obj.addProperty(
            "App::PropertyString", "inner_region_box_z_far",
            "Position", "Box far side position" ).inner_region_box_z_far = "0.2"
        obj.addProperty(
            "App::PropertyString", "inner_region_box_potential",
            "Potential", "Inner region potential" ).inner_region_box_potential = "0.0"                
        obj.ViewObject.addProperty("App::PropertyColor", "Color",
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
        x0 = float( obj.getPropertyByName("inner_region_box_x_left") )
        y0 = float( obj.getPropertyByName("inner_region_box_y_bottom") )
        z0 = float( obj.getPropertyByName("inner_region_box_z_near") )
        xlen = float( obj.getPropertyByName("inner_region_box_x_right") ) - x0
        ylen = float( obj.getPropertyByName("inner_region_box_y_top") ) - y0
        zlen = float( obj.getPropertyByName("inner_region_box_z_far") ) - z0
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
        conf_part.append( "[Inner_region_box.{0}]\n".format( self.doc_object.getPropertyByName( "Label" ) ) )
        export_property_names = [ "inner_region_box_x_left", "inner_region_box_x_right",
                                  "inner_region_box_y_bottom", "inner_region_box_y_top",
                                  "inner_region_box_z_near", "inner_region_box_z_far",
                                  "inner_region_box_potential" ]
        for x in export_property_names:
            conf_part.append( "{0} = {1}\n".format( x, self.doc_object.getPropertyByName( x ) ) )
        conf_part.append("\n")
        return conf_part



class InnerRegionSTEPConfigPart():
    """STEP inner region"""

    def __init__( self, obj ):
        obj.addProperty(
            "App::PropertyLink", "STEP_model",
            "Model", "Link to STEP object" )
        obj.addProperty(
            "App::PropertyString", "inner_region_STEP_file",
            "Model", "STEP object export filename" ).inner_region_STEP_file = "model.stp"
        obj.addProperty(
            "App::PropertyString", "inner_region_STEP_potential",
            "Potential", "Inner region potential" ).inner_region_STEP_potential = "0.0"                
        # obj.ViewObject.addProperty("App::PropertyColor", "Color",
        #                            "Inner region color", "Inner region color").Color=(0.5, 0.5, 0.0)        
        obj.Proxy = self
        obj.ViewObject.Proxy = self
        self.doc_object = obj
        self.view_object = obj.ViewObject

    def execute(self, fp):
        return

    def attach(self, obj):
        return

    def updateData(self, obj, prop ):
        return
    
    # def getDisplayModes(self,obj):
    #     "Return a list of display modes."
    #     modes=[]
    #     return modes
 
    # def getDefaultDisplayMode(self):
    #     "Return the name of the default display mode. It must be defined in getDisplayModes."
    #     return 

    # def setDisplayMode(self,mode):
    #     return mode

    def onChanged(self, vp, prop):
        "Executed if any property is changed"
        return
    
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
        conf_part.append( "[Inner_region_STEP.{0}]\n".format( self.doc_object.getPropertyByName( "Label" ) ) )
        export_property_names = [ "inner_region_STEP_file",
                                  "inner_region_STEP_potential" ]
        for x in export_property_names:
            conf_part.append( "{0} = {1}\n".format( x, self.doc_object.getPropertyByName( x ) ) )
        conf_part.append("\n")
        return conf_part

    def export_step_model( self, export_dir ):
        export_model_name = os.path.join( export_dir,
                                          self.doc_object.getPropertyByName( "inner_region_STEP_file" ) )
        export_model = self.doc_object.getPropertyByName( "STEP_model" )
        if export_model is not None:
            export_models = [ export_model ]
            Part.export( export_models, export_model_name )
        return

        
FreeCADGui.addCommand( 'CreateEpicfConfig', CreateEpicfConfig() )
FreeCADGui.addCommand( 'AddSourceRegion', AddSourceRegion() )
FreeCADGui.addCommand( 'AddInnerRegionBox', AddInnerRegionBox() )
FreeCADGui.addCommand( 'AddInnerRegionSTEP', AddInnerRegionSTEP() )
FreeCADGui.addCommand( 'GenerateConfFile', GenerateConfFile() )
