class EpicfWorkbench (Workbench):
        
    MenuText = "Epicf"
    ToolTip = "FreeCAD config generator for epicf"
    #Icon = """paste here the contents of a 16x16 xpm icon"""
    Icon = """
        /* XPM */
        static const char *test_icon[]={ 
        "16 16 2 1",
        "a c #000000",
        ". c None",
        "................",
        "................",
        ".########.......",
        ".########...###.",
        ".########..##.##",
        ".###.......##...",
        ".###.......##...",
        ".#######...##...",
        ".#######.######.",
        ".###.....######.",
        ".###.......##...",
        ".###.......##...",
        ".########..##...",
        ".########..##...",
        ".########..##...",
        "................"};
        """
        
    def Initialize(self):
        "Executed when FreeCAD starts"
        # import all FreeCAD commands defined by module
        # and append them to Toolbar and Menu
        import Commands
        self.cmd_list = [ "CreateEpicfConfig",
                          "AddSourceRegion", "AddInnerRegionBox",
                          "AddInnerRegionSTEP",
                          "GenerateConfFile" ]        
        self.appendToolbar( "Epicf commands", self.cmd_list )
        self.appendMenu( "Epicf", self.cmd_list )
            
    def Activated(self):
        "Executed when the workbench is activated"        
        return
        
    def Deactivated(self):
        "Executed when the workbench is deactivated"        
        return
        
    def ContextMenu(self, recipient):
        "Executed whenever the user right-clicks on screen"
        # add commands to the context menu
        self.appendContextMenu("Epicf", self.cmd_list) 
            
    def GetClassName(self): 
        # this function is mandatory if this is a full python workbench
        return "Gui::PythonWorkbench"
        
Gui.addWorkbench( EpicfWorkbench() )
