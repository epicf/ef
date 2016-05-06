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
        "..############..",
        "..############..",
        "..############..",
        "......####......",
        "......####......",
        "......####......",
        "......####......",
        "......####......",
        "......####......",
        "......####......",
        "......####......",
        "......####......",
        "................",
        "................"};
        """
        
    def Initialize(self):
        "This function is executed when FreeCAD starts"
        # import here all the needed files that create your FreeCAD commands
        import Commands
        # A list of command names created in the line above
        self.cmd_list = [ "My_Command", "CreateEpicfConfig",
                          "AddSourceRegion", "AddInnerRegionBox",
                          "GenerateConfFile" ]
        # creates a new toolbar with your commands
        self.appendToolbar("Epicf commands", self.cmd_list) 
        self.appendMenu("Epicf", self.cmd_list) # creates a new menu
        # appends a submenu to an existing menu
        #self.appendMenu(["An existing Menu","My submenu"], self.cmd_list) 
        #Log ("Loading MyModule... done\n")
            
    def Activated(self):
        "This function is executed when the workbench is activated"
        FreeCAD.Console.PrintMessage('Hello, World!')
        return
        
    def Deactivated(self):
        "This function is executed when the workbench is deactivated"
        FreeCAD.Console.PrintMessage('Bye, World!')
        return
        
    def ContextMenu(self, recipient):
        "This is executed whenever the user right-clicks on screen"
        # "recipient" will be either "view" or "tree"
        # add commands to the context menu
        self.appendContextMenu("Epicf", self.cmd_list) 
            
    def GetClassName(self): 
        # this function is mandatory if this is a full python workbench
        return "Gui::PythonWorkbench"
        
Gui.addWorkbench( EpicfWorkbench() )
