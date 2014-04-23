import xml.etree.ElementTree as ET

def getkey(attrib, key, default):
    if key in attrib:
        return attrib[key]
    else:
        return default

class Target(object):
    def __init__(self, targetFile=None):
        self.boards = []
        self.total_population_1d_count = 0
        self.total_population_2d_count = 0
        if targetFile is None:
            self.name = "(unnamed target)"
        else:
            self.parseTargetFile(targetFile)

    def parseTargetFile(self, targetFile):
        tree = ET.parse(targetFile)
        root = tree.getroot()
        if root.tag != 'target':
            raise ValueError(targetFile + " does not look like a well-formed target file;"
                             " root element is '" + root.tag + "' but should be 'target'")
        
        self.name = getkey(root.attrib, 'name', targetFile)
        for child in root:
            if child.tag == 'board':
                board = Board(child)
                self.boards.append(board)
                self.total_population_1d_count += board.population_1d_count
                self.total_population_2d_count += board.population_2d_count
            else:
                raise ValueError("unknown element type " + child.tag)
        

class Board(object):
    def __init__(self, root=None):
        self.outputs = []
        self.inputs = []
        self.controls = []
        self.ios = []
        if root is None:
            self.name = "(unnamed board)"
            self.dv_count = 0
            self.input_dv_count = 0
            self.first_input_dv_index = 0 # FIXME not necessary if we decide that input always starts at 96
            self.population_1d_count = 0
            self.population_2d_count = 0
        else:
            self.parseRoot(root)

    def parseRoot(self, root):
        self.name = getkey(root.attrib, 'name', '(unnamed board)')
        self.dv_count = int(getkey(root.attrib, 'dv_count', 0))
        self.input_dv_count = int(getkey(root.attrib, 'input_dv_count', 0))
        self.first_input_dv_index = int(getkey(root.attrib, 'first_input_dv_index', 0))
        self.population_1d_count = int(getkey(root.attrib, 'population_1d_count', 0))
        self.population_2d_count = int(getkey(root.attrib, 'population_2d_count', 0))
        for child in root: # FIXME finish parsing
            if child.tag == 'output':
                pass
            elif child.tag == 'input':
                pass
            elif child.tag == 'control':
                self.controls.append(Control(child))
            elif child.tag == 'io':
                pass
            else:
                raise ValueError("unknown element type " + child.tag +
                                 " encountered while parsing board " + self.name)

class Output(object):
    def __init__(self):
        self.name = "(unnamed output)"
        self.index = 0
        self.first_dv_index = 0
        self.last_dv_index = 0

class Input(object):
    def __init__(self):
        self.name = "(unnamed input)"
        self.index = 0
        self.first_dv_index = 0
        self.last_dv_index = 0

class Control(object):
    def __init__(self, root):
        if root is None:
            self.name = "(unnamed control)"
            self.type = ""
        else:
            self.parseRoot(root)

    def parseRoot(self, root):
        self.name = getkey(root.attrib, 'name', "(unnamed control)")
        self.type = getkey(root.attrib, 'type', "")
        if self.type == '':
            raise ValueError("control type unspecified for " + self.name + 
                             " while parsing board " + self.name)
        # type-specific attrs
        if self.type == 'ethernet':
            self.mac_address = getkey(root.attrib, 'mac_address', "")
            self.device = getkey(root.attrib, 'device', "")

class IO(object):
    def __init__(self):
        self.name = "(unnamed I/O)"
        self.type = ""
        self.input = ""
        self.output = ""
