class Target(object):
    def __init__(self):
        self.name = "(unnamed target)"
        self.boards = []
        self.total_population_1d_count = 0
        self.total_population_2d_count = 0

class Board(object):
    def __init__(self):
        self.name = "(unnamed board)"
        self.dv_count = 0
        self.input_dv_count = 0
        self.first_input_dv_index = 0
        self.population_1d_count = 0
        self.population_2d_count = 0
        self.outputs = []
        self.inputs = []
        self.controls = []
        self.ios = []

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
    def __init__(self):
        self.name = "(unnamed control)"
        self.type = ""

class IO(object):
    def __init__(self):
        self.name = "(unnamed I/O)"
        self.type = ""
        self.input = ""
        self.output = ""
