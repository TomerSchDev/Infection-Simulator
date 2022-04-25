# in order to create exe file pyinstaller --noconfirm --onofile --console  infectionSimulator.py


import pygame
import matplotlib.pyplot as plt
import csv
import random
import time
from tqdm import tqdm
import os

# delete all the auto comments that the packages are sanding in the terminal
os.system('cls' if os.name == 'nt' else 'clear')
pygame.init()
# const parameters for the simulation
HEALTHY = 0
INFECTED = 1
RECOVER = 2

HEIGHT = 800
WIDTH = 600

WHITE = (255, 255, 255)
FPS = 60
ROWS = COLS = 200
EXTENSION_FOR_TEXT = 100
HEADERS = [("Healthy", (0, 255, 0)), ("Infected", (255, 0, 0)), ("Recovered", (0, 0, 255))]


class Cell:
    """
    this class represents a cell that lives in the matrix of the simulation and can get infected
    """

    def __init__(self, state, pos):
        self.state = state
        self.pos = pos
        self.time = 0
        self.size = WIDTH // 200
        self.turn = 0
        self.got_infected = False
        self.move_range = range(-1, 2, 1)
        self.move_set = self.calculate_move_set()
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]

    def calculate_move_set(self):
        """
        calculate the move set for the cell
        :return: list of moves for the cell
        """
        move_set = []
        for row in self.move_range:
            for col in self.move_range:
                if row == 0 and col == 0: continue
                move_set.append((row, col))
        return move_set

    def render(self, win, distance):
        """
        render the cell to the screen
        :param win: the screen we want to render into
        :param distance: the distance of each cell from one to another
        :return: nothing
        """
        color = self.colors[self.state]  # the color of the cell depending on the state of the cell
        y, x = self.pos
        pygame.draw.rect(win, color, (x * distance, y * distance + EXTENSION_FOR_TEXT, self.size, self.size))

    def get_moves(self):
        # calculate moves for the cell based on its location and move set
        r, c = self.pos
        moves = []
        for row, col in self.move_set:
            moves.append(((r + row) % ROWS, (c + col) % COLS))
        return moves

    def update(self, simulate, gen):
        self.turn = gen

        # if the cell recovered, he won't do anything
        if self.state == RECOVER: return
        if self.state == HEALTHY and not self.got_infected: return

        # if the cell healthy he needs to check if he got infected
        if self.state == HEALTHY and self.got_infected:
            self.state = INFECTED
            self.time = gen
            return
        # if cell is not recovered and not healthy he must be infected
        row, col = self.pos
        neighbors = []
        # gets all the 8 neighbors of the cell to get them infected
        for i in range(-1, 2, 1):
            n_r = (i + row) % ROWS
            for j in range(-1, 2, 1):
                if j == 0 and i == 0: continue
                n_c = (j + col) % COLS
                other_cell = simulate.matrix[n_r][n_c]
                # checks to remove the cells that are not gone get infected (they are recovered or infected already)
                if other_cell is None or other_cell.state != HEALTHY or other_cell.got_infected: continue
                neighbors.append(other_cell)
        chances = [random.choices((0, 1), weights = [1 - simulate.p, simulate.p]) for x in neighbors]
        for chance, other_cell in zip(chances, neighbors):
            # checks if the other cell got infected
            if chance[0]: other_cell.set_infected(gen)
        # checks if the cell recovered on this turn
        if simulate.time - self.time >= simulate.recovery_time:
            self.state = RECOVER

    def move(self, moves):
        """
        this function moves the cell to different location on the matrix
        :param moves: the valid moves that the cell can move
        :return: the new location of the cell
        """
        # adding the option to stay in place
        moves.append(self.pos)
        # selecting randomly a place to move
        selected_move = moves[random.randint(0, len(moves) - 1)]
        self.pos = selected_move
        return selected_move

    def set_infected(self, gen):
        """
        setting the cell to get infected in the end of his turn
        :param gen: the generation number the cell got infected
        :return: nothing
        """
        self.got_infected = True
        # if the cell already did it turn, if so it needs to update now
        if self.turn == gen:
            self.state = INFECTED
            self.turn = gen


class jumpingCell(Cell):
    """
    class that represents subclass of Cell that are capable of jumping to distance of 10 instead of 1
    """

    def __init__(self, state, pos):
        super().__init__(state, pos)
        self.move_range = range(-10, 11, 1)
        self.move_set = self.calculate_move_set()
        # lowering his rendering size in order to different from normal cells
        self.size = 1

    def render(self, win, distance):
        """
                render the cell to the screen
                :param distance: the distance of each cell from one to another
                :param win: the screen we want to render into
                :return: nothing
                """
        color = self.colors[self.state]  # the color of the cell depending on the state of the cell
        y, x = self.pos
        pygame.draw.rect(win, color, (
            x * distance + self.size, y * distance + self.size + EXTENSION_FOR_TEXT, self.size, self.size))


class Simulation:
    """
    class that represents us each simulation we do.
    """

    def __init__(self, parameters):
        self.rows = ROWS
        self.cols = COLS
        self.time = 0
        self.cells = []
        self.matrix = create_matrix(self.rows, self.cols)
        # init parameters for simulation for parameters dictionary
        self.name = parameters["simulation name"]
        self.population = float(parameters["population size"])
        self.recovery_time = int(parameters["recovery time"])
        self.p1 = float(parameters["P1"])
        self.p2 = float(parameters["P2"])
        # setting so ps[0] is the lower percent and ps[1] is the higher percent.
        self.ps = (max(self.p1, self.p2), min(self.p1, self.p2))
        self.p = max(self.ps)
        self.t = float(parameters["T"])
        self.r = float(parameters["R"])
        self.start_sick = float(parameters["infected population"])
        self.infected_cells = 0
        self.init_simulation()
        # checking which percent the simulation starts with
        self.update_p()
        self.info = []

    def update_p(self):
        """
        checks the number of infected cells is bigger than T and update p accordingly.
        :return: nothing
        """
        if self.infected_cells >= self.t * self.population:
            self.p = self.ps[1]
        else:
            self.p = self.ps[0]

    def get_valid_moves(self, moves, new_gen):
        """
        checks what moves are valid for the cell to down.
        :param moves: list of moves that the cell can move to.
        :param new_gen: the matrix of the new generation to check if the place got taken before in this turn.
        :return:list of valid moves that the cell can go into
        """
        valid = []
        for move in moves:
            row, col = move
            # if in this generation and the next generation the place is empty then the move is valid
            if self.matrix[row][col] is None and new_gen[row][col] is None:
                valid.append(move)
        return valid

    def simulate(self, win, large_font, small_font):
        """
        Doing the actual simulation
        :param win: the screen we want to render into
        :param large_font: the large font object for the text
        :param small_font: the small font object for the text
        :return: information about the simulation
        """
        clock = pygame.time.Clock()
        run = True
        while run:
            # making sure the simulation not run to fast
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
            self.render(win, large_font, small_font)
            self.update()
            # checks if there is infected cells in the simulation and if so will continue the simulation.
            run = self.infected_cells > 0
        return self.info

    def update(self):
        self.time += 1
        sick_num = 0
        # creating new empty matrix for the new generation
        new_gen = create_matrix(self.rows, self.cols)
        for cell in self.cells:
            # summing up all the infected cells to know when to stop the simulation
            if cell.state == INFECTED:
                sick_num += 1
            # gets all the valid moves the cell can go
            valid_moves = self.get_valid_moves(cell.get_moves(), new_gen)
            n_r, n_c = cell.move(valid_moves)
            new_gen[n_r][n_c] = cell
            # calling the cell to update
            cell.update(self, self.time)
        # updating the matrix with the new generation matrix
        self.matrix = new_gen
        self.infected_cells = sick_num
        self.update_p()
        self.info.append(sick_num)

    def render(self, win, large_font, small_font):
        """
        rendering the simulation to the window
        :param win: the window object to render into.
        :param large_font: the small font object for the text to render.
        :param small_font: the large font object for the text to render.
        :return:nothing
        """
        # rendering the screen with white color
        pygame.draw.rect(win, WHITE, (0, 0, WIDTH, HEIGHT))
        # render simulation name
        simulation_name_text = large_font.render("Simulation Name : " + self.name, 1, (100, 100, 100))
        win.blit(simulation_name_text, ((WIDTH - simulation_name_text.get_width()) // 2, 0))
        # how spaces is the headers for each other
        different = 40
        x = 0
        # blit legend to the window
        for text, color in HEADERS:
            text_box = large_font.render(text, 1, color)
            win.blit(text_box, (x + different, EXTENSION_FOR_TEXT // 2))
            x += different + text_box.get_width()
        # blit generation number to window
        gen_text = large_font.render(f"Generation number : {self.time}", 1, (100, 100, 100))
        win.blit(gen_text, ((WIDTH - gen_text.get_width()) // 2, EXTENSION_FOR_TEXT + 200 * (WIDTH // 200)))
        # transform the parameters to percent
        if self.start_sick >= 1:
            self.start_sick /= self.population
        if self.r >= 1:
            self.r /= self.population
        self.population = int(self.population)
        self.recovery_time = int(self.recovery_time)
        # blit the parameters to the window
        parameters = f"N = {self.population}, D = {self.start_sick} , R = {self.r}, X = {self.recovery_time}," \
                     f" P1 = {self.ps[0]}, P2 = {self.ps[1]}, T = {self.t}"
        parameters_text = small_font.render(parameters, 1, (100, 100, 100))
        win.blit(parameters_text, ((WIDTH - parameters_text.get_width()) // 2, HEIGHT - 50))
        # rendering each cell into the window
        for cell in self.cells:
            cell.render(win, WIDTH // self.cols)
        pygame.display.update()

    def init_simulation(self):
        """
        initialize the simulation with all its cells
        :return: nothing
        """
        # if we get for population number a percent of the space and not an amount number
        if 1 > self.population > 0:
            self.population = int(self.rows * self.cols * self.population)
        # if we get for infected population number a percent of the population and not an amount number
        start_sick = self.start_sick
        if 1 > start_sick > 0:
            start_sick = int(self.population * start_sick)
        # if we get for R size number a percent of the population and not an amount number
        start_speedy = self.r
        if 1 >= start_speedy > 0:
            start_speedy = int(self.population * start_speedy)
        # getting random numbers to represent the place in the matrix the cell get
        cells_pos = random.sample(range(self.rows * self.cols), int(self.population))
        # getting random numbers to represent the indexes of the different cells
        sick_cells = random.sample(range(int(self.population)), int(start_sick))
        speedy_cells = random.sample(range(int(self.population)), int(start_speedy))
        self.time = 0
        self.infected_cells = start_sick
        print("creating simulation...")
        # looping over all the random numbers and assign them
        for index in tqdm(range(len(cells_pos))):
            # getting the cell location
            pos = cells_pos[index]
            row = pos // self.rows
            col = pos % self.cols
            # getting the cell starting state
            state = HEALTHY
            if index in sick_cells:
                state = INFECTED
            # getting the cell type
            if index in speedy_cells:
                cell = jumpingCell(state, (row, col))
            else:
                cell = Cell(state, (row, col))
            # adding the cell to the matrix and the cells list
            self.matrix[row][col] = cell
            self.cells.append(cell)


def readParameters():
    """
    this function reads the parameters from the csv file.
    if csv file is not available it will generate default parameters
    :return: list of dictionary that each dictionary contains the parameters of the simulation
    """
    paths = ["parameters.csv"]
    parameters = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, 'r') as csv_file:
            header = ["simulation name", "population size", "infected population", 'R', 'T', 'P1', 'P2',
                      "recovery time"]
            csv_reader = csv.DictReader(csv_file, fieldnames = header, delimiter = ',')
            simulation_file = []
            for simulation_row in csv_reader:
                # checks if the simulation row is not empty
                skip = False
                for key in simulation_row.items():
                    if key[0] != "simulation name" and key[1] == '':
                        skip = True
                        break
                if skip:
                    continue
                simulation_file.append(simulation_row)
            simulation_file.remove(simulation_file[0])
            parameters.extend(simulation_file)
    # if there is no parameters then it will generate default parameters
    if not parameters:
        print("not found any parameters files....")
        print("using default parameters for simulation")
        default = {
            "simulation name": "Default simulation", "population size": 0.4, "infected population": 0.001, "R": 0.1,
            "T": 0.21,
            "P1": 0.1,
            "P2": 0.05, "recovery time": 15
        }
        parameters.append(default)
    return parameters


def create_matrix(rows, cols):
    matrix = []
    for r in range(rows):
        row = []
        for col in range(cols):
            row.append(None)
        matrix.append(row)
    return matrix


def main():
    """
    the main function
    :return: nothing
    """
    # gets the simulations parameters from the csv file
    simulations_parameters = readParameters()

    for parameters in simulations_parameters:
        sim_name = parameters["simulation name"]
        print("Starting Simulation: " + sim_name)
        # creating the simulation for given parameters
        s = Simulation(parameters)
        # initialize all pygame parameters
        pygame.init()
        win = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Infection Simulation")
        large_font = pygame.font.SysFont('comicsans', 35)
        small_font = pygame.font.SysFont('comicsans', 13)
        # simulate the simulation and getting the information of the simulation
        info = s.simulate(win, large_font, small_font)
        if info is None:
            return
        # plot the simulation into a plot
        plotInfo(sim_name, info)
        pygame.quit()


def plotInfo(simulation_name, info):
    """
    plotting the information that the simulation returned
    :param simulation_name: the simulation name we got from the csv file
    :param info: the information we want to plot from the simulation
    :return: nothing
    """
    fig = plt.gcf()
    plt.plot(info, color = 'r')
    plt.xlabel("generation")
    plt.ylabel("Infected population")
    plt.title(simulation_name)
    plt.show(block = True)
    user_answer = ''
    # asking the user if he wants to save the plot
    while user_answer != 'y' and user_answer != 'n':
        answer = input("Do you want to save " + simulation_name + " plot into PDF file? (y/n)\n")
        user_answer = answer.lower()
        if user_answer == "stop":
            quit()
        if user_answer != "y" and user_answer != "n":
            print("Please only enter y or n")
    if user_answer == 'y':
        # saves the plot into pdf file with the name of the simulation
        path = simulation_name.replace(":", "-")
        fig.savefig(path + ".pdf")


if __name__ == "__main__":
    print("Welcome to Infection Simulator!")
    main()
