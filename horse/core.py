import pandas as pd
import matplotlib.pyplot as plt
import pulp as pl
from matplotlib.patches import Rectangle
import numpy as np

#helper to navigate the grid constraints.
def get_neighbors(row, col, grid_shape):
    """
    Returns the neighbor coordinates for a node in a grid. Accounts for edges.

    Parameters:
    -----------

    row: int
        The row index of the square whose neighbors to get.

    col: int
        The column index of the square whose neighbors what to get for, at

    grid_shape: (int,int)
        The shape of the grid.
    """

    neighbors = {}
    no_rows, no_cols = grid_shape
    
    #get locations above
    if row != 0:
        neighbors.update({
            "N":(row - 1, col)
        })
    #get locations below
    if row != no_rows - 1:
        neighbors.update({
            "S":(row + 1, col)
        })
    #get locations to the left
    if col != 0:
        neighbors.update({
            "W":(row, col - 1)
        })
    #get locations to the right
    if col != no_cols - 1:
        neighbors.update({
            "E":(row, col + 1)
        })

    return neighbors

def get_opposite_direction(direction):
    if direction == "N":
        return "S"
    elif direction == "S":
        return "N"
    elif direction == "W":
        return "E"
    elif direction == "E":
        return "W"




### solve

##############################################################
# retrieve puzzle from a csv
#######################

class HorseSolver:

    day_no = input("What horse day? ")
    day_file_path = f"./maps/day{day_no}.csv"
    
    legend = {
        'buildable':0,
        'water':1,
        "horse":'s', #it'S a horse. S'horse. Source. s.
        'cherry':'c', # +3 points. The cherry square is worth 4 total.
        'portal':'P_X', #P_1a, P_1b, P_2a, P_2b, ...
    }
    
    wall_budget = input("Wall budget: ")
    wall_budget = int(wall_budget)
    
    try:
        puzz = pd.read_csv(day_file_path, header=None, dtype = str)
    except:
        raise ValueError(f"day{day_no}.csv not found.")
    print(f"Found day{day_no}.csv")
    
    
    puzz.fillna("0", inplace = True)
    
    #note the source horse location.
    [[s_row, s_col]] = np.argwhere(puzz == 's')
    s_row, s_col = int(s_row), int(s_col)
    
    #get cherry locations
    cherry_spots = np.argwhere(puzz == 'c')
    
    #get portal locations
    portal_spots = np.argwhere(puzz == 'P1')
    
    water = puzz.replace(
        {
            's':0,
            'c':0,
            'P1':0
        }
    ).astype(int).to_numpy()
    
    #plot the unsolved puzzle
    fig, ax = plt.subplots()
    
    ax.set_facecolor("#55FFAB")
    
    water_masked = np.ma.masked_where(water == 0, water)
    
    ax.imshow(water_masked, vmin = 0, vmax = 1, cmap = "Blues")
    
    print(f"Adding horse at {(s_col-.5,s_row-.5)}")
    #print da horse
    ax.add_patch(
        Rectangle(
            (s_col-.5,s_row-.5),
            1,
            1,
            color = 'white'
        )
    )
    
    #print tha cherries
    for (cherry_row, cherry_col) in cherry_spots:
        print(f"adding red cherry at {int(cherry_row),int(cherry_col)}")
        ax.add_patch(
            Rectangle(
                (
                    cherry_col-0.5,
                    cherry_row-0.5
                ),
                1,
                1,
                color = 'red'
            )
        )
    
    #print portals
    for (portal_row, portal_col) in portal_spots:
        print(f"adding red cherry at {int(portal_row),int(portal_col)}")
        ax.add_patch(
            Rectangle(
                (
                    portal_col-0.5,
                    portal_row-0.5
                ),
                1,
                1,
                color = 'cyan'
            )
        )
        
    plt.show()
    
    
    # create a water array. 1s for water, 0s for not water.
    # water = puzz.replace({'s':0}).astype(int).to_numpy()
    no_rows, no_cols = list(water.shape)
    
    #create model
    model = pl.LpProblem("Enclose.Horse", pl.LpMaximize)
    
    
    
    ####################################################
    # Create variables #
    ####################
    
    #whether to build a wall at a node
    build_wall = {
        (row, col):pl.LpVariable(
            f"wall_{row}_{col}",
            0,
            1,
            cat = "Binary"
        )
        for col in range(no_cols)
        for row in range(no_rows)
    }
    
    #define reachability from source as a variable
    reachable = {
        (row,col):pl.LpVariable(
            f"reach_{row}_{col}",
            0,
            1, 
            cat = "Binary"
        )
        for col in range(no_cols)
        for row in range(no_rows)
    }
    
    #create a flow variable for each square edge
    #flow is outward --- as divergence
    flow = {
        (row, col, direction):pl.LpVariable(
            f"flow{row}_{col}_{direction}",
            -no_rows*no_cols,
            no_rows*no_cols,
            cat = "Continuous"
        )
        for col in range(no_cols)
        for row in range(no_rows)
        for direction in ["N","S","E","W"]
    }
    
    
    
    ###########################################
    #set objective and rules#
    #########################
            
    #objective: maximize reachability
    # model += pl.lpSum(reachable.values())
    model.setObjective(
        pl.lpSum(reachable.values())
        + pl.lpSum(
            reachable[(int(cherryx), int(cherryy))] * 3
            for cherryx, cherryy in cherry_spots
        )
    )
            
    #force reachability at the source
    model += reachable[(s_row, s_col)] == 1
    #force flow at the source to power all reachable squares
    model += flow[(s_row, s_col, "N")] +flow[(s_row, s_col, "S")] +flow[(s_row, s_col, "W")] +flow[(s_row, s_col, "E")] == pl.lpSum(reachable.values()) - 1
    
    
    # #force no wall at the source
    model += build_wall[(s_row, s_col)] == 0
    #enforce wall budget
    model += pl.lpSum(build_wall.values()) <= wall_budget
    
    
    #no walls on cherries!
    for cherryrow, cherrycol in cherry_spots:
        model += build_wall[(cherryrow,cherrycol)] == 0
    #no walls on rows!
    for portalrow, portalcol in portal_spots:
        model += build_wall[(portalrow, portalcol)] == 0
    
    ###########################################
    #reachability constraints#
    ###########################
    
    #Do these to every square:
    for col in range(no_cols):
        for row in range(no_rows):
            #constrain whether it's reachable or has a wall. Never both.
            model += reachable[(row,col)] + build_wall[(row,col)] <= 1
            #water spots only!...
            if water[row,col]:
                #force nonreachability to water spots
                model += reachable[(row,col)] == 0
                #force nonbuildability to water spots
                model += build_wall[(row,col)] == 0
                #force zero flow on water spots
                model += flow[(row,col,"N")] == 0
                model += flow[(row,col,"S")] == 0 
                model += flow[(row,col,"E")] == 0 
                model += flow[(row,col,"W")] == 0 
            #enforce nonreachability to all edges
            if row == 0 or row == no_rows-1 or col == 0 or col == no_cols-1:
                model += reachable[(row,col)] == 0
                
            #get neighbor indices to prepare flow constraints
            neighbors = get_neighbors(row, col, (no_rows, no_cols))
            no_neighbors = len(neighbors)
            #no loss through pipes between squares. Doubles the variables necessary, but hey!
            for direction, (i, j) in neighbors.items():
                opposite_direction = get_opposite_direction(direction)
                model += flow[(row, col, direction)] == -flow[(i, j, opposite_direction)]
            #loss within a square. maybe restructure the flow variables so i can just lpSum.
            # -1 because it's outward flow, like divergence.
            if (row, col) != (s_row, s_col):
                model += flow[(row,col,"N")] + flow[(row,col,"S")] + flow[(row,col,"E")] + flow[(row,col,"W")] == -reachable[(row, col)]
            #flow only through reachable squares
            M = no_rows * no_cols
            for dir in ["N","S","E","W"]:
                model += flow[(row,col,dir)] <=  M * reachable[(row,col)]
                model += flow[(row,col,dir)] >= -M * reachable[(row,col)]
                
    #intersquare constraints
    for col in range(no_cols-1):
        for row in range(no_rows-1):
            #two adjacent BUILDABLE squares cannot both be reached if a wall occupies either of them
            if not water[(row,col)]:
                if not water[(row,col+1)]:
                    model += reachable[(row,col+1)] - reachable[(row,col)] <= build_wall[(row,col)] + build_wall[(row, col+1)]
                    model += reachable[(row,col)] - reachable[(row,col+1)] <= build_wall[(row,col)] + build_wall[(row, col+1)]
                if not water[(row+1,col)]:
                    model += reachable[(row+1,col)] - reachable[(row,col)] <= build_wall[(row,col)] + build_wall[(row+1, col)]
                    model += reachable[(row,col)] - reachable[(row+1,col)] <= build_wall[(row,col)] + build_wall[(row+1, col)]
    
    ############################################
    #########################################
    #portals --- this doesn't work correctly!!!
    ####################### ################
    ##########################################
    
    p1a, p1b = portal_spots
    model += reachable[p1a[0],p1a[1]] == reachable[p1b[0], p1b[1]]
    # model += flow[p1a] == flow[p1b]
    
    ########################################
    # Solve #####
    #############
    
    model.solve(pl.PULP_CBC_CMD(msg = True))
    
    #make an np array of the resulting pulp variables
    wall_np = np.array([
        [
            pl.value(
                build_wall[(row,col)]
            )
            for col in range(no_cols)
        ]
            for row in range(no_rows)
    ])
    
    reachable_np = np.array([
        [
            pl.value(
               reachable[(row,col)]
            )
            for col in range(no_cols)
        ]
            for row in range(no_rows)
    ])
    
    
    #######################################
    #plot the result#
    #################
    
    fig, ax = plt.subplots()
    
    ax.set_facecolor("#55FFAB")
    
    water_masked = np.ma.masked_where(water == 0, water)
    wall_masked = np.ma.masked_where(wall_np == 0, wall_np)
    reachable_masked = np.ma.masked_where(reachable_np == 0, reachable_np)
    
    #print tha cherries
    for (cherry_row, cherry_col) in cherry_spots:
        print(f"adding red cherry at {int(cherry_row),int(cherry_col)}")
        ax.add_patch(
            Rectangle(
                (
                    cherry_col-0.5,
                    cherry_row - 0.5
                ),
                1,
                1,
                color = 'red'
            )
        )
    
    ax.imshow(water_masked, vmin = 0, vmax = 1, cmap = "Blues")
    ax.imshow(wall_masked, vmin = 0, vmax = 1, cmap = "turbo")
    ax.imshow(reachable_masked, vmin=0, vmax=1)
    
    #print da horse
    ax.add_patch(
        Rectangle(
            (s_col-.5,s_row-.5),
            1,
            1,
            color = 'white'
        )
    )
    
    plt.show()
    
    model_status = pl.LpStatus[model.status] 
    
    assert model_status == 'Optimal', f"Model status: {model_status}"