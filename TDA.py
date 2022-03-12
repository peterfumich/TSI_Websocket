# IMPORT STATEMENTS ARE THERE FOR CURRENT OR FUTURE DEPENDENCY.
import numpy as np
import pandas as pd
import requests, json
import time
import gudhi
from gudhi.representations import Landscape
from matplotlib import pyplot as plt
import csv
import os

###############################################
def Create_Data(t, time_step_int,symbol):
    delta = 1

    base_url = "https://api.gemini.com/v2"
    fail = True
    while fail == True:
        try:
            response = requests.get(base_url + "/candles/" + symbol + "/" + time_steps[time_step_int])
            fail = False
        except:
            print("Failed to requests.get candlestick data. Trying again in one minute. ")
            time.sleep(60)
            fail = True
    try:

        candle_data = response.json()
        # print(candle_data)
        prices = [x[4] for x in candle_data]  # 1 for open data 4 for closing data
        candle_volume = [x[-1] for x in candle_data]
    except:
        return([0,0],0)

    #PLOT DATA
    #plt.scatter(x, Y[::-1])
    #plt.show()
    #plt.savefig(os.path.join(final_directory,"Graph"+ str(t) + ".png"))
    #plt.close()
    # Transformed Data:
    # Z = []
    # for y in range(len(Y)-1):
    #     Z.append(Y[y]/Y[y+1])
    # trans_averages = TDA.Create_Moving_Average(Z, 0)
    # trans_slopes = TDA.Create_Slopes(Z, 0, delta)
    # transformed_array = TDA.Create_Averaging_Arrays(trans_averages, trans_slopes)
# price stuff
#     print(prices)
    averages = Create_Moving_Average(prices, time_step_int)
    slopes = Create_Slopes(prices, time_step_int, delta)
    price_averaging_array = Create_Averaging_Arrays(averages, slopes)
# Volume Stuff
    candle_volume_non_empty = [i for i in candle_volume if i > .01]
    volume_averages = Create_Moving_Average(candle_volume, time_step_int)
    volume_slopes = Create_Slopes(candle_volume, time_step_int, delta)
    volume_averaging_array = Create_Averaging_Arrays(volume_averages, volume_slopes)
    return(price_averaging_array, volume_averaging_array, prices,candle_volume_non_empty)


# THIS PROCEDURE CALCULATES THE SET OF MOVING AVERAGES  Input prices must be listed with most recent value first.
def Create_Moving_Average(prices, window_depth,time_step):#time_step in seconds
    set_of_moving_averages = [prices[0]]
    try:
        for x in range(1, window_depth):  # range(1,int(60/time_step_int)): Moving
            set_of_moving_averages.append((set_of_moving_averages[-1] * x + prices[x]) / (x + 1))
        return (set_of_moving_averages)
    except:
        return (set_of_moving_averages)
# THIS PROCEDURE CALCULATES THE SET OF SLOPES OF THE MOVING AVERAGES. Input prices must be listed with most recent value first.
def Create_Slopes(prices, delta, window_depth,time_step):
    set_of_slopes = [(prices[0] - prices[1])/(delta)]
    try:
        for y in range(1, window_depth):  # range(1,int(60/time_step_int)):
            set_of_slopes.append((prices[0] - prices[y + 1]) / (delta * (y + 1)))

        return (set_of_slopes)
    except:
        return (set_of_slopes)
# Creates the space which will be the INPUT for the Persistence Homology Procedure. OUTPUT CONSISTS OF POINTS OF THE FORM (DISTANCE BETWEEN TWO ELEMENTS OF THE MOVING AVERAGE SET, DISTANCE BETWEEN TWO ELEMENTS OF THE SLOPES SET)
def Create_Averaging_Arrays(MA_set, Slope_set):
    length = min(len(MA_set),len(Slope_set))
    MA_set = MA_set[0:length]
    Slope_set = Slope_set[0:length]
    precision = 10
    index = 0
    average_diff = []
    slope_diff = []
    for large_average in MA_set:
        for small_average_index in range(index):
            small_average = MA_set[small_average_index]
            #print(small_average-large_average)
            average_diff.append(round(small_average - large_average, precision))
        index += 1
    index = 0
    for large_slope in Slope_set:
        for small_slope_index in range(index):

            small_slope = Slope_set[small_slope_index]
            slope_diff.append(round(large_slope - small_slope, precision))
        index += 1
    output = []
    for x, avg in enumerate(average_diff):
        output.append(avg)
        output.append(slope_diff[x])
    return([average_diff,slope_diff,output])#return ([average_diff, slope_diff])

###############################################
###############################################
#Center Stuff
###############################################
###############################################

# PROCEDURE CALCULATES THE CENTER OF MASS OF THE MOVING AVERAGE SPACE. INPUT IS Create_Averaging_Arrays OUTPUT IS [HORIZONTAL COMPONENT OF CENTER OF MASS, VERTICAL COMPONENT OF CENTER OF MASS]
def Center_of_Mass(avg_diff, slope_diff):
    X = 0
    for x in avg_diff:
        X += x
    avg_center = X / len(avg_diff)
    Y = 0
    for y in slope_diff:
        Y += y
    slope_center = Y / len(slope_diff)
    return ([avg_center, slope_center])
# OUTPUT : CENTER
#
def Confidence_Weighted_Center_of_Mass(avg_diff, slope_diff):
    X = 0
    Y = 0
    for i in range(len(avg_diff)):
        try:
            weight = abs(slope_diff[i] / (avg_diff[i] ** 2 + slope_diff[i] ** 2) ** (1 / 2))
        except:
            weight = 0
        X += avg_diff[i] * weight
        Y += slope_diff[i] * weight
    avg_center = X / len(avg_diff)
    slope_center = Y / len(slope_diff)
    return ([avg_center, slope_center])
    # cwcm = [0,0]
    # for i in range(len(avg_diff)):
    #     angle = np.arctan(slope_diff[i]/avg_diff[i])
    #     confidence_value = np.cos(angle)**2
    #     cwcm = [cwcm[0]+avg_diff[i]*confidence_value,cwcm[1]+slope_diff[i]*confidence_value]
    # cwcm = [x/len(avg_diff) for x in cwcm]
    # return(cwcm)
# OUTPUT : CENTER
#
def Boundary_Box_Center(averages, slopes):
    avg_min = np.amin(averages)
    avg_max = np.amax(averages)
    slope_min = np.amin(slopes)
    slope_max = np.amax(slopes)
    return ([(avg_max + avg_min) / 2, (slope_min + slope_max) / 2])
# OUTPUT : CENTER
#
def Center_Evolution(centers,new_center):
    # Calculate angle given complex representation of coordinates.
    angle_of_center= np.angle([new_center[0]+new_center[1]*1j])
    if 0<=angle_of_center and angle_of_center<1/2*np.pi:
        center_state = [1,1]
    elif 1/2*np.pi<=angle_of_center and angle_of_center<np.pi:
        center_state = [-1,1]
    elif -np.pi<=angle_of_center and angle_of_center<-1/2*np.pi:
        angle_of_center += 2*np.pi
        center_state = [-1,-1]
    elif -1/2*np.pi<=angle_of_center<0:
        angle_of_center += 2 * np.pi
        center_state = [1,-1]
    else:
        center_state = [0,0]

    d_center = np.subtract(new_center,centers[-1])
    d_center_prev = np.subtract(centers[-1],centers[-2])
    dd_center = np.subtract(d_center,d_center_prev)
    cm_evolution_vector = np.add(d_center,1/2*dd_center)
    #expected_positions = np.add(new_center_set,cm_evolution_vectors)
    length_of_evolution_vector = np.linalg.norm(cm_evolution_vector)
    angle_of_evolution_vector = np.angle(cm_evolution_vector[0]+cm_evolution_vector[1]*1j)
    if angle_of_evolution_vector<0:
        angle_of_evolution_vector+=2*np.pi
    type_vector_angle = np.subtract(angle_of_evolution_vector,angle_of_center)
    if type_vector_angle<0:
        type_vector_angle+=2*np.pi
    return([center_state,type_vector_angle,length_of_evolution_vector])
#OUTPUT : "return([center_state,type_vector_angle,length_of_evolution_vector])"
#
#TOPOLOGICAL PROCEDURES#
#
def Diagram_to_Array(diag):
    out_array = []
    for value in diag:
        out_array.append([value[1][0], value[1][1]])
    # print(out_array)
    return (out_array)
def Diagram_to_Dimension_Arrays(diag):
    out_diags = []
    for i in range(2):
        dim_diag = []
        for x in diag:
            if x[0] == i:
                dim_diag.append(x[1])
        out_diags.append(dim_diag)
    return(out_diags)
def Simplex_Tree(Points, max_edg_len, minimum_persistence):
    print("Building Simplex Tree...")
    # witnesses = Points
    # landmarks = gudhi.pick_n_random_points(points=witnesses, nb_points=10)
    # witness_complex = gudhi.EuclideanWitnessComplex(witnesses=witnesses, landmarks=landmarks)
    # simplex_tree = witness_complex.create_simplex_tree(max_alpha_square=.01, limit_dimension=3)
    rips_complex = gudhi.RipsComplex(points=Points,
                                     max_edge_length=max_edg_len, sparse = .001)#Sparse = .1
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)

    # result_str = 'Rips complex is of dimension ' + repr(simplex_tree.dimension()) + ' - ' + \
    #     repr(simplex_tree.num_simplices()) + ' simplices - ' + \
    #     repr(simplex_tree.num_vertices()) + ' vertices.'
    # print(result_str)
    #os.system("say 'Persistence'")
    print("Simplex Tree Built, computing persistence")
    simplex_tree.compute_persistence()  # see what happens with just persistence()

    LS = gudhi.representations.Landscape(resolution=1000)
    print('here')
    #L0 = LS.fit_transform([simplex_tree.persistence_intervals_in_dimension(0)])
    L1 = LS.fit_transform([simplex_tree.persistence_intervals_in_dimension(1)])

    new_betti_numbers = simplex_tree.betti_numbers()
    print(new_betti_numbers)

    #tree = simplex_tree.persistence()
    #print(tree)
    #print(simplex_tree)

    diag = simplex_tree.persistence(min_persistence=minimum_persistence)
    return (diag, L1, new_betti_numbers)
#OUTPUT : Persistence Diagram, "return(diag)"
def Plot_Persistence_Diagram(diag, landscape, T, directory):
    print("PLOTTING")
    #plt.subplot(1,2,1)
    gudhi.plot_persistence_barcode(diag)
    plt.savefig(os.path.join(directory, str(T) +"Persistence Barcode" +  ".png"), dpi=300)

    print("Barcode Built")
    # plt.subplot(1,2,2)
    # gudhi.plot_persistence_diagram(diag, legend = True)
    # plt.savefig(os.path.join(directory, str(T) + "Persistence Diagram" + ".png"), dpi=300)
    # print("Persistence Diagram Built")
    plt.clf()
    plt.close()

    L = landscape
    #     plt.subplot(1,2,i+1)
    #
    plt.plot(L[0][:1000])
    plt.plot(L[0][1000:2000])
    plt.plot(L[0][2000:3000])
    #     plt.title("Landscape")
    # for i in range(2):
    #     L = landscape[i]
    #     plt.subplot(1,2,i+1)
    #
    #     plt.plot(L[0][:1000])
    #     plt.plot(L[0][1000:2000])
    #     plt.plot(L[0][2000:3000])
    #     plt.title("Landscape")
    plt.savefig(os.path.join(directory, str(T) +"Persistence Landscape" +  ".png"), dpi=300)
    # plt.subplot(1,3,1)
    # gudhi.plot_persistence_density(diag,
    #                                max_intervals=0, dimension=0, legend=True)
    # plt.subplot(1,3,2)
    # gudhi.plot_persistence_density(diag,
    #                                max_intervals=0, dimension=1)
    # plt.subplot(2,3,3)
    # gudhi.plot_persistence_density(diag,
    #                                 max_intervals=0, dimension=1, legend=True)
    # plt.savefig(os.path.join(directory, str(T) + "Persistence Heat Diagram" + ".png"), dpi=300)
    # print("Barcode Density")
    # # plt.show()
    # plt.close()
    #betti_curve = gudhi.representations.vector_methods.BettiCurve(resolution=100)(Diagram_to_Array(diag))
    #dim_1_diag, dim_2_diag = Diagram_to_Dimension_Arrays(diag)
    #print(dim_1_diag)
    #print(dim_2_diag)
    #input()

#OUTPUT : PLOT

#OUTPUT : ARRAY. Converts persistence diagram to an array.
#OUTPUT : UNUSED
def Calculate_Persistence(old_diagram, market_space, max_dist, min_pers):
    tree = Simplex_Tree(market_space[0], market_space[1], max_dist, min_pers)
    diagram = Diagram_to_Array(tree)
    if old_diagram != False:
        persistence_distance = gudhi.bottleneck_distance(old_diagram, diagram)
        #print(persistence_distance)
        # try:
        #     persistence_distance =  gudhi.bottleneck_distance(old_diagram, diagram)
        # except:
        #     return(0,diagram)
    else:
        persistence_distance = 0
    return(persistence_distance,diagram)
#OUTPUT : "return(persistence_distance, persistence_diagram)"
# Other
def Stats(array):
    sum = 0
    # print(array)
    max = 0
    for x in array:
        # print(x)
        if float(x) > max:
            max = float(x)
        sum += float(x)
    mean = sum / 50
    median = array[25]
    return (sum, mean, median, max)

def Operator(new, old, del_op):
    new_avg = new[0]
    old_avg = old[0]
    new_slp = new[1]
    old_slp = old[1]
    delta_avg = []
    delta_slp = []
    for i in range(len(new_avg)):
        delta_avg.append((new_avg[i] - old_avg[i]) / del_op)
        delta_slp.append((new_slp[i] - old_slp[i]) / (del_op ** 2))
    # plt.xlim(-100,100)
    # plt.ylim(-50,50)
    plt.quiver(old_avg, old_slp, delta_avg, delta_slp, angles='xy')
    plt.quiver(np.mean(delta_avg), np.mean(delta_slp))

    for avg in new_avg:
        old_avg.append(avg)

    for slp in new_slp:
        old_slp.append(slp)

    # plt.scatter(new_avg.extend(old_avg),new_slp.append(old_slp), s=4)
    plt.scatter(old_avg, old_slp, s=4, c='b')
    plt.savefig(os.path.join(root_directory, "LinearizationZ.png"), dpi=300)  # strftime("%Y-%m-%d %H:%M", gmtime())
    # plt.show()
    plt.close()
