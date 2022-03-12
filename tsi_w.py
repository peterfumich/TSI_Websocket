import pandas as pd
import numpy as np
import os
import time
import TDA
import gudhi
import datetime
from matplotlib import pyplot as plt
import Websocket
####
sides = ['asks','bids']
assets = ['ethusd']
initial_ms = {'bids':0,'asks':0}

websocket_column_names = ['timestampms', 'socket_sequence','event_price','amount_remaining']
persistence_experimenting_names = ["time","N","n","delta_t","min_persistence","max_persistence","betti_numbers","persistence_distance"]
column_names = ['time', 'price']
root_directory = os.getcwd()
directories = {'directory':os.getcwd()}#root_directory
new_files = 'y'

moving_average_diff_sets = {'bids':[],'asks':[]}
chunked_ma_diff_sets = {'bids':[],'asks':[]}
last_moving_average_diag = {'bids':[],'asks':[]}
market_data = {'bids_price':0,'bids_center':'[0,0]', 'bids_diag':'[]', 'asks_price':0, 'asks_center':'[0,0]', 'asks_diag':'[]'}
# market_df = pd.DataFrame.from_records(market_data, index=[0])
# market_df.to_csv(os.path.join(directories['directory'], "data" + ".csv"), mode='a', header=None)
# print(market_df)
persistences = {'bids':[],'asks':[]}
out_diagrams = {'bids':[],'asks':[]}
delta_t = 1000 # Bucket size #base interval: number of miliseconds
n = 60#60# Number of base intervals in Market Space
chunks = 10
action_rate = int(n/chunks)
N = 30# Number of Market Space point clouds into rips complex
#max_persistence = 10 works for N = 20, n = 15, and generally works for N=20, n=60. But python crashes for n=300.
max_persistence = 500#Trying to determine a minimum max_persistence so that python does not crash during persistence calculations.
min_persistence = 1
last_lines = {'bids':0,'asks':0}

generated_prices = {'bids':[],'asks':[]}
sleep_timer ={'timer':0}

MSPC_max_persistence =.1
MSPC_min_peristence = .001
number_of_samples = 200
parameters_dict = {'delta_t':delta_t,'n':n,'chunks':chunks,'N':N,'Number of MSPC samples':number_of_samples,'MSPC_max_persistence':MSPC_max_persistence,
                   'MSPC_min_persitence':MSPC_min_peristence}

def Generate_Data_2(chunk_count):
    print("Generating Data")
    for asset in assets:
        for side in sides:
            out_data = generated_prices[side]#[]
            seconds = 0
            data = pd.read_csv(os.path.join(directories['directory'],asset + side + '.csv'))
            #print(data)
            if len(data)==last_lines[side]:
                print("Socket disconnected!")
                print(data)
            initial_ms[side] = data.iloc[last_lines[side]]['timestampms']
            prices = []
            data_length = len(data)
            i = 0
            while True:

            #for i in range(last_lines[side]):
                if i+last_lines[side]< data_length:
                    if data.iloc[i+last_lines[side]]['timestampms'] < (initial_ms[side] + (seconds + 1) * delta_t):
                        prices.append(float(data.iloc[i+last_lines[side]]['event_price']))
                    else:
                        if prices != []:
                            price = Price_Stat(prices)
                            seconds += 1
                            # out_row = [seconds, price]#pd.DataFrame({'time': seconds, 'price': price}, index=[i])#+seconds_dict[side]
                            out_data.append(price)  # out_data.append(out_row)
                            prices = [float(data.iloc[i+last_lines[side]]['event_price'])]

                        else:
                            prices = [float(data.iloc[i+last_lines[side]]['event_price'])]
                            seconds += 1
                else:
                    break
                i+=1
            if len(out_data)==len(generated_prices[side]) and len(prices)!=0:
                out_data.append(Price_Stat(prices))

            print("Prices generated:")
            #print(out_data)
            generated_prices[side] = out_data
            if len(out_data) >= n:
                print('making new data')
                price, cm =Analyze_Data(out_data, side, chunk_count)
                market_data[str(side+"_price")] = str(price)
                market_data[str(side+"_center")] = str(cm)
                market_data[str(side+"_diag")] = str(last_moving_average_diag[side])
            last_lines[side] = data_length

    chunk_count+=1
    if market_data['bids_price'] != 0:
        print('writing new data')
        print("Market Data: ", market_data)
        market_df = pd.DataFrame.from_records(market_data, index=[0])
        print("data frame built")
        market_df.to_csv(os.path.join(directories['directory'], "data"+".csv"), mode='a', header=None)
    return(chunk_count)
def Price_Stat(prices):
    out_price = np.mean(prices)
    return(out_price)
def Analyze_Data(data,side, chunk_count):
    print("Analyzing Data")
    timer = time.time()
    flipped_data = np.flip(data,axis=0)
    last_price = flipped_data[0]
    ma_set = TDA.Create_Moving_Average(flipped_data,n,1)
    dma_set = TDA.Create_Slopes(flipped_data,1,n,1)
    ma_diff_set, dma_diff_set, market_space = TDA.Create_Averaging_Arrays(ma_set, dma_set)

    plt.subplot(1,2,1)
    plt.title("Price")
    plt.xlabel('time')
    plt.ylabel('Price')

    plt.plot(generated_prices[side][-n::])#plt.plot(generated_prices[side][-1*action_rate::])#plt.plot(generated_prices[side])#
    # plt.savefig(os.path.join(directories['directory'],"pics/"+"prices_"+ side+str(timer) + ".png"))
    # plt.clf()
    # plt.close

    # plt.close()
    plt.subplot(1,2,2)
    plt.title("MSPC"+str(datetime.datetime.now()))
    plt.xlabel('difference in moving averages')
    plt.ylabel('difference in slopes of moving averages')
    plt.xlim([-2, 2])
    plt.ylim([-1,1])
    plt.scatter(ma_diff_set, dma_diff_set, s=1)

    #plt.show()
    plt.savefig(os.path.join(directories['directory'],"pics/"+side+"_Data_"+ str(timer) + ".png"))
    plt.clf()
    plt.close()
    MSPC_Homology(ma_diff_set,dma_diff_set,side)
    cm = TDA.Center_of_Mass(ma_diff_set, dma_diff_set)
    bcm = TDA.Boundary_Box_Center(ma_diff_set,dma_diff_set)


    print("Center of Mass of Market Space: "+str(cm))
    print("Boundary box center", str(bcm))
    if chunk_count%chunks == 0:
        print("Chunked")
        #moving_average_diff_sets[side].append([ma_diff_set,dma_diff_set])
        moving_average_diff_sets[side].append(market_space)
    else:
        #chunked_ma_diff_sets[side] = [ma_diff_set,dma_diff_set]
        chunked_ma_diff_sets[side] = market_space

    if len(moving_average_diff_sets[side])>N:
        print('Calculating Persistence of', N, "market space point clouds")
        timer = time.time()
        print(timer)
        print("Initial max_persistence = ", max_persistence)
        SW_Homology(side, chunk_count)
    else:
        print("Not enough data")
    return(last_price,cm)
def SW_Homology(side, chunk_count):
    # print('Calculating Persistence of', N, "market space point clouds")
    timer = time.time()
    # print(timer)
    maximum_persistence = 1 * max_persistence
    persistence_input_data = moving_average_diff_sets[side]
    if chunk_count%chunks != 0:
        persistence_input_data.append(chunked_ma_diff_sets[side])
    else:
        out_diagrams[side].append(last_moving_average_diag[side])
        moving_average_diff_sets[side] = moving_average_diff_sets[side][1::]


    persistence_diagram, landscape, betti_numbers = TDA.Simplex_Tree(persistence_input_data, max_persistence,
                                                          min_persistence)
    while betti_numbers[0] > 1:
        maximum_persistence += 1  # min_persistence
        print("Maximum Persistence = ", maximum_persistence)
        persistence_diagram, landscape, betti_numbers = TDA.Simplex_Tree(persistence_input_data, maximum_persistence,
                                                              min_persistence)
    print("Done computing Betti numbers")
    # Alternatively, start with sufficiently large max_persistence and then decrease the max_persistence until
    # the feature changes.

    # while betti_numbers[0]==1:
    #     maximum_persistence -= 1
    #     print("Maximum Persistence = ", maximum_persistence)
    #     moving_average_diag, landscape, betti_numbers = TDA.Simplex_Tree(moving_average_diff_sets[side], maximum_persistence,
    #                                                           min_persistence)
    if last_moving_average_diag[side] != []:
        persistence_distance = gudhi.bottleneck_distance(TDA.Diagram_to_Array(
            last_moving_average_diag[side]), TDA.Diagram_to_Array(persistence_diagram))
        persistences[side].append(persistence_distance)
        print(persistences)
    else:
        persistence_distance = 0
    #["time","N","n","delta_t","min_persistence","max_persistence","betti_numbers","persistence_distance"]
    #persistence_df_line = pd.DataFrame([str(timer),str(N),str(n),str(delta_t),str(min_persistence),str(maximum_persistence),str(betti_numbers),str(persistence_distance)],columns=persistence_experimenting_names)
    #print("Persistence Data Frame Built")
    #persistence_df_line.to_csv("persistence_experiment"+asset+side+".csv", mode='a', header=None)
    print("Betti numbers for "+side+" = "+str(betti_numbers))
    print("Maximum Persistence used",maximum_persistence)
    last_moving_average_diag[side] = persistence_diagram


    #[]#Alternatively drop the first diagram,

    TDA.Plot_Persistence_Diagram(persistence_diagram, landscape, 'interleaved' + side + str(timer),
                                 os.path.join(directories['directory'],"pics"))
    #What other data needs to be stored?
def MSPC_Homology(x_list,y_list,side):
    t_0 = time.time()

    timer = time.time()
    points = []
    for i,x in enumerate(x_list):
        point = [x,y_list[i]]
        points.append(point)
    points = np.array(points)
    if number_of_samples != 0:
        sample_points = points[np.random.choice(points.shape[0], number_of_samples, replace=False), :]#points[0:number_of_samples]#
    else:
        sample_points = points
    print('sample points generated')
    diag, landscape, bett_numbers = TDA.Simplex_Tree(sample_points,MSPC_max_persistence,MSPC_min_peristence)#2*max(abs(x_list)),min(abs(y_list)))
    print(diag)
    TDA.Plot_Persistence_Diagram(diag, landscape,str(side)+"MSPC"+str(timer),os.path.join(directories['directory'],"pics"))
    t_1 = time.time()
    sleep_timer['timer'] = t_1-t_0
    print('total time to compute homology of MSPC = ', str(t_1-t_0))
def tsi_run(in_directory):
    initial_line = 0
    directories['directory'] = in_directory
    parameters_df = pd.DataFrame.from_records(parameters_dict, index=[0])
    parameters_df.to_csv(os.path.join(directories['directory'], "parameters.csv"))
    if new_files == 'y':
        dff = pd.DataFrame(columns=['bids_price','bids_center','bids_diag', 'asks_price', 'asks_center', 'asks_diag'])
        dff.to_csv(os.path.join(directories['directory'], "data"+".csv"))
        for asset in assets:
            df = pd.DataFrame(columns=column_names)
            df.to_csv(os.path.join(directories['directory'], "sorted" + asset + "bids" + ".csv"))
            df.to_csv(os.path.join(directories['directory'], "sorted" + asset + "asks" + ".csv"))


    new_persistences = 'n'
    if new_persistences == 'y':
        persistence_df = pd.DataFrame(columns=persistence_experimenting_names)
        persistence_df.to_csv("persistence_experiment" + asset + "bids.csv")
        persistence_df.to_csv("persistence_experiment" + asset + "asks.csv")

    count = 0
    chunk_count = 0
    prices = []
    print('initial sleep to collect data')
    #time.sleep(delta_t*n/1000)
    while True:
        sleep = max(0,(action_rate)*delta_t/1000-sleep_timer['timer'])
        print('sleeping for ', sleep)
        #time.sleep((action_rate)*delta_t/1000)
        time.sleep(sleep)
        count += 1#/chunks
        print(str(count)+". Chunk count= "+str(count%chunks))
        try:
            chunk_count = Generate_Data_2(chunk_count)
        except:
            print("ERROR!")
            pass

