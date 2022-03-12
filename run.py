import Websocket
import tsi_w
import os
import multiprocessing
import time
root_directory = os.getcwd()
id = '033'

directory = os.path.join(root_directory, id)
if not os.path.exists(directory):
    os.makedirs(directory)
    os.makedirs(os.path.join(directory,'pics'))
    print('start', id)
def multi_run(type):
    if type == 'sockets':
        print('socket')
        Websocket.socket_run(directory)
    elif type == 'analysis':
        print('analysis')
        tsi_w.tsi_run(directory)
def main():
    iterable = [['sockets'],['analysis']]
    pool = multiprocessing.Pool()
    pool.starmap(multi_run,iterable)
    pool.close()
    pool.join()
if __name__ == "__main__":
    main()