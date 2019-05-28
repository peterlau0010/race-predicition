# import multiprocessing
import itertools
# import time
# from multiprocessing import Pool
from multiprocessing import Process,Pool
import time
# import multiprocessing 

# train_test_col =['class', 'Draw', 'Age', 'AWT', 'Horse Wt. (Declaration)', 'Rtg.+/-', 'Runs_1', 'Runs_2', 'Runs_3', 'Runs_4', 'B', 'H', 'TT', 'CP', ]
# train_test_col =['class', 'Draw']
train_test_col =['class', 'Draw', 'Age', 'AWT', 'Horse Wt. (Declaration)', 'Rtg.+/-', 'Runs_1', 'Runs_2', 'Runs_3', 'Runs_4', 'B', 'H', 'TT', 'CP', 'V', 'XB', 'Sex_c', 'Sex_f', 'Sex_g', 'Sex_h', 'Sex_r', 'going_GOOD', 'going_GOOD TO FIRM', 'going_GOOD TO YIELDING', 'going_YIELDING', 'raceCourse_HV', 'TrainerRank', 'SireRank', 'JockeyRank', 'raceCourse_ST', 'horseRank', 'DamRank']
train_test_col = itertools.permutations(train_test_col)
def print_func(continent):
    print('The name of continent is : ', continent)
    time.sleep(2)

if __name__ == "__main__":  # confirms that the code is under main function
    names = train_test_col
    procs = []
    # proc = Process(target=print_func)  # instantiating without any argument
    # procs.append(proc)
    # proc.start()

    # instantiating process with arguments
    pool = Pool(processes=3)
    for name in names:
        # print(name)
        pool.map_async(print_func, (name,))
        # print(name)
        # proc = Process(target=print_func, args=(name,))
        # procs.append(proc)
        # proc.start()
    # pool.close()
    # pool.join()
    # complete the processes
    # for proc in procs:
    #     proc.join()