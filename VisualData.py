
from multiprocessing import Process, Value, Lock, Pool, Manager
import itertools


def f(name,counter):
    counter.value +=1
    
    print('F called:',name)
    return counter.value
    
if __name__ == "__main__":
    train_test_col = ['B', 'H', 'TT', 'CP', 'V', 'XB', 'SR', 'P', 'PC', 'E', 'BO', 'PS', 'SB', 'Sex_c', 'Sex_f', 'Sex_g', 'Sex_h', 'Sex_r', 'going_GOOD', 'going_GOOD TO FIRM', 'going_GOOD TO YIELDING', 'going_YIELDING', 'raceCourse_HV', 'TrainerRank', 'SireRank', 'horseRank', 'JockeyRank', 'Runs_1', 'Runs_2', 'Runs_3', 'Runs_4', 'Runs_5', 'Runs_6', 'raceCourse_ST', 'Draw', 'Age', 'AWT', 'Rtg.+/-', 'DamRank', 'Horse Wt. (Declaration)', 'class']
    # train_test_col = ['B', 'H', 'TT']
    perm = itertools.permutations(train_test_col)
    names = perm

    manager = Manager()
    counter = manager.Value('i',0)

    
    pool = Pool(processes=2)
    for name in names:
        print('start')
        result = pool.apply_async(f, (name,counter))
        # print(result.get())
    pool.close()
    pool.join()
print(counter.value)
