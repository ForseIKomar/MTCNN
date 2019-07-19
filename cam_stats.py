import datetime
import cv2
import matplotlib.pyplot as plt 
from numba import jit

@jit
def save_stats(img, time_last_seen, time_first_seen, screen_time, active_time, looking_time, bound, frame_list, active_bound):
    file_name = datetime.datetime.now().strftime("save/%Y%m%d_%H%M%S.txt")
    file = open(file_name, 'w')
    file2_name = datetime.datetime.now().strftime("save/%Y%m%d_%H%M%S_all.txt")
    file2 = open(file2_name, 'w')
    for i in range(len(bound)):
        if (time_last_seen[i] - time_first_seen[i] > 1) and ((active_bound[i][0] == True) or (active_bound[i][1] == 'dead_by_timeout')):
            file.write(str(i) + ": время наблюдения = " + str(round(active_time[i] * 1.23) / 10) +
                " секунд, процент внимания = " + str(round(1000 * (looking_time[i] / active_time[i])) / 10) + 
                ", процент нахождения на экране = " + str(round(1000 * (screen_time[i] / active_time[i])) / 10) + 
                "\n " + active_bound[i][1] + ", замечен впервые в " + str(round(time_first_seen[i] * 10) / 10) + ", последнее присутствие в " + str(round(time_last_seen[i] * 10) / 10) + 
                ", bound = " + str(bound[i]) + "\n")
        file2.write(str(i) + ": время наблюдения = " + str(round(active_time[i] * 1.23) / 10) +
            " секунд, процент внимания = " + str(round(1000 * (looking_time[i] / active_time[i])) / 10) + 
            ", процент нахождения на экране = " + str(round(1000 * (screen_time[i] / active_time[i])) / 10) + 
            "\n " + active_bound[i][1] + ", замечен впервые в " + str(round(time_first_seen[i] * 10) / 10) + ", последнее присутствие в " + str(round(time_last_seen[i] * 10) / 10) + 
            ", bound = " + str(bound[i]) + "\n")
                
    file.close()
    file2.close()
    cv2.imwrite(datetime.datetime.now().strftime("save/%Y%m%d_%H%M%S.jpg"), img)
    print(file_name + ' saving done')
    
#Группировка 20 в 1, 10000 в 100
@jit   
def print_graph(x, y, look):
    time_log = x
    stats_log = y 
    look_log = look
    count = len(time_log)  
    X3 = []    
    if (count <= 100):
        plt.plot(time_log, stats_log)
        plt.plot(time_log, look_log)
        for i in range(len(stats_log)):
            X3.append(10 * look_log[i] / stats_log[i])
        plt.plot(time_log, X3)
            
    else:
        distance = count / 50
        last = 0
        sum = 0
        sum2 = 0
        count = 0
        X = []
        X2 = []
        T = []
        for i in range(len(time_log)):
            if (round(i / distance) != last):
                X.append(sum / count)
                X2.append(sum2 / count)
                if (sum > 0):
                    X3.append(10 * sum2 / sum)
                else:
                    X3.append(0)
                T.append(time_log[i])
                sum = 0
                sum2 = 0
                count = 0
                last = round(i / distance)
            else:
                sum += stats_log[i]
                sum2 += look_log[i]
                count += 1
        plt.plot(T, X)
        plt.plot(T, X2)
        plt.plot(T, X3)
    plt.show()