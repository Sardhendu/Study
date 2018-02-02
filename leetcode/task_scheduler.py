
import operator

def task_schedule(arr, m):
    # Parse once through the data set to get all numbers
    dict = {}
    for i in arr:
        try:
            dict[i] += 1
        except KeyError:
            dict[i] = 1

    tasks = list(dict.keys())
    counts = list(dict.values())
    sorted_idx = sorted(range(len(counts)), key=counts.__getitem__, reverse=True)
    # tasks = tasks[sorted_idx]

    tot_cnt = 0
    while len(tasks) >0:
        # print('')
        # print(sorted_idx)
        i = 0
        n = m+1
        while i<n:
            # print (n,i,len(tasks))
            if i < len(tasks):
                cur_idx = sorted_idx[i]
                tot_cnt += 1
                # print ('asdsd', i, cur_idx, counts[cur_idx],tasks[cur_idx])
                counts[cur_idx] = counts[cur_idx] - 1
                if counts[cur_idx] == 0:
                    # print ('count0count0count0')
                    tasks.pop(cur_idx)
                    counts.pop(cur_idx)
                    sorted_idx = sorted(range(len(counts)), key=counts.__getitem__, reverse=True)
                    i-=1
                    n-=1
            elif len(tasks)>=1:
                tot_cnt += 1
                # print ('idle')
            else:
                pass
                
            i+=1
        sorted_idx = sorted(range(len(counts)), key=counts.__getitem__, reverse=True)
    return tot_cnt



arr = ['A', 'A', 'A', 'A', 'B', 'B','B','B', 'C','C','D']
arr = ["A","A","A","B","B","B"]
task_schedule(arr,2)