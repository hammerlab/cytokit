
from queue import Queue
from time import sleep

def load_data(q):
    i = 0
    while True:
        sleep(5)
        q.put(i)
        i += 1

def run_job(idx):

    queue = Queue()

    from threading import Thread
    load_thread = Thread(target=load_data, args=(queue,))
    load_thread.start()

    print('Wating for item')
    item = queue.get(block=True, timeout=10)
    print('got item', item)

    
    

if __name__ == '__main__':
    run_job(1)