import time

class Timer:

    def __init__(self, key):
        self.key = key
        self.start = time.time()
        self.stop = None
        self.position = time.time()*100 # a timestamp is used to cope with parallel jobs
        print('Start: {}'.format(self.key))

    def total(self):
        if self.stop is None:
            self.stop = time.time()
            print('End: {}'.format(self.key))
        return self.stop - self.start

    def save(self, filename, io_data):
        if self.stop is None:
            self.stop = time.time()
            print('End: {}'.format(self.key))

        total_time = self.stop - self.start
        io_data.save_time('{}:{}:{}'.format(self.key, total_time, self.position), filename)
