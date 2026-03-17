## This file has other functions that may be helpful like print wrapper and a global parallel processor
from multiprocessing import Pool
from os import cpu_count
from time import strftime

def dprint(func):
    def wrapped_func(*args, **kwargs):
        return func(strftime("%H:%M:%S - "),*args,**kwargs)
    return wrapped_func

print = dprint(print)
# class Batcher:
#     def __init__(self,
#                     folder='/home/rs/19CS91R05/DarKSkuLL/BI/Anguli_200k_1M',
#                     ext='*.tiff',
#                     batch_size=1000
#                 ) -> iter:
#         self.folder=folder
#         self.ext=ext
#         self.batch_size=1000
#         self.proc_count=cpu_count()-1
#         self.batch = iter()

#     # def
#     def __call__(self, ) -> iter:
#         return next(self.batch)
class Parallel:
    def __init__(self,debug=False) -> object:
        self.debug=debug
        print("Parallel Instance Created")
    
    def batcher(self,doc):
        if doc:
            self.doc_list.append(doc)
            # self.ctr += 1
            self.done += 1
            if len(self.doc_list) >= self.batch_size:
                # self.ctr = 0
                print(f"{self.done}/{len(self.paths)} documents processed")
                # if len(self.doc_list):
                dc=self.doc_list[:]
                self.doc_list = []
                return dc
            return None
            # if len(self.doc_list):
            #     dc=self.doc_list[:]
            #     self.doc_list = []
            #     return dc

    def __call__(
        self, atomic_function, paths, batch_size=100, chunksize=10, free_core=1
    ) -> iter:
        self.paths=paths
        self.batch_size=batch_size
        self.ctr = self.done = 0
        self.doc_list = []
        if not self.debug:
            with Pool(cpu_count() - free_core - 1) as p:
                for doc in p.imap(atomic_function, paths, chunksize):
                    if (dc:=self.batcher(doc)):
                        yield dc
                if len(dc:=self.doc_list[:]):
                    self.doc_list = []
                    yield dc
        else:
            for path in paths:
                if (dc:=self.batcher(atomic_function(path))):
                    yield dc
            if len(dc:=self.doc_list[:]):
                self.doc_list = []
                yield dc
