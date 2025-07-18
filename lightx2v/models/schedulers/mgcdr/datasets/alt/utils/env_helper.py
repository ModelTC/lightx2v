from loguru import logger

MASTER_RANK = 0


class Environment:
    def __init__(self):
        try:
            from mpi4py import MPI
            self._keep_value = False
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.world_size = self.comm.Get_size()
        except Exception as e:
            logger.waring(e)
            self.world_size = self.rank = 0

    def is_master(self):
        
        return self.rank == MASTER_RANK

    @property
    def distributed(self):
        return self.world_size > 1

    def keep_value(self, keep=True):
        self._keep_value = keep

    def barrier(self):
        if not self.distributed:
            return
        self.comm.Barrier()

    def gather(self, data, root=0, merge=True):
        if not self.distributed:
            return data
        gathered_data = self.comm.gather(data, root=root)
        if root == self.rank and merge:
            if isinstance(gathered_data[0], list):
                final_data = []
                for item in gathered_data:
                    final_data.extend(item)
            elif isinstance(gathered_data[0], dict):
                final_data = gathered_data  # TODO: merge dicts
            else:
                final_data = gathered_data
        else:
            final_data = gathered_data
        return final_data

    def is_my_showtime(self, idx):
        if not self.distributed:
            return True
        return (idx - self.rank) % self.world_size == 0

    def belong_me(self, input):
        per_num = len(input) // self.world_size + 1
        output = []
        for i in range(self.world_size):
            output.append(input[i * per_num:(i + 1) * per_num])
        return output[self.rank]
    
    def bcast(self, data, root=0):
        if not self.distributed:
            return data
        data = self.comm.bcast(data, root=root)
        return data

env = Environment()
