import os
from mpi4py import MPI

ksdgdebug = set(os.getenv('KSDGDEBUG', default='').split(':'))

def log(*args, system = 'KSDG', **kwargs):
    comm = MPI.COMM_WORLD
    rank = comm.rank
    if system in ksdgdebug or 'ALL' in ksdgdebug:
        print('{system}, rank={rank}:'.format(system=system, rank=rank), *args, flush=True, **kwargs)
