from builtins import zip
from builtins import str
from builtins import map
from builtins import range

import glob
import ipyparallel
from ipyparallel import Client
import logging
import multiprocessing
from multiprocessing import Pool
import numpy as np
import os
import platform
import psutil
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

def shell_source(script: str) -> None:
    """ Run a source-style bash script, copy resulting env vars to current process. """
    # XXX This function is weird and maybe not a good idea. People easily might expect
    #     it to handle conditionals. Maybe just make them provide a key-value file
    #introduce echo to indicate the end of the output
    pipe = subprocess.Popen(f". {script}; env; echo 'FINISHED_CLUSTER'", stdout=subprocess.PIPE, shell=True)

    env = dict()
    while True:
        line = pipe.stdout.readline().decode('utf-8').rstrip()
        if 'FINISHED_CLUSTER' in line:         # find the keyword set above to determine the end of the output stream
            break
        logger.debug("shell_source parsing line[" + str(line) + "]")
        lsp = str(line).split("=", 1)
        if len(lsp) > 1:
            env[lsp[0]] = lsp[1]

    os.environ.update(env)
    pipe.stdout.close()

def start_server(slurm_script: str = None, ipcluster: str = "ipcluster", ncpus: int = None) -> None:
    """
    programmatically start the ipyparallel server

    Args:
        ncpus: int
            number of processors

        ipcluster : str
            ipcluster binary file name; requires 4 path separators on Windows. ipcluster="C:\\\\Anaconda3\\\\Scripts\\\\ipcluster.exe"
            Default: "ipcluster"
    """
    logger.info("Starting cluster...")
    if ncpus is None:
        ncpus = psutil.cpu_count()

    if slurm_script is None:

        if ipcluster == "ipcluster":
            subprocess.Popen("ipcluster start -n {0}".format(ncpus), shell=True, close_fds=(os.name != 'nt'))
        else:
            subprocess.Popen(shlex.split("{0} start -n {1}".format(ipcluster, ncpus)),
                             shell=True,
                             close_fds=(os.name != 'nt'))
        time.sleep(1.5)
        # Check that all processes have started
        client = ipyparallel.Client()
        time.sleep(1.5)
        while len(client) < ncpus:
            sys.stdout.write(".")                              # Give some visual feedback of things starting
            sys.stdout.flush()                                 # (de-buffered)
            time.sleep(0.5)
        logger.debug('Making sure everything is up and running')
        client.direct_view().execute('__a=1', block=True)      # when done on all, we're set to go
    else:
        shell_source(slurm_script)
        pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
        logger.debug([pdir, profile])
        c = Client(ipython_dir=pdir, profile=profile)
        ee = c[:]
        ne = len(ee)
        logger.info(f'Running on {ne} engines.')
        c.close()
        sys.stdout.write("start_server: done\n")


def stop_server(ipcluster: str = 'ipcluster', pdir: str = None, profile: str = None, dview=None) -> None:
    """
    programmatically stops the ipyparallel server

    Args:
        ipcluster : str
            ipcluster binary file name; requires 4 path separators on Windows
            Default: "ipcluster"a

        pdir : Undocumented
        profile: Undocumented
        dview: Undocumented

    """
    if 'multiprocessing' in str(type(dview)):
        dview.terminate()
    else:
        logger.info("Stopping cluster...")
        try:
            pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
            is_slurm = True
        except:
            logger.debug('stop_server: not a slurm cluster')
            is_slurm = False

        if is_slurm:
            if pdir is None and profile is None:
                pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
            c = Client(ipython_dir=pdir, profile=profile)
            ee = c[:]
            ne = len(ee)
            logger.info(f'Shutting down {ne} engines.')
            c.close()
            c.shutdown(hub=True)
            shutil.rmtree('profile_' + str(profile))
            try:
                shutil.rmtree('./log/')
            except:
                logger.info('creating log folder')     # FIXME Not what this means

            files = glob.glob('*.log')
            os.mkdir('./log')

            for fl in files:
                shutil.move(fl, './log/')

        else:
            if ipcluster == "ipcluster":
                proc = subprocess.Popen("ipcluster stop",
                                        shell=True,
                                        stderr=subprocess.PIPE,
                                        close_fds=(os.name != 'nt'))
            else:
                proc = subprocess.Popen(shlex.split(ipcluster + " stop"),
                                        shell=True,
                                        stderr=subprocess.PIPE,
                                        close_fds=(os.name != 'nt'))

            line_out = proc.stderr.readline()
            if b'CRITICAL' in line_out:
                logger.info("No cluster to stop...")
            elif b'Stopping' in line_out:
                st = time.time()
                logger.debug('Waiting for cluster to stop...')
                while (time.time() - st) < 4:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                    time.sleep(1)
            else:
                logger.error(line_out)
                logger.error('**** Unrecognized syntax in ipcluster output, waiting for server to stop anyways ****')

            proc.stderr.close()

    logger.info("stop_cluster(): done")


def setup_cluster(backend: str = 'multiprocessing',
                  n_processes: int = None,
                  single_thread: bool = False,
                  ignore_preexisting: bool = False,
                  maxtasksperchild: int = None) -> Tuple[Any, Any, Optional[int]]:
    """Setup and/or restart a parallel cluster.
    Args:
        backend: str
            'multiprocessing' [alias 'local'], 'ipyparallel', and 'SLURM'
            ipyparallel and SLURM backends try to restart if cluster running.
            backend='multiprocessing' raises an exception if a cluster is running.
        ignore_preexisting: bool
            If True, ignores the existence of an already running multiprocessing
            pool, which is usually indicative of a previously-started CaImAn cluster

    Returns:
        c: ipyparallel.Client object; only used for ipyparallel and SLURM backends, else None
        dview: ipyparallel dview object, or for multiprocessing: Pool object
        n_processes: number of workers in dview. None means guess at number of machine cores.
    """

    if n_processes is None:
        if backend == 'SLURM':
            n_processes = np.int(os.environ.get('SLURM_NPROCS'))
        else:
            # roughly number of cores on your machine minus 1
            n_processes = np.maximum(np.int(psutil.cpu_count() - 1), 1)

    if single_thread:
        dview = None
        c = None
    else:
        sys.stdout.flush()

        if backend == 'SLURM':
            try:
                stop_server()
            except:
                logger.debug('Nothing to stop')
            slurm_script = '/mnt/home/agiovann/SOFTWARE/CaImAn/SLURM/slurmStart.sh' # FIXME: Make this a documented environment variable
            logger.info([str(n_processes), slurm_script])
            start_server(slurm_script=slurm_script, ncpus=n_processes)
            pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
            logger.info([pdir, profile])
            c = Client(ipython_dir=pdir, profile=profile)
            dview = c[:]
        elif backend == 'ipyparallel':
            stop_server()
            start_server(ncpus=n_processes)
            c = Client()
            logger.info(f'Started ipyparallel cluster: Using {len(c)} processes')
            dview = c[:len(c)]

        elif (backend == 'multiprocessing') or (backend == 'local'):
            if len(multiprocessing.active_children()) > 0:
                if ignore_preexisting:
                    logger.warn('Found an existing multiprocessing pool. '
                                'This is often indicative of an already-running CaImAn cluster. '
                                'You have configured the cluster setup to not raise an exception.')
                else:
                    raise Exception(
                        'A cluster is already runnning. Terminate with dview.terminate() if you want to restart.')
            if (platform.system() == 'Darwin') and (sys.version_info > (3, 0)):
                try:
                    if 'kernel' in get_ipython().trait_names():        # type: ignore
                                                                       # If you're on OSX and you're running under Jupyter or Spyder,
                                                                       # which already run the code in a forkserver-friendly way, this
                                                                       # can eliminate some setup and make this a reasonable approach.
                                                                       # Otherwise, seting VECLIB_MAXIMUM_THREADS=1 or using a different
                                                                       # blas/lapack is the way to avoid the issues.
                                                                       # See https://github.com/flatironinstitute/CaImAn/issues/206 for more
                                                                       # info on why we're doing this (for now).
                        multiprocessing.set_start_method('forkserver', force=True)
                except:                                                # If we're not running under ipython, don't do anything.
                    pass
            c = None

            dview = Pool(n_processes, maxtasksperchild=maxtasksperchild)
        else:
            raise Exception('Unknown Backend')

    return c, dview, n_processes
