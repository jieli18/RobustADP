#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Jie Li
#  Description: Create simulator
import ray
from ..trainer.simulator import Simulator


def create_simulator(**kwargs):
    simulator_file_name = kwargs['simulator_name'].lower()
    trainer = kwargs['trainer']
    try:
        file = __import__(simulator_file_name)
    except NotImplementedError:
        raise NotImplementedError('This simulator does not exist')
    simulator_name = formatter(simulator_file_name)
    #
    if hasattr(file, simulator_name):  #
        simulator_cls = getattr(file, simulator_name)
        if trainer == 'off_serial_trainer' or trainer == 'on_serial_trainer' \
                or trainer == 'on_serial_trainer_shared_network':
            simulator = simulator_cls(**kwargs)
        elif trainer == 'off_async_trainer' or trainer == 'on_sync_trainer' or trainer == 'off_async_trainermix':
            simulator = ray.remote(num_cpus=1)(Simulator).remote(**kwargs)
        else:
            raise NotImplementedError("This trainer is not properly defined")
    else:
        raise NotImplementedError("This simulator is not properly defined")

    print("Create simulator successfully!")
    return simulator


def formatter(src: str, firstUpper: bool = True):
    arr = src.split('_')
    res = ''
    for i in arr:
        res = res + i[0].upper() + i[1:]

    if not firstUpper:
        res = res[0].lower() + res[1:]
    return res
