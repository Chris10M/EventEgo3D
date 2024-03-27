import cv2
import os
import platform
import os
import torch.multiprocessing as mp

os.makedirs('logs', exist_ok=True)

if platform.system() == 'Windows':
    os.environ['BATCH_SIZE'] = '27'
else:
    cv2.setNumThreads(0)

    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    i = 1
    while True:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (131072 / i, rlimit[1]))
            break
        except ValueError:
            i += 1

    print(f'rlimit: {resource.getrlimit(resource.RLIMIT_NOFILE)}')

