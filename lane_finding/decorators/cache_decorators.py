import os.path as path
import pickle


def calibration_cache(func):
    """
    Decorator for calibration function to avoid re-computing calibration every time.
    """
    calibration_pickle = 'data/camera_cal/calibration_data.pickle'

    def wrapper(*args, **kwargs):
        if path.exists(calibration_pickle):
            print('Loading cached camera calibration data...', end=' ')

            with open(calibration_pickle, 'rb') as dump_file:
                calibration = pickle.load(dump_file)
        else:
            print('Calculating camera calibration data...', end=' ')
            calibration = func(*args, **kwargs)

            with open(calibration_pickle, 'wb') as dump_file:
                pickle.dump(calibration, dump_file)

        return calibration

    return wrapper
