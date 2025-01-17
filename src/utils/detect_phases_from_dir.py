

DEBUG = False

def detect_phases(dir_1d_mean):
    """
    Detect five cardiac phases from a 1D direction curve
    Args:
        dir_1d_mean (): np.ndarray() with shape t,

    Returns:
        ndarray with five cardiac phases (ED, MS, ES, PF, MD)

    """
    import numpy as np

    length = len(dir_1d_mean)

    # MS
    # Global min of f(x)
    ms = get_ms(dir_1d_mean, length=length)
    if DEBUG: print('ms: {}'.format(ms))

    # ES
    # First time f(x)>0 after MS
    # ES should be between MS and approx PF == argmax
    es = get_es(dir_1d_mean, lower_idx=ms, upper_idx=None, first_transition=True, length=length)
    if DEBUG: print('es: {}'.format(es))

    # PF
    # First peak of the two biggest peaks
    # else argmax
    # on es - ms
    # sometimes the relaxation after the MD phase is stronger than the PF relaxation
    # otherwise we could simply:
    # pf = np.argmax(dir_1d_mean)
    pf = get_pf(dir_1d_mean, lower_idx=es, upper_idx=ms)
    if DEBUG: print('pf: {}'.format(pf))

    # ED 1st
    # Between pf and ms: last time f(x) cross zero from positive to negative
    ed = get_ed(dir_1d_mean, length=length, lower_idx=pf, upper_idx=ms, first_transition=False)
    if DEBUG: print('ed: {}'.format(ed))

    # MD
    # take the biggest peak between pf and ed 1st
    # on pf - ed
    md = get_md(dir_1d_mean, lower_idx=pf, upper_idx=ed)
    if DEBUG: print('md: {}'.format(md))

    # ED 2nd
    # Between md and ms: first time f(x) cross zero from positive to negative
    ed = get_ed(dir_1d_mean, length=length, lower_idx=md, upper_idx=ms, first_transition=False) # False or True
    if DEBUG: print('ed: {}'.format(ed))

    # ES 2nd - now with better borders based on the found phases: ms and pf
    # take the last transition from negative to positive between the border indices
    es = get_es(dir_1d_mean, lower_idx=ms, upper_idx=pf, first_transition=False, length=length)
    if DEBUG: print('es: {}'.format(es))

    # MD 2nd - based on better border indices and with fallback scenario
    md = get_md_second(dir_1d_mean, lower_idx=pf, upper_idx=ed)
    if DEBUG: print('md: {}'.format(md))

    # MS 2nd - based on the labeling definition of ms being exactly between ed and es
    # ms = get_ms_second(dir_1d_mean, lower_idx=ed, upper_idx=es)

    return np.array([ed, ms, es, pf, md])


def get_pf_second(dir_1d_mean, lower_idx, upper_idx):
    import numpy as np
    import scipy.signal as sig
    length = len(dir_1d_mean)
    lower_idx = np.mod(lower_idx, length)
    if lower_idx < upper_idx:
        cycle = dir_1d_mean[lower_idx:upper_idx]
    else:
        cycle = np.concatenate([dir_1d_mean[lower_idx:], dir_1d_mean[:upper_idx]])

    peaks, prop = sig.find_peaks(cycle, prominence=(
        None, None))  # height=0.6 we normalise between -1 and 1, PF should be close to argmax
    if len(peaks) == 0:
        pf_max = np.argmax(cycle)
        pf = lower_idx + pf_max
    else:  # more than 2 peaks
        # more than 3 peaks
        # sort the first two peaks by prominence or absolute value, take the bigger one
        peak_values = cycle[peaks]
        # get the biggest peak between ES and MD
        pf = lower_idx + peaks[0]
    pf = np.mod(pf, length)
    return pf


def get_md(dir_1d_mean, lower_idx, upper_idx):
    """

    Args:
        dir_1d_mean ():
        lower_idx ():
        upper_idx ():

    Returns:

    """
    import numpy as np
    import scipy.signal as sig
    length = len(dir_1d_mean)
    if lower_idx < upper_idx:
        cycle = dir_1d_mean[lower_idx:upper_idx]
    else:
        cycle = np.concatenate([dir_1d_mean[lower_idx:], dir_1d_mean[:upper_idx]])

    peaks, prop = sig.find_peaks(cycle, prominence=(None, None))
    if len(peaks) > 0:
        peak_values = cycle[peaks]
        peaks = [p for p, _ in sorted(zip(peaks, peak_values), key=lambda pair: pair[1],
                                      reverse=True)]  # or use the prominence: prop['prominences']
        md = lower_idx + peaks[0]  # max(peaks)  # take the biggest peak between the boundaries
    else:
        try:
            md = lower_idx + np.argmax(cycle)
        except Exception as e:
            if lower_idx < upper_idx:
                md = (lower_idx + upper_idx) // 2  # direct middle
            else:
                md = (upper_idx + length + lower_idx) // 2  # middle across the cycle border
    if lower_idx < upper_idx:
        md = (lower_idx + upper_idx) // 2  # direct middle
    else:
        md = (upper_idx + length + lower_idx) // 2  # middle across the cycle border
    md = np.mod(md, length)
    return md


def get_ms_second(dir_1d_mean, lower_idx, upper_idx):
    length = len(dir_1d_mean)
    if lower_idx < upper_idx:
        ms = (lower_idx + upper_idx) // 2  # direct middle
    else:
        ms = (upper_idx + length + lower_idx) // 2  # middle across the cycle border

    ms = np.mod(ms, length)
    return ms

def get_md_second(dir_1d_mean, lower_idx, upper_idx):
    """

    Args:
        dir_1d_mean ():
        lower_idx ():
        upper_idx ():

    Returns:

    """
    import numpy as np
    import scipy.signal as sig
    length = len(dir_1d_mean)
    if lower_idx < upper_idx:
        cycle = dir_1d_mean[lower_idx:upper_idx]
    else:
        cycle = np.concatenate([dir_1d_mean[lower_idx:], dir_1d_mean[:upper_idx]])

    peaks, prop = sig.find_peaks(cycle, prominence=(None, None))
    if len(peaks) > 0:
        # get the idx of last peak and take the frame before the last peak
        md = lower_idx + peaks[-1]
    else:
        if lower_idx < upper_idx:
            md = (lower_idx + upper_idx) // 2  # direct middle
        else:
            md = (upper_idx + length + lower_idx) // 2  # middle across the cycle border

    md = np.mod(md, length)
    return md


def get_ed(dir_1d_mean, lower_idx, upper_idx, length, first_transition=True):
    import numpy as np
    if lower_idx < upper_idx:
        cycle = dir_1d_mean[lower_idx:upper_idx]
    else:
        cycle = np.concatenate([dir_1d_mean[lower_idx:], dir_1d_mean[:upper_idx]])

    temp_ = 0
    ed_found = False
    last_idx_positive = True  # we start at the md, which is the peak(dir)
    for idx, elem in enumerate(cycle):
        # this enables to find the last transition from pos to neg
        if elem >= 0:
            last_idx_positive = True
        # remember the last idx before the direction gets negative the last time before ms
        elif elem < 0 and last_idx_positive:  # first time direction negative
            ed_found = True  # for fallbacks
            temp_ = idx  # idx before negative direction
            # print('found transition at: {}, {}'.format(lower_idx+idx, cycle))
            last_idx_positive = False  # remember only the first idx after transition
            if first_transition:
                break
    if ed_found:
        ed = lower_idx + temp_
        # print('ed:{}, pf:{}, temp_:{}, lenght: {}'.format(ed,pf,temp_,length))
    else:
        # if we dont find a transition from positive to negative, take the idx which is the closest to zero
        temp_ = np.argmin(np.abs(cycle))  # make sure we have a minimal distance
        ed = lower_idx + temp_
        # print('ED: no transition found between {}-{} ,len: {} closest id to 0: {}, ed = {}'.format(upper_idx, lower_idx, length, ed,np.mod(ed, length)))
    ed = np.mod(ed, length)
    return ed


def get_pf(dir_1d_mean, lower_idx, upper_idx):
    import numpy as np
    import scipy.signal as sig
    length = len(dir_1d_mean)
    lower_idx = np.mod(lower_idx, length)

    if lower_idx < upper_idx:
        cycle = dir_1d_mean[lower_idx:upper_idx]
    else:
        cycle = np.concatenate([dir_1d_mean[lower_idx:], dir_1d_mean[:upper_idx]])

    peaks, prop = sig.find_peaks(cycle, prominence=(
        None, None))  # height=0.6 we normalise between -1 and 1, PF should be close to argmax
    if len(peaks) == 0:
        pf_max = np.argmax(cycle)
        pf = lower_idx + pf_max
    elif len(peaks) in [1, 2]:  # two peaks, take the first, leave the second as MD
        pf = lower_idx + peaks[0]  + 1
    else:  # more than 2 peaks
        # more than 3 peaks
        # sort the first two peaks by promminence or absolute value, take the bigger one
        peak_values = cycle[peaks]
        # get the idx of two biggest (value or prominence) peaks
        peaks = [p for p, _ in sorted(zip(peaks, peak_values), key=lambda pair: pair[1],
                                      reverse=True)][:2]  # or use the prominence: prop['prominences']
        pf = lower_idx + min(peaks)  # take the peak, that appears first
    pf = np.mod(pf, length)
    return pf


def get_es(dir_1d_mean, lower_idx, length, upper_idx=None, first_transition=True):
    """
    Determines timestep of end-systole (ES). First time f(x)>0 after MS. ES should be between MS and approx PF == argmax
    Args:
        dir_1d_mean ():
        length ():
        lower_idx ():
        upper_idx ():

    Returns:
        time step of end-systole (es)

    """
    import numpy as np
    if upper_idx is None:
        upper_idx = np.argmax(dir_1d_mean)

    if lower_idx < upper_idx:
        cycle = dir_1d_mean[lower_idx:upper_idx]
    else:
        cycle = np.concatenate([dir_1d_mean[lower_idx:], dir_1d_mean[:upper_idx]])
    temp_ = 0
    es_found = False
    negative_slope = False
    for idx, elem in enumerate(cycle):
        if elem < 0:
            negative_slope = True
        elif elem > 0 and negative_slope:
            es_found = True
            temp_ = idx
            negative_slope = False
            if first_transition:
                break  # stop after first zero-transition
    if es_found:
        es = lower_idx + temp_

    else:
        es = lower_idx + 1  # + len(cycle)//2  # fallback: middle between min and max
        print('no ES found between: {} - {} set to ms+1={}'.format(lower_idx, upper_idx, lower_idx + 1))
    if es >= length:
        logging.debug('ES overflow: {}, ms:{}'.format(es, lower_idx))
    es = np.mod(es, length)
    return es


def get_ms(dir_1d_mean, length):
    """
    Determines timestep of mid-systole (MS), by global min of f(x).
    Args:
        dir_1d_mean (MaskedArray): one-dimensional motion descriptor
    Returns:
        timestep of mid-systole (MS)
    """
    import numpy as np
    ms = np.argmin(dir_1d_mean)
    ms = np.mod(ms, length)
    return ms