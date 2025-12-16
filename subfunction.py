import math

"""
def get_frac(a):
    
    # a에 따른 k 자동 설정
    if a == 0.02:
        k = 0.0025
    elif a == 0.03:
        k = 0.0038
    elif a == 0.04:
        k = 0.0049
    elif a == 0.05:
        k = 0.0060
    else:
        raise ValueError("a 값은 0.02, 0.03, 0.04, 0.05 중 선택하세요.")

    return a, k
"""
def get_max_frac_new(iteration):
    if iteration < 3000:
        return 0.05  # 3%
    elif iteration < 6000:
        return 0.04   # 2%
    elif iteration < 9000:
        return 0.03   # 1%
    elif iteration < 15000:
        return 0.02
    elif iteration < 27000:
        return 0.01
    else:
        return 0.0
    """
    step = iteration // 100         # 0,1,2,... (최대 150번 정도라고 가정)
    f0 = 0.04
    k = 0.0049
    return f0 * math.exp(-k * step)
    """


def get_opacity_threshold(iteration):
    if iteration < 10000:
        return 0.02  # 3%
    else:
        return 0.015