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
    if iteration < 5000:
        return 0.03  # 3%
    elif iteration < 15000:
        return 0.02   # 2%
    elif iteration < 27000:
        return 0.01   # 1%
    else:
        return 0.0
    """
    step = iteration // 100         # 0,1,2,... (최대 150번 정도라고 가정)
    f0 = 0.04
    k = 0.0049
    return f0 * math.exp(-k * step)
    """