import math


# ============================================================
# Attention Python Golden Model
#
# 对应当前 RTL:
#   S = 8
#   d = 4
#
# Q/K/V:
#   signed 8-bit integer
#
# Weight:
#   UQ1.4
#   fixed = round(exp(score - x_max) * 16)
#
# Output:
#   O = round(ACC / L)
#
# 四舍五入规则：
#   nearest, ties away from zero
#
# 注意：
#   本程序故意不使用 Python 内置 round()
#   因为 Python round() 使用 ties-to-even。
# ============================================================

S = 8
D = 4

WEIGHT_SCALE = 16


# ============================================================
# signed 8-bit 检查
# ============================================================

def check_s8(value):
    if value < -128 or value > 127:
        raise ValueError(
            f"{value} is outside signed 8-bit range [-128, 127]"
        )


# ============================================================
# 非负数四舍五入
#
# 只用于 exp() * 16。
# exp() 的结果一定 >= 0。
#
# 例如：
# 5.886 -> 6
# 2.165 -> 2
# 0.797 -> 1
# 0.293 -> 0
# ============================================================

def round_nonnegative(value):
    return int(
        math.floor(value + 0.5)
    )


# ============================================================
# EXP Golden Model
#
# Python 真正调用 math.exp()
# 而不是复制 RTL LUT。
#
# 这是 Golden Model 独立性的关键。
# ============================================================

def exp_fixed(x):

    if x > 0:
        raise ValueError(
            f"score - x_max must <= 0, actual = {x}"
        )

    real_weight = math.exp(x)

    scaled_weight = (
        real_weight
        *
        WEIGHT_SCALE
    )

    fixed_weight = round_nonnegative(
        scaled_weight
    )

    # UQ1.4，本题 exp 最大为 1
    # 因此硬件范围为 0 ~ 16
    fixed_weight = max(
        0,
        min(16, fixed_weight)
    )

    return fixed_weight


# ============================================================
# RTL 对应的整数除法四舍五入
#
# 不使用 Python round()。
#
# 实现：
#
# 正数：
#     (ACC + L/2) / L
#
# 负数：
#     (ACC - L/2) / L
#
# 效果：
#      3.5 ->  4
#     -3.5 -> -4
#
# 即：
# nearest, ties away from zero
# ============================================================

def round_div_away_from_zero(
    numerator,
    denominator
):

    if denominator <= 0:
        raise ValueError(
            f"denominator must > 0, actual = {denominator}"
        )

    half = denominator // 2

    if numerator >= 0:

        return (
            numerator + half
        ) // denominator

    else:

        return -(
            (
                -numerator + half
            )
            // denominator
        )


# ============================================================
# signed 8-bit 饱和保护
# ============================================================

def saturate_s8(value):

    if value > 127:
        return 127

    if value < -128:
        return -128

    return value


# ============================================================
# Golden Model 主函数
# ============================================================

def attention_golden(
    q,
    k,
    v
):

    # --------------------------------------------------------
    # 维度检查
    # --------------------------------------------------------

    if len(q) != D:
        raise ValueError(
            f"Q length must be {D}"
        )

    if len(k) != S:
        raise ValueError(
            f"K token count must be {S}"
        )

    if len(v) != S:
        raise ValueError(
            f"V token count must be {S}"
        )


    # --------------------------------------------------------
    # signed 8-bit 输入检查
    # --------------------------------------------------------

    for d in range(D):
        check_s8(q[d])

    for t in range(S):

        if len(k[t]) != D:
            raise ValueError(
                f"K[{t}] length must be {D}"
            )

        if len(v[t]) != D:
            raise ValueError(
                f"V[{t}] length must be {D}"
            )

        for d in range(D):

            check_s8(k[t][d])
            check_s8(v[t][d])


    # ========================================================
    # STEP 1
    # Score[t] = Q · K[t]
    # ========================================================

    score = []

    for t in range(S):

        value = 0

        for d in range(D):

            value += (
                q[d]
                *
                k[t][d]
            )

        score.append(value)


    # ========================================================
    # STEP 2
    # x_max
    # ========================================================

    x_max = max(score)


    # ========================================================
    # STEP 3
    # score - x_max
    # ========================================================

    score_shifted = []

    for t in range(S):

        shifted = (
            score[t]
            -
            x_max
        )

        # 正常 Attention 中必须 <= 0
        assert shifted <= 0

        score_shifted.append(
            shifted
        )


    # ========================================================
    # STEP 4
    # Weight
    #
    # fixed =
    # round(exp(score-x_max) * 16)
    # ========================================================

    weight = []

    for t in range(S):

        weight.append(
            exp_fixed(
                score_shifted[t]
            )
        )


    # ========================================================
    # STEP 5
    # L
    # ========================================================

    L = sum(weight)


    # 至少有一个 Score 等于 x_max
    # 因此至少一个 weight = 16
    assert L >= 16


    # ========================================================
    # STEP 6
    # ACC[d]
    # ========================================================

    ACC = [0] * D

    for d in range(D):

        for t in range(S):

            ACC[d] += (
                weight[t]
                *
                v[t][d]
            )


    # ========================================================
    # STEP 7
    # O[d] = round(ACC[d] / L)
    # ========================================================

    O = []

    for d in range(D):

        result = round_div_away_from_zero(
            ACC[d],
            L
        )

        result = saturate_s8(
            result
        )

        O.append(result)


    return {
        "score": score,
        "x_max": x_max,
        "score_shifted": score_shifted,
        "weight": weight,
        "L": L,
        "ACC": ACC,
        "O": O
    }


# ============================================================
# 打印
# ============================================================

def print_result(
    case_name,
    result
):

    print()
    print(
        "============================================================"
    )

    print(
        case_name
    )

    print(
        "============================================================"
    )

    print(
        "Score         =",
        result["score"]
    )

    print(
        "x_max         =",
        result["x_max"]
    )

    print(
        "Score-x_max   =",
        result["score_shifted"]
    )

    print(
        "Weight fixed  =",
        result["weight"]
    )

    print(
        "L             =",
        result["L"]
    )

    print(
        "ACC           =",
        result["ACC"]
    )

    print(
        "O             =",
        result["O"]
    )


# ============================================================
# 四舍五入自检
# ============================================================

def rounding_self_test():

    assert (
        round_div_away_from_zero(7, 2)
        ==
        4
    )

    assert (
        round_div_away_from_zero(-7, 2)
        ==
        -4
    )

    assert (
        round_div_away_from_zero(6, 2)
        ==
        3
    )

    assert (
        round_div_away_from_zero(-6, 2)
        ==
        -3
    )

    print(
        "ROUNDING SELF-CHECK PASSED"
    )

    print(
        "  3.5  ->  4"
    )

    print(
        " -3.5  -> -4"
    )


# ============================================================
# CASE 1
#
# 对应 Vivado TB：
# DIRECTED CASE 1 - HAND EXAMPLE
# ============================================================

def case_1():

    Q = [
        1,
        2,
        0,
        0
    ]

    K = [
        [1,   0, 0, 0],
        [0,   1, 0, 0],
        [-10, -10, 0, 0],
        [-10, -10, 0, 0],
        [-10, -10, 0, 0],
        [-10, -10, 0, 0],
        [-10, -10, 0, 0],
        [-10, -10, 0, 0]
    ]

    V = [
        [4, 2, 0, 0],
        [1, 6, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]

    result = attention_golden(
        Q,
        K,
        V
    )

    print_result(
        "CASE 1 - HAND EXAMPLE",
        result
    )

    assert result["score"] == [
        1,
        2,
        -30,
        -30,
        -30,
        -30,
        -30,
        -30
    ]

    assert result["x_max"] == 2

    assert result["weight"] == [
        6,
        16,
        0,
        0,
        0,
        0,
        0,
        0
    ]

    assert result["L"] == 22

    assert result["ACC"] == [
        40,
        108,
        0,
        0
    ]

    assert result["O"] == [
        2,
        5,
        0,
        0
    ]

    print(
        "CASE 1 CHECK PASSED"
    )


# ============================================================
# CASE 2
#
# 专门验证：
#
#  3.5 ->  4
# -3.5 -> -4
#
# 对应你的 Vivado ROUNDING CASE。
# ============================================================

def case_2():

    Q = [
        0,
        0,
        0,
        0
    ]

    K = [
        [0, 0, 0, 0]
        for _ in range(S)
    ]

    V = []

    for t in range(S):

        V.append([
            t,
            t * 2,
            -t,
            4
        ])

    result = attention_golden(
        Q,
        K,
        V
    )

    print_result(
        "CASE 2 - ROUNDING",
        result
    )

    assert result["weight"] == [
        16,
        16,
        16,
        16,
        16,
        16,
        16,
        16
    ]

    assert result["L"] == 128

    assert result["ACC"] == [
        448,
        896,
        -448,
        512
    ]

    assert result["O"] == [
        4,
        7,
        -4,
        4
    ]

    print(
        "CASE 2 CHECK PASSED"
    )


# ============================================================
# CASE 3
#
# 极限位宽测试
#
# Q:
# [-128,-128,-128,-128]
#
# K0:
# [-128,-128,-128,-128]
#
# Score0 = 65536
#
# K1:
# [127,127,127,127]
#
# Score1 = -65024
#
# Score1-x_max:
# -130560
#
# 验证 18-bit Score / shifted Score 极限。
# ============================================================

def case_3():

    Q = [
        -128,
        -128,
        -128,
        -128
    ]

    K = [
        [-128, -128, -128, -128],
        [127, 127, 127, 127],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]

    V = [
        [-128, 127, -128, 127],
        [10, 20, 30, 40],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]

    result = attention_golden(
        Q,
        K,
        V
    )

    print_result(
        "CASE 3 - EXTREME WIDTH",
        result
    )

    assert result["score"][0] == 65536

    assert result["score"][1] == -65024

    assert (
        result["score_shifted"][1]
        ==
        -130560
    )

    assert result["weight"][0] == 16

    assert result["L"] == 16

    assert result["O"] == [
        -128,
        127,
        -128,
        127
    ]

    print(
        "CASE 3 CHECK PASSED"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "============================================================"
    )

    print(
        "             PYTHON GOLDEN MODEL"
    )

    print(
        "============================================================"
    )

    rounding_self_test()

    case_1()

    case_2()

    case_3()

    print()
    print(
        "============================================================"
    )

    print(
        "       ALL GOLDEN MODEL TESTS PASSED"
    )

    print(
        "============================================================"
    )