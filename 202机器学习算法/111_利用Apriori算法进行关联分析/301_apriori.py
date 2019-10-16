

def load_data_set():
    """
    构建数据集
    :return: 二维数组
    """
    return [
        [1, 3, 4],
        [2, 3, 5],
        [1, 2, 3, 5],
        [2, 5]
    ]


def create_c1(data_set):
    """
    将所有元素转换为frozenset型字典，存放到列表中
    """
    c1 = []
    for transaction in data_set:
        for item in transaction:
            if not [item] in c1:
                c1.append([item])
    c1.sort()
    # 使用frozenset是为了后面可以将这些值作为字典的键
    return list(map(frozenset, c1))  # frozenset一种不可变的集合，set可变集合


def scan_d(d, ck, min_support):
    """
    过滤掉不符合支持度的集合
    :param d: 数据集合
    :param ck: 需要求的组合
    :param min_support: 最小支持度
    :return: 频繁项集列表retList 所有元素的支持度字典
    """
    ss_cnt = {}
    for tid in d:
        for can in ck:
            if can.issubset(tid):  # 判断can是否是tid的《子集》 （这里使用子集的方式来判断两者的关系）
                if can not in ss_cnt:  # 统计该值在整个记录中满足子集的次数（以字典的形式记录，frozenset为键）
                    ss_cnt[can] = 1
                else:
                    ss_cnt[can] += 1
    num_items = float(len(d))
    ret_list = []  # 重新记录满足条件的数据值（即支持度大于阈值的数据）
    support_data = {}  # 每个数据值的支持度
    for key in ss_cnt:
        support = ss_cnt[key] / num_items
        if support >= min_support:
            ret_list.insert(0, key)
        support_data[key] = support
    return ret_list, support_data  # 排除不符合支持度元素后的元素 每个元素支持度


def apriori_gen(lk, k):
    """
    生成所有可以组合的集合
    频繁项集列表Lk 项集元素个数k  [frozenset({2, 3}), frozenset({3, 5})] -> [frozenset({2, 3, 5})]
    """
    ret_list = []
    len_lk = len(lk)
    for i in range(len_lk):  # 两层循环比较Lk中的每个元素与其它元素
        for j in range(i + 1, len_lk):
            L1 = list(lk[i])[:k - 2]  # 将集合转为list后取值
            L2 = list(lk[j])[:k - 2]
            L1.sort();
            L2.sort()  # 这里说明一下：该函数每次比较两个list的前k-2个元素，如果相同则求并集得到k个元素的集合
            if L1 == L2:
                ret_list.append(lk[i] | lk[j])  # 求并集
    return ret_list  # 返回频繁项集列表Ck



def apriori(data_set, min_support=0.5):
    """
    封装所有步骤的函数
    返回 所有满足大于阈值的组合 集合支持度列表
    """
    d = list(map(set, data_set))  # 转换列表记录为字典  [{1, 3, 4}, {2, 3, 5}, {1, 2, 3, 5}, {2, 5}]
    c1 = create_c1(
        data_set)  # 将每个元素转会为frozenset字典    [frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5})]
    L1, supportData = scan_d(d, c1, min_support)  # 过滤数据
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):  # 若仍有满足支持度的集合则继续做关联分析
        Ck = apriori_gen(L[k - 2], k)  # Ck候选频繁项集
        Lk, supK = scan_d(d, Ck, min_support)  # Lk频繁项集
        supportData.update(supK)  # 更新字典（把新出现的集合:支持度加入到supportData中）
        L.append(Lk)
        k += 1  # 每次新组合的元素都只增加了一个，所以k也+1（k表示元素个数）
    return L, supportData


# 获取关联规则的封装函数
def generateRules(L, supportData, minConf=0.7):  # supportData 是一个字典
    bigRuleList = []
    for i in range(1, len(L)):  # 从为2个元素的集合开始
        for freqSet in L[i]:
            # 只包含单个元素的集合列表
            H1 = [frozenset([item]) for item in freqSet]  # frozenset({2, 3}) 转换为 [frozenset({2}), frozenset({3})]
            # 如果集合元素大于2个，则需要处理才能获得规则
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)  # 集合元素 集合拆分后的列表 。。。
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


# 对规则进行评估 获得满足最小可信度的关联规则
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []  # 创建一个新的列表去返回
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]  # 计算置信度
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


# 生成候选规则集合
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):  # 尝试进一步合并
        Hmp1 = apriori_gen(H, m + 1)  # 将单个集合元素两两合并
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):  # need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


dataSet = load_data_set()
L, suppData = apriori(dataSet, min_support = 0.5)
rules = generateRules(L, suppData, minConf=0.7)
# rules = generateRules(L,suppData,minConf=0.5)
print(rules)


"""
data_set = load_data_set()
L, suppData = apriori(data_set)
print(L)
print(suppData)
"""


