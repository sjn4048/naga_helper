import functools
import json
import math
import re
from collections import defaultdict
import demjson3 as demjson
import numpy as np
from bs4 import BeautifulSoup
from mahjong.shanten import Shanten
from mahjong.tile import TilesConverter
import sys
import requests


def merge_tenhou_log_to_one(records: list[str], returns_prefix: bool = True) -> str:
    # 使用第一个牌谱的title, name, rule，合并所有牌谱的log
    logs = [json.loads(_r.replace('https://tenhou.net/6/#json=', ''))['log'] for _r in records]  # 获取JSON字符串
    # 解析URL
    first_json_str = records[0].replace('https://tenhou.net/6/#json=', '')

    # 解析JSON字符串
    first_json_data = json.loads(first_json_str)
    # 更新JSON数据
    first_json_data['log'] = sum(logs, [])
    ret = json.dumps(first_json_data, ensure_ascii=False, separators=(',', ':'))
    if returns_prefix:
        return 'https://tenhou.net/6/#json=' + ret
    return ret


_naga_B = {
    "1m": 0,
    "2m": 1,
    "3m": 2,
    "4m": 3,
    "5m": 4,
    "5mr": 4.1,
    "6m": 5,
    "7m": 6,
    "8m": 7,
    "9m": 8,
    "1p": 9,
    "2p": 10,
    "3p": 11,
    "4p": 12,
    "5p": 13,
    "5pr": 13.1,
    "6p": 14,
    "7p": 15,
    "8p": 16,
    "9p": 17,
    "1s": 18,
    "2s": 19,
    "3s": 20,
    "4s": 21,
    "5s": 22,
    "5sr": 22.1,
    "6s": 23,
    "7s": 24,
    "8s": 25,
    "9s": 26,
    "E": 27,
    "S": 28,
    "W": 29,
    "N": 30,
    "P": 31,
    "F": 32,
    "C": 33,
    "?": 34,
    "1z": 27,
    "2z": 28,
    "3z": 29,
    "4z": 30,
    "5z": 31,
    "6z": 32,
    "7z": 33,
}

_naga_B_rev = {v: k for k, v in _naga_B.items()}

# NAGA定义的副露类型
_naga_huro_types = {
    'pass': '0',
    'chi1': '1',  # 吃最小的牌
    'chi2': '2',  # 吃中间的牌
    'chi3': '3',  # 吃最大的牌
    'pon': '4',
    'kan': '5'
}

_naga_replace_d = {
    'E': '1z',
    'S': '2z',
    'W': '3z',
    'N': '4z',
    'P': '5z',
    'F': '6z',
    'C': '7z'
}

_total_hais = [f'{x}s' for x in range(1, 10)] + [f'{x}p' for x in range(1, 10)] + [f'{x}m' for x in range(1, 10)] + [
    f'{x}z' for x in range(1, 8)]


def _to_mortal_text(naga_text: str) -> str:
    naga_text = merge_tenhou_log_to_one(naga_text.strip().split('\n'))

    prevalents = ["東", "南", "西", "北"]
    for p in prevalents:
        naga_text = naga_text.replace(f'場風 {p}(1飜)', '役牌:場風牌(1飜)')
        naga_text = naga_text.replace(f'自風 {p}(1飜)', '役牌:自風牌(1飜)')
    naga_text = naga_text.replace('ダブル立直(2飜)"', '両立直(2飜)')
    naga_text = naga_text.replace('https://tenhou.net/6/#json=', '')  # 去除前缀，只取json
    return naga_text


def _get_naga_var(text: str) -> dict[str, ...]:
    soup = BeautifulSoup(text, 'html.parser')
    # read variables
    script_tags = soup.find_all('script')
    variables_dict: dict[..., ...] = {}

    whitelist_keys = ['pred', 'playerInfo', 'haihuId', 'nagaVersion', 'gameType', 'nagaTypes',
                      'dataVersion', 'reportInfo', 'customHaihu']  # 注入脚本中的变量，不要修改

    for script in script_tags:
        # 使用正则表达式找到所有的变量赋值
        matches = re.finditer(r'const\s+(\w+)\s*=\s*(.*?)\n', script.string if script.string else '')
        for match in list(matches):
            var_name = match.group(1)
            var_value = match.group(2)
            if var_name not in whitelist_keys:
                continue
            try:
                var_value = json.loads(var_value.replace("'", '"'))
            except json.decoder.JSONDecodeError:
                try:
                    var_value = demjson.decode(var_value.replace("'", '"'))
                except demjson.JSONDecodeError:
                    pass
            variables_dict[var_name] = var_value
    return variables_dict


def _write_back_to_naga(var_dict: dict[str, ...], orig_text: str) -> str:
    soup = BeautifulSoup(orig_text, 'html.parser')
    script_tags = soup.find_all('script')

    for script in script_tags:
        if script.string:
            script_content = script.string
            for var_name, var_value in var_dict.items():
                # 将变量值转换为JSON字符串
                # 对nagaTypes特殊处理
                if var_name == 'nagaTypes':
                    var_value_str = '{' + ', '.join(
                        f"{int(k)}: '{v}'" for k, v in var_value.items()) + '}'  # 允许整数作为key类型，与原始naga保持一致
                else:
                    var_value_str = json.dumps(var_value, ensure_ascii=False, separators=(',', ':'))
                # 使用正则表达式替换变量赋值
                script_content = re.sub(
                    rf'(const\s+{var_name}\s*=\s*)(.*?)\n',
                    rf'const {var_name} = {var_value_str}\n',
                    script_content
                )
            script.string.replace_with(script_content)

    return str(soup)


def _check_if_can_merge(naga_pred: list, mortal_data: dict[str, ...]):
    naga_game_count = len(naga_pred)
    mortal_game_count = len(mortal_data['review']['kyokus'])
    if naga_game_count != mortal_game_count:
        # print(f'Naga game count: {naga_game_count}, Mortal game count: {mortal_game_count}')
        return False
    # print(f'Naga game count: {naga_game_count}, Mortal game count: {mortal_game_count}')
    return True


def _get_riichi_pai(tehai: list[str]) -> list[str]:
    # 计算打哪张牌可以立直
    riichi_candidates = []
    assert len(tehai) % 3 == 2, tehai  # 可能有暗杠
    st = Shanten()

    for t in set(tehai):
        # 遍历每种牌，看看是否去掉之后向听数仍然为0
        tehai_cp = tehai[:]
        tehai_cp.remove(t)
        new_st = st.calculate_shanten(_naga_tehai_to_tiles(tehai_cp, None))
        if new_st == 0:
            riichi_candidates.append(t)

    return riichi_candidates


def merge_mortal_to_naga(naga_text: str, mortal_text: str) -> str:
    naga_replace_d_rev = {v: k for k, v in _naga_replace_d.items()}

    try:
        mortal_data = json.loads(mortal_text)
    except json.decoder.JSONDecodeError:
        print('Cannot load mortal_text')
        return naga_text
    naga_dict = _get_naga_var(naga_text)
    can_merge = _check_if_can_merge(naga_dict['pred'], mortal_data)
    if not can_merge:
        print('Cannot merge mortal and naga. Try anyway')
    # add Mortal to naga player types
    naga_types: dict | None = naga_dict.get('nagaTypes', None)
    if naga_types is None:
        raise NotImplementedError('Does not support oldest naga reports')
    mortal_idx = max(naga_types.keys()) + 1
    naga_types[mortal_idx] = 'Mortal'
    naga_dict['nagaTypes'] = naga_types
    naga_prob_sum = 10000
    m_actor_id = mortal_data['player_id']

    def _normalize_to_sum(lst: list[float], sum_v: int, precise: bool = False) -> list[float]:
        if not lst:
            return []

        if sum(lst) == 0:
            return lst

        # 计算放缩因子
        total = sum(lst)
        scale_factor = sum_v / total

        # 初步放缩并取整
        scaled_lst = [int(round(x * scale_factor)) for x in lst]

        if not precise:
            return scaled_lst

        # 计算初步放缩后的和
        current_sum = sum(scaled_lst)

        # 计算误差
        error = sum_v - current_sum

        # 调整误差
        if error != 0:
            # 计算每个元素的误差
            fractional_parts = [(x * scale_factor) - int(x * scale_factor) for x in lst]
            # 按误差的绝对值排序
            sorted_indices = sorted(range(len(lst)), key=lambda _i: fractional_parts[_i], reverse=(error > 0))

            # 调整误差
            for i in sorted_indices:
                if error == 0:
                    break
                adjustment = 1 if error > 0 else -1
                scaled_lst[i] += adjustment
                error -= adjustment

        return scaled_lst

    # add Mortal decisions to each round
    # enumerate games (naga and mortal)
    mgs = iter(mortal_data['review']['kyokus'])

    for game_idx, ng in enumerate(naga_dict['pred']):
        # 双链表结构，外层遍历mortal的切牌（因为更少）；内层遍历naga的切牌，且保持index递增。对照条件为left_hai_num相等
        # m_turns中，具体切牌结构在 'details' (list) 中，每个元素为 {action: {type, actor, pai, tsumogiri}, prob}.
        # n_turns中，具体切牌结构在 dahai_pred 中（可能不存在，不存在则跳过）
        # 如果两者对应上，则将n_turns每个action中的pai与对应prob写入turn['dahai_pred'][mortal_idx]中
        n_turns = ng

        # 寻找对应的Mortal的牌谱；可能有三种情况：完美匹配、Mortal缺失该局、Mortal中存在NAGA没有的对局
        bakaze_base = {
            'E': 0,
            'S': 1,
            'W': 2,
            'N': 3,
        }
        n_kyoku = (bakaze_base[n_turns[0]['info']['msg']['bakaze']] * 4 + n_turns[0]['info']['msg']['kyoku'] - 1,
                   n_turns[0]['info']['msg']['honba'])

        has_mortal_game = False

        while True:
            mg = next(mgs, None)
            if not mg:
                break  # 没有更多Mortal对局了
            m_kyoku = (mg['kyoku'], mg['honba'])
            if m_kyoku == n_kyoku:
                # print(f'Found match game: {m_kyoku}')
                has_mortal_game = True
                break  # 找到了对局

        if has_mortal_game:
            m_turns = mg['entries']

            m_turns_iter = iter(m_turns)
            m_turn = next(m_turns_iter, None)
        else:
            m_turn, m_turns_iter = None, None

        for turn_idx, n_turn in enumerate(n_turns):
            try:
                n_actor_id = n_turn['info']['msg']['actor']
            except (Exception,):
                # 一般是因为start_kyoku等全局事件，此时不做解析、不做修改
                continue

            n_left_hai_num = n_turn['info']['msg']['left_hai_num']
            is_naki_turn = False

            # 做保底的修改。注意要赋值值而不是引用，否则后面会改挂
            if 'huro' in n_turn:
                for k in n_turn['huro']:
                    n_turn['huro'][k].append(n_turn['huro'][k][0].copy())
            if 'kan' in n_turn:
                n_turn['kan'].append(n_turn['kan'][0])
            if 'reach' in n_turn:
                n_turn['reach'].append(n_turn['reach'][0])
            if 'dahai_pred' in n_turn:
                n_turn['dahai_pred'].append([0] * 34)
                n_turn['info']['msg']['pred_dahai'].append(n_turn['info']['msg']['pred_dahai'][0][:])
            else:
                # 没有dahai_pred直接continue
                # continue
                pass
            if not has_mortal_game:
                continue

            while m_turn and m_turn['tiles_left'] > n_turn['info']['msg']['left_hai_num']:
                m_turn = next(m_turns_iter, None)

            if 'huro' in n_turn:
                can_naki_ids = list(n_turn['huro'].keys())  # 副露信息会放在上一个人的切牌后面
                if str(m_actor_id) in can_naki_ids:
                    # 注意这里有一个小区别，如果上家打出的牌被碰走，NAGA仍然会有鸣牌条，但Mortal没有。
                    n_actor_id = m_actor_id  # 此时就需要处理
                    is_naki_turn = True
                    # print(f'{game_idx}-{n_left_hai_num} huro_info: {n_turn["huro"][str(n_actor_id)][0]}')

            if m_turn and m_turn['tiles_left'] == n_left_hai_num and n_actor_id == m_actor_id:
                m_pred = [0] * 34
                if is_naki_turn:
                    huro_info = n_turn['huro'][str(m_actor_id)][-1]
                else:
                    huro_info = {}

                # 下面的解析分为两次遍历；第一次获取是否有加杠/立直等meta信息，第二次真正做数据处理
                max_dahai_prob = 0
                sum_dahai_prob = 0  # 计算所有切牌（非立直、非加杠）概率之和。注意，这里要使用累加切牌概率，而不是1减去立直/加杠的概率，因为有时候Mortal的prob给的过于极端，使1 - riichi_prob_sum - kan_prob_sum之后为0，导致除0异常
                can_dahai = False
                for ma in m_turn['details']:
                    at = ma['action']['type']
                    ap = ma['prob']
                    if at == 'dahai':
                        can_dahai = True
                        sum_dahai_prob += ap
                        max_dahai_prob = max(max_dahai_prob, ap)

                for m_action in m_turn['details']:
                    action = m_action['action']

                    if action['type'] == 'dahai':
                        m_pred[int(_naga_B[action['pai']])] += math.ceil(
                            m_action['prob'] / sum_dahai_prob * naga_prob_sum)
                    if action['type'] == 'reach':
                        # dama自摸的时候可能没有reach条
                        if 'reach' in n_turn:
                            # 为了让立直条过线，立直概率不需要超过50%，仅需超过所有非立直切牌的max(prob)即可
                            n_turn['reach'][-1] = math.ceil(
                                m_action['prob'] / (max_dahai_prob + m_action['prob']) * naga_prob_sum)

                    if action['type'] == 'pon':
                        pon_prob = math.ceil(m_action['prob'] * naga_prob_sum)
                        huro_info[_naga_huro_types['pon']] = pon_prob
                        # print(f'pon: {pon_prob}')
                    if action['type'] == 'none':
                        none_prob = math.ceil(m_action['prob'] * naga_prob_sum)
                        huro_info[_naga_huro_types['pass']] = none_prob
                        # print(f'none: {none_prob}')
                    if action['type'] == 'chi':
                        # 需要判定是哪一种吃
                        chi_pai = int(action['pai'][0])
                        consumed_pai = [int(x[0]) for x in action['consumed']]
                        if chi_pai < consumed_pai[0] and chi_pai < consumed_pai[1]:
                            naki_act = 'chi1'
                        elif chi_pai > consumed_pai[0] and chi_pai > consumed_pai[1]:
                            naki_act = 'chi3'
                        else:
                            naki_act = 'chi2'
                        chi_prob = math.ceil(m_action['prob'] * naga_prob_sum)
                        huro_info[_naga_huro_types[naki_act]] = chi_prob
                        # print(f'{naki_act}: {chi_prob}')
                    if action['type'] == 'kan':
                        if can_dahai:
                            # 同理，暗杠/加杠概率不需要超过50%，仅需超过所有非加杠切牌的max(prob)即可
                            kan_prob = math.ceil(m_action['prob'] / (max_dahai_prob + m_action['prob']) * naga_prob_sum)
                        else:
                            # 大明杠正常处理
                            kan_prob = math.ceil(m_action['prob'] * naga_prob_sum)

                        n_turn['kan'][-1] = kan_prob
                        huro_info[_naga_huro_types['kan']] = kan_prob
                        # print(f'kan: {kan_prob}')
                if is_naki_turn:
                    # print(f'changed huro_info: {huro_info}')
                    n_turn['huro'][str(m_actor_id)][-1] = huro_info
                    # print(n_turn['huro'])

                # 宣言和立直在Mortal是两个事件，NAGA是一个事件，需要把宣言牌的概率加到NAGA上
                player_chosen_riichi = m_turn['actual']['type'] == 'reach'
                if player_chosen_riichi:
                    # Mortal立直概率
                    riichi_candidates = _get_riichi_pai(m_turn['state']['tehai'])
                    # print(f'player chosen riichi: {riichi_candidates} ({game_idx})')
                    for m_action in m_turn['details']:
                        if m_action['action']['type'] == 'reach':
                            m_riichi_prob = m_action['prob']
                            break
                    else:
                        raise ValueError(f'Cannot find reach action from Mortal when player called riichi: {m_turn}')

                    # 将立直条赋值给立直巡目
                    try:
                        riichi_m_turn = next(m_turns_iter)
                    except StopIteration:
                        riichi_m_turn = None
                    if riichi_m_turn and riichi_m_turn['tiles_left'] == m_turn['tiles_left']:  # 应对多种立直选择
                        for a in riichi_m_turn['details']:
                            if a['action']['type'] == 'dahai':
                                m_pred[int(_naga_B[a['action']['pai']])] += int(
                                    a['prob'] * naga_prob_sum * m_riichi_prob)
                    else:
                        # 只有一种立直选择
                        assert len(riichi_candidates) == 1, riichi_candidates
                        m_pred[int(_naga_B[riichi_candidates[0]])] += naga_prob_sum * m_riichi_prob

                m_pred = _normalize_to_sum(m_pred, naga_prob_sum, precise=False)
                # 将Mortal结果写回NAGA
                if 'dahai_pred' in n_turn:
                    n_turn['dahai_pred'][-1] = m_pred
                    n_turn['info']['msg']['pred_dahai'][-1] = (
                        naga_replace_d_rev.get(_naga_B_rev[np.argmax(m_pred)], _naga_B_rev[np.argmax(m_pred)]))
                try:
                    m_turn = next(m_turns_iter, None)  # 本turn信息使用完毕，跳下一turn
                except StopIteration:
                    pass

    return _write_back_to_naga(naga_dict, naga_text)


def _naga_tehai_to_tiles(tehai: list[str], nakis: list[str] = None):
    if nakis is None:
        nakis = []
    tehai = tehai.copy()
    for n in nakis:
        try:
            tehai.remove(n)
        except ValueError as ex:
            print(f'[ERROR] {n} not in {tehai}.')
            raise ex

    ms, ps, ss, zs = [], [], [], []
    tehai = sorted([_naga_replace_d.get(x, x) for x in tehai])
    for t in tehai:
        if t[1] == 'm':
            ms.append(t[0])
        if t[1] == 'p':
            ps.append(t[0])
        if t[1] == 's':
            ss.append(t[0])
        if t[1] == 'z':
            zs.append(t[0])
    return TilesConverter.string_to_34_array(man=''.join(ms), pin=''.join(ps), sou=''.join(ss), honors=''.join(zs))


def _calculate_maisuu(tehai: list[str], visible_maisuu: dict[str, int], nakis: list[str] = None) \
        -> tuple[int, list[str]]:
    # 计算进张枚数
    result = 0
    result_hais = []
    assert len(tehai) == 13, tehai
    assert len(_total_hais) == 34, _total_hais
    st = Shanten()
    current_st = st.calculate_shanten(_naga_tehai_to_tiles(tehai, nakis))
    if current_st == 0:
        # 已经听牌的进张枚数是0
        return 0, []
    count_dict = defaultdict(int)
    for t in tehai:
        count_dict[t] += 1
    for x in _total_hais:
        if count_dict[x] == 4:
            continue
        # 枚举可能摸进的所有牌，看是否能推进向听
        new_tehai = tehai.copy()
        new_tehai.append(x)
        new_st = st.calculate_shanten(_naga_tehai_to_tiles(new_tehai, nakis))
        if new_st < current_st:
            result += 4 - visible_maisuu.get(x, 0)
            result_hais.append(x)
    return result, result_hais


def _to_normal_hai(s: str) -> str:
    # 转化成标准的1-9mps1-7z格式，忽略赤宝
    return _naga_replace_d.get(s, s).removesuffix('r')


@functools.lru_cache(maxsize=100, typed=False)
def parse_report(text: str) -> dict:
    soup = BeautifulSoup(text, 'html.parser')
    # read variables
    script_tags = soup.find_all('script')
    variables_dict: dict[..., ...] = {}
    stc = Shanten()

    for script in script_tags:
        # 使用正则表达式找到所有的变量赋值
        matches = re.finditer(r'const\s+(\w+)\s*=\s*(.*?)\n', script.string if script.string else '')
        for match in list(matches):
            var_name = match.group(1)
            var_value = match.group(2)
            try:
                var_value = json.loads(var_value.replace("'", '"'))
            except json.decoder.JSONDecodeError:
                try:
                    var_value = demjson.decode(var_value.replace("'", '"'))
                except demjson.JSONDecodeError:
                    pass
            variables_dict[var_name] = var_value

    decision_count = defaultdict(int)
    decision_same = defaultdict(lambda: defaultdict(int))
    bad_moves = defaultdict(lambda: defaultdict(int))
    bad_moves_info = defaultdict(lambda: defaultdict(list))
    naga_rate = defaultdict(lambda: defaultdict(float))
    decision_danger = defaultdict(float)
    rank_uplift = defaultdict(float)
    naga_consensus = defaultdict(int)
    shantens = defaultdict(int)
    shanten_start = defaultdict(dict)
    shantens_diff = defaultdict(int)

    try:
        player_names = variables_dict['playerInfo']['name']
    except (ValueError, TypeError):
        player_names = ['P1', 'P2', 'P3', 'P4']

    naga_types: dict | None = variables_dict.get('nagaTypes', None)
    if naga_types is None:
        raise NotImplementedError('Does not support oldest naga reports')
    naga_types = {int(k): v for k, v in naga_types.items()}
    one_naga_idx = list(naga_types.keys())[0]  # 当不需要指定nagaType的时候，随便用一个

    # 恶手解析
    # 当有人立直时不打危险牌
    # 牌效率（进张枚数）
    # etc.

    for game_idx, game in enumerate(variables_dict['pred']):
        last_actor_idx = None
        last_expect_rank = None
        tehais: list[list[str]] = []  # 四家手牌，需要动态维护
        nakis: list[list[str]] = []  # 四家副露牌，需要动态维护
        scores: list[float] = []  # 四家分数，需要动态维护
        riichi_players = []  # 立直的玩家
        game_name = f"{game[0]['info']['msg']['bakaze']}{game[0]['info']['msg']['kyoku']}"
        if (honba := game[0]['info']['msg']['honba']) > 0:
            game_name += f'.{honba}'
        is_all_last = game_name.startswith('S4')
        actor_rounds = defaultdict(int)
        # 四家可见的枚数，第一层key为actor_idx，第二层key为replace后的牌名([1-9][mps]/[1-7]z)。如下几种情况会增加：宝牌指示牌（TODO：新杠宝指示牌）、他牌家河、自家自摸、他家副露、他家杠
        visible_maisuu = defaultdict(lambda: defaultdict(int))
        dora_marker = game[0]['info']['msg']['dora_marker']
        dora_marker = _to_normal_hai(dora_marker)

        for turn_idx, turn in enumerate(game):
            if 'tehais' in turn['info']['msg']['p_msg']:
                # 首巡初始化手牌与副露牌
                tehais = turn['info']['msg']['p_msg']['tehais']
                for i in range(4):
                    tehais[i] = [_to_normal_hai(s) for s in tehais[i]]
                    nakis.append([])

                for a_idx, th in enumerate(tehais):
                    for t in th:
                        visible_maisuu[a_idx][_to_normal_hai(t)] += 1  # 记录起始枚数
                    visible_maisuu[a_idx][dora_marker] += 1  # 记录宝牌指示牌
                scores = turn['info']['msg']['p_msg']['scores']
            # calc expect rank
            expect_ranks = {}
            for i, rks in enumerate(turn['game_rank']):
                expect_ranks[i] = (1 * rks[0] + 2 * rks[1] + 3 * rks[2] + 4 * rks[3]) / 1e4

            if last_actor_idx is not None:
                assert last_expect_rank is not None
                rank_uplift[player_names[last_actor_idx]] += last_expect_rank - expect_ranks[last_actor_idx]
                last_actor_idx = last_expect_rank = None

            # 记录立直玩家
            if turn['info']['msg']['type'] == 'reach_accepted':
                riichi_players.append(turn['info']['msg']['actor'])

            if 'dahai_pred' not in turn:
                continue

            # 更新actor_idx
            actor_idx = turn['info']['msg']['actor']
            actor_name = player_names[actor_idx]
            actor_rounds[actor_idx] += 1
            if sum(turn['dahai_pred'][one_naga_idx]) == 0:
                # 没有pred不考虑（一般是因为立直了）
                continue

            real_dahai: str | None = turn['info']['msg'].get('real_dahai')
            if not real_dahai:
                # 胡牌
                continue
            if real_dahai == '?':
                # 不知道打了啥，一般是因为杠
                count_dict = defaultdict(int)
                new_pai = _to_normal_hai(turn['info']['msg']['pai'])

                tehais[actor_idx].append(new_pai)
                for t in tehais[actor_idx]:
                    count_dict[_to_normal_hai(t)] += 1
                # 从下一turn的数据中找杠了哪个
                next_turn = game[turn_idx + 1]
                next_turn_t = next_turn['info']['msg']['type']
                if next_turn_t == 'kakan':
                    k = _to_normal_hai(next_turn['info']['msg']['pai'])
                elif next_turn_t == 'ankan':
                    k = _to_normal_hai(next_turn['info']['msg']['consumed'][0])
                else:
                    raise ValueError(f'Expected ankan/kakan after real_dahai == ?. Got {next_turn_t}')

                for a_idx in range(4):
                    visible_maisuu[a_idx][k] = 4

                if nakis[actor_idx].count(k) < 3:
                    # 暗杠，加入副露牌；明杠不加
                    nakis[actor_idx].extend([k, k, k])
                tehais[actor_idx].remove(k)
                continue

            real_dahai = _to_normal_hai(real_dahai)  # TODO：目前对赤的处理缺失
            real = int(_naga_B[real_dahai])
            last_actor_idx = actor_idx
            last_expect_rank = expect_ranks[last_actor_idx]
            for a_idx in range(4):
                if a_idx == actor_idx:
                    continue
                visible_maisuu[a_idx][_to_normal_hai(real_dahai)] += 1

            # 顺位
            score_rank = np.argsort(scores)[::-1][actor_idx] + 1

            naga_player_pred = []
            decision_count[actor_name] += 1

            # 摸牌前的向听
            shanten_before = stc.calculate_shanten(_naga_tehai_to_tiles(tehais[actor_idx], nakis[actor_idx]))

            # 摸到的牌加入手牌计算向听
            if turn['info']['msg']['type'] != 'tsumo':
                for h in turn['info']['msg']['consumed']:
                    nakis[actor_idx].append(_to_normal_hai(h))
                nakis[actor_idx].append(_to_normal_hai(turn['info']['msg']['pai']))
                for a_idx in range(4):
                    if a_idx == actor_idx:
                        continue
                    for h in turn['info']['msg']['consumed']:
                        visible_maisuu[a_idx][_to_normal_hai(h)] += 1

            new_pai = turn['info']['msg']['pai']
            new_pai = _to_normal_hai(new_pai)
            visible_maisuu[actor_idx][new_pai] += 1
            tehais[actor_idx].append(new_pai)
            tehais_14 = tehais[actor_idx].copy()
            shanten_after_draw = stc.calculate_shanten(_naga_tehai_to_tiles(tehais[actor_idx], nakis[actor_idx]))
            # 切牌
            try:
                tehais[actor_idx].remove(real_dahai)
            except (Exception,):
                raise ValueError(
                    f'未找到切牌: {game_name}-{actor_rounds[actor_idx]}:{actor_name} {tehais[actor_idx]} - {real_dahai}')
            shanten_after_discard = stc.calculate_shanten(_naga_tehai_to_tiles(tehais[actor_idx], nakis[actor_idx]))

            shantens[actor_name] += shanten_after_draw
            if game_idx not in shanten_start[actor_name]:
                shanten_start[actor_name][game_idx] = shanten_after_draw
            shantens_diff[actor_name] += max(0, shanten_before - shanten_after_draw)

            # 所有34张牌的危险度
            total_danger = np.sum(
                [turn['danger_k'][actor_idx], turn['danger_t'][actor_idx], turn['danger_s'][actor_idx]], axis=0)
            my_danger = total_danger[real] - total_danger.min()
            decision_danger[actor_name] += my_danger
            # 手里14张牌的危险度
            my_hand_danger = total_danger[[int(_naga_B[x]) for x in tehais_14]]
            # 切出手牌的危险度
            discard_danger_order = len([x for x in my_hand_danger if x >= my_hand_danger[tehais_14.index(real_dahai)]])

            for naga_idx, naga_name in naga_types.items():
                sum_prob = sum(turn['dahai_pred'][naga_idx])
                if sum_prob == 0:
                    continue
                norm_pred = np.array(turn['dahai_pred'][naga_idx]) / sum_prob
                pred = np.argmax(norm_pred)
                naga_player_pred.append(pred)
                is_bad_move = norm_pred[real] < .05
                naga_rate[naga_name][actor_name] += abs(norm_pred[real] - max(norm_pred))
                decision_same[naga_name][actor_name] += int(pred == real)
                if is_bad_move:
                    bad_moves[naga_name][actor_name] += 1
                    # 可能的恶手原因
                    possible_reasons = []
                    # 场况条件
                    situation_tags = []
                    if riichi_players:
                        situation_tags.append(f'{len(riichi_players)}家立直')
                    if is_all_last:
                        situation_tags.append('AL')
                    situation_tags.append(f'{score_rank}位')
                    if shanten_after_draw == 0:
                        situation_tags.append(f'听牌')
                    else:
                        situation_tags.append(f'{shanten_after_draw}向听')
                    # TODO 幺九与役牌选择
                    real_maisuu, _ = _calculate_maisuu(tehais[actor_idx], visible_maisuu[actor_idx], nakis[actor_idx])
                    tehais_after_pred = tehais_14.copy()
                    for t in tehais_after_pred:
                        if int(_naga_B[t]) == pred:
                            pred_name = t
                            tehais_after_pred.remove(t)
                            break
                    else:
                        raise ValueError(f'Cannot find discard target {pred} from {tehais_after_pred}')
                    pred_maisuu, _ = _calculate_maisuu(tehais_after_pred, visible_maisuu[actor_idx], nakis[actor_idx])

                    # 进攻系列
                    pred_danger_order = len(
                        [x for x in my_hand_danger if x >= my_hand_danger[tehais_14.index(pred_name)]])
                    if shanten_after_discard > shanten_after_draw:
                        possible_reasons.append(['牌效率', '退向听'])
                    elif real_maisuu < pred_maisuu:
                        possible_reasons.append(['牌效率', '进张枚数'])
                    elif pred_danger_order < discard_danger_order:
                        possible_reasons.append(['防守', '危牌先走'])
                    elif actor_rounds[actor_idx] <= 5:
                        possible_reasons.append(['牌效率', '早巡浮牌手顺'])

                    if riichi_players and shanten_after_draw + len(
                            riichi_players) >= 3 and pred_danger_order > discard_danger_order:
                        possible_reasons.append(['防守', '面对立直家按安全度全弃'])

                    bad_moves_info[naga_name][actor_name].append({
                        'reasons': possible_reasons,
                        'situation': situation_tags,
                        'game': game_name,
                        'turn': actor_rounds[actor_idx],
                        'pred': _naga_replace_d.get(pred_name, pred_name),
                        'real': _naga_replace_d.get(real_dahai, real_dahai)
                    })

            # NAGA不同模型的一致性
            naga_consensus[actor_name] += int(len(set(naga_player_pred)) == 1)

    ret = {}
    for k in decision_count.keys():
        for naga_name in naga_types.values():
            if k not in ret:
                ret[k] = {
                    'count': decision_count[k],
                    'danger': round(decision_danger[k] / decision_count[k] / 10000, 3),
                    'rank_uplift': round(rank_uplift[k], 3),
                    'consensus': round(naga_consensus[k] / decision_count[k], 3),
                    'shanten_avg': round(shantens[k] / decision_count[k], 3),
                    'shanten_uplift': round(shantens_diff[k] / decision_count[k], 3),
                    'shanten_start': round(sum(shanten_start[k].values()) / max(1, len(shanten_start[k])), 3),
                }
            ret[k][naga_name] = {
                'accuracy': round(decision_same[naga_name][k] / decision_count[k], 3),
                'rate': round(
                    (decision_count[k] - naga_rate[naga_name][k]) / decision_count[k] * 100, 3),
                'bad_rate': round(bad_moves[naga_name][k] / decision_count[k], 3),
                'bad_info': bad_moves_info[naga_name][k],
            }
            # 特别的，如果accuracy为0，则rate也为0，这是因为Mortal只绑定了一个视角
            if ret[k][naga_name]['accuracy'] == 0:
                ret[k][naga_name]['rate'] = 0
            # check for rate NaN
            if ret[k][naga_name]['rate'] != ret[k][naga_name]['rate']:
                ret[k][naga_name]['rate'] = 0
    return ret


if __name__ == '__main__':
    naga_url = sys.argv[1]
    if 'naga.dmv.nico' not in naga_url:
        naga_url = f'https://naga.dmv.nico/htmls/{naga_url}.html'
    if len(sys.argv) >= 3:
        mtk = sys.argv[2]
    else:
        mtk = None

    sess = requests.session()
    with sess.get(naga_url) as r:
        r.encoding = 'utf-8'

    content = r.text
    if mtk:
        # 需要合并Mortal进来
        mortal_url = f'https://mjai.ekyu.moe/report/{mtk}.json'
        content = merge_mortal_to_naga(content, requests.get(mortal_url).text)
    print(parse_report(content))
