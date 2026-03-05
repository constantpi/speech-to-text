THRESHOLD = 0.2  # クラスタリングの閾値（秒）


class MergeResult:
    def __init__(self, determined_text: str, determined_end: float, undetermined_text: str):
        self.determined_text = determined_text
        self.determined_end = determined_end
        self.undetermined_text = undetermined_text


def extract_alphabet(word: str) -> str:
    '''単語からアルファベットのみを抽出する'''
    return "".join([c for c in word if c.isalpha()])


def is_same_word(word1: str, word2: str) -> bool:
    '''2つの単語が同じかどうかを判定する'''
    return extract_alphabet(word1).lower() == extract_alphabet(word2).lower()


class Class:
    def __init__(self, start: float, end: float, text: str, index: int):
        self.word_list: list[tuple[str, int]] = [(text, index)]  # text, index
        self.range_list: list[tuple[float, float]] = [(start, end)]  # start, end
        self.middle = (start + end) / 2

    def add(self, start: float, end: float, text: str, index: int):
        self.word_list.append((text, index))
        self.range_list.append((start, end))
        self.middle = (self.middle * (len(self.word_list) - 1) + (start + end) / 2) / len(self.word_list)


def word_merge(word_list: list[list[tuple[float, float, str]]], audio_start: float) -> MergeResult:

    # 調整のためにaudio_startを1秒前倒しする。そうしないと本当にピリオドかどうかの判定が難しいため
    audio_start -= 1.0

    # 事前におかしな検出を削除しておく。
    word_list = clean_word_list(word_list, 0.0)

    flatten_list: list[tuple[float, float, str, int]] = []  # start, end, text, index
    for i, sublist in enumerate(word_list):
        texts = "".join([text for _, _, text in sublist])
        print(f"sublist {i}: {texts}")
        # もしsublistのすべてにおいてstart==endであったら、sublistをスキップする
        for start, end, text in sublist:
            # もしtextが単語+...のようになっている場合は、textからアルファベットのみを抽出する
            if text.endswith("..."):
                text = extract_alphabet(text[:-3])
            elif text.startswith("..."):
                text = extract_alphabet(text[3:])
            flatten_list.append((start, end, text, i))
    # 区間の中央値でソートする
    flatten_list.sort(key=lambda x: (x[0] + x[1]) / 2)
    if not flatten_list:
        return MergeResult(determined_text="", determined_end=0.0, undetermined_text="")
    max_end = max(flatten_list, key=lambda x: x[1])[1]
    # クラスタリングする
    clusters: list[Class] = []
    for start, end, text, index in flatten_list:
        # clustersの中で、startとendの中央値がstartとendの中央値からTHRESHOLD以内のものを探す
        middle = (start + end) / 2
        candidates_index = []
        for i, cluster in enumerate(clusters):
            if abs(cluster.middle - middle) <= THRESHOLD:
                # いずれかの区間と重なっているかを確認する
                overlap = False
                for cluster_start, cluster_end in cluster.range_list:
                    if not (end <= cluster_start or start >= cluster_end):
                        overlap = True
                        break
                # 同じindexのものが入っていたら不可
                if any(word_index == index for _, word_index in cluster.word_list):
                    overlap = False
                if overlap:
                    candidates_index.append(i)
        # その中でmiddleに最も近いものを選ぶ
        if candidates_index:
            closest_index = min(candidates_index, key=lambda i: abs(clusters[i].middle - middle))
            clusters[closest_index].add(start, end, text, index)
        else:
            clusters.append(Class(start, end, text, index))
    # clustersをmiddleでソートする
    clusters.sort(key=lambda x: x.middle)
    result: list[tuple[str, float]] = []
    for cluster in clusters:
        max_end = max(cluster.range_list, key=lambda x: x[1])[1]
        num = len(cluster.word_list)
        if num == 1:
            result.append((cluster.word_list[0][0], max_end))
        elif num == 2:
            # このクラスタが全体から見て前半にあるか後半にあるかで、前半ならindexが小さい方、後半ならindexが大きい方を採用する
            if cluster.middle < max_end / 2:
                result.append((cluster.word_list[0][0], max_end))
            else:
                result.append((cluster.word_list[1][0], max_end))
        else:
            # clusterの中でindexが小さい順にtextを結合する
            sorted_word_list = sorted(cluster.word_list, key=lambda x: x[1])
            # 中央値を採用する
            result.append((sorted_word_list[num // 2][0], max_end))
    # 同じ単語が連続している場合は1つにまとめる
    final_result: list[tuple[str, float]] = []
    for r in result:
        if not final_result or not is_same_word(final_result[-1][0], r[0]):
            final_result.append(r)
        if is_same_word(final_result[-1][0], r[0]):
            final_result[-1] = (final_result[-1][0], max(final_result[-1][1], r[1]))
    is_determined = True
    determined_end = 0.0
    determined_length = 0
    for i, (word, end) in enumerate(final_result):
        if is_determined:
            is_determined = audio_start > end
        if is_determined:
            # 単語の終わりがピリオドであったら、そこで確定とする
            if word.endswith(".") or word.endswith("?") or word.endswith("!"):
                determined_end = end
                determined_length = i + 1

    print("determined_text:", "".join(word for word, _ in final_result[:determined_length]))
    print("undetermined_text:", "".join(word for word, _ in final_result[determined_length:]))

    return MergeResult(
        determined_text="".join(word for word, _ in final_result[:determined_length]),
        determined_end=determined_end,
        undetermined_text="".join(word for word, _ in final_result[determined_length:])
    )


def clean_word_list(word_list: list[list[tuple[float, float, str]]], determined_end: float) -> list[list[tuple[float, float, str]]]:
    '''word_listから、determined_endより前の単語を削除する'''
    new_word_list: list[list[tuple[float, float, str]]] = []
    for sublist in word_list:
        if not sublist:
            continue
        new_sublist = []
        start_time_list = [start for start, _, _ in sublist]
        end_time_list = [end for _, end, _ in sublist]
        if start_time_list == end_time_list:
            # すべての単語が0秒で検出されている場合は、誤検出の可能性が高いためスキップする
            print(f"skipping sublist because all start and end are the same: {sublist}")
            continue
        first_start = sublist[0][0]
        first_end = sublist[0][1]
        if len(sublist) > 1:
            if all(start == first_start for start, _, _ in sublist) or all(end == first_end for _, end, _ in sublist):
                # すべて同じ位置に検出されている単語は、誤検出の可能性が高いためスキップする
                print(f"skipping sublist because all start or end are the same: {sublist}")
                continue

        for start, end, text in sublist:
            if start >= determined_end - 0.01:  # determined_endの少し前から残す
                new_sublist.append((start, end, text))
        if new_sublist:
            new_word_list.append(new_sublist)
    return new_word_list
