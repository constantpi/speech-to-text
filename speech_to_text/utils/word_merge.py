THRESHOLD = 0.2  # クラスタリングの閾値（秒）


class Class:
    def __init__(self, start: float, end: float, text: str, index: int):
        self.word_list: list[tuple[str, int]] = [(text, index)]  # text, index
        self.range_list: list[tuple[float, float]] = [(start, end)]  # start, end
        self.middle = (start + end) / 2

    def add(self, start: float, end: float, text: str, index: int):
        self.word_list.append((text, index))
        self.range_list.append((start, end))
        self.middle = (self.middle * (len(self.word_list) - 1) + (start + end) / 2) / len(self.word_list)


def word_merge(word_list: list[list[tuple[float, float, str]]]) -> str:
    flatten_list: list[tuple[float, float, str, int]] = []  # start, end, text, index
    for i, sublist in enumerate(word_list):
        texts = "".join([text for _, _, text in sublist])
        print(f"Sublist {i}: {texts}")
        for start, end, text in sublist:
            flatten_list.append((start, end, text, i))
    # 区間の中央値でソートする
    flatten_list.sort(key=lambda x: (x[0] + x[1]) / 2)
    if not flatten_list:
        return ""
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
    result = []
    for cluster in clusters:
        num = len(cluster.word_list)
        if num == 1:
            result.append(cluster.word_list[0][0])
        elif num == 2:
            # このクラスタが全体から見て前半にあるか後半にあるかで、前半ならindexが小さい方、後半ならindexが大きい方を採用する
            if cluster.middle < max_end / 2:
                result.append(cluster.word_list[0][0])
            else:
                result.append(cluster.word_list[1][0])
        else:
            # clusterの中でindexが小さい順にtextを結合する
            sorted_word_list = sorted(cluster.word_list, key=lambda x: x[1])
            # 中央値を採用する
            result.append(sorted_word_list[num // 2][0])
    # 同じ単語が連続している場合は1つにまとめる
    final_result = []
    for word in result:
        if not final_result or final_result[-1] != word:
            final_result.append(word)

    print("".join(final_result))
    return "".join(final_result)
