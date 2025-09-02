def negSample_like_word2vec(train_data, all_items, all_users, neg_num=10):
    """
    为所有item计算一个采样概率，根据概率为每个用户采样neg_num个负样本，返回所有负样本对
    1. 统计所有item在交互中的出现频次
    2. 根据频次进行排序，并计算item采样概率（频次出现越多，采样概率越低，打压热门item）
    3. 根据采样概率，利用多线程为每个用户采样 neg_num 个负样本
    """
    pos_samples = train_data[train_data["click"] == 1][["user_id", "article_id"]]

    pos_samples_dic = {}
    for idx, u in enumerate(pos_samples["user_id"].unique().tolist()):
        pos_list = list(pos_samples[pos_samples["user_id"] == u]["article_id"].unique().tolist())
        if len(pos_list) >= 30:  # 30是拍的  需要数据统计的支持确定
            pos_samples_dic[u] = pos_list[30:]
        else:
            pos_samples_dic[u] = pos_list

    # 统计出现频次
    article_counts = train_data["article_id"].value_counts()
    df_article_counts = pd.DataFrame(article_counts)
    dic_article_counts = dict(zip(df_article_counts.index.values.tolist(), df_article_counts.article_id.tolist()))

    for item in all_items:
        if item[0] not in dic_article_counts.keys():
            dic_article_counts[item[0]] = 0

    # 根据频次排序, 并计算每个item的采样概率
    tmp = sorted(list(dic_article_counts.items()), key=lambda x: x[1], reverse=True)  # 降序
    n_articles = len(tmp)
    article_prob = {}
    for idx, item in enumerate(tmp):
        article_prob[item[0]] = cal_pos(idx, n_articles)

    # 为每个用户进行负采样
    article_id_list = [a[0] for a in article_prob.items()]
    article_pro_list = [a[1] for a in article_prob.items()]
    pos_sample_users = list(pos_samples_dic.keys())

    all_users_list = [u[0] for u in all_users]

    print("start negative sampling !!!!!!")
    pool = multiprocessing.Pool(core_size)
    res = pool.map(SampleOneProb((pos_sample_users, article_id_list, article_pro_list, pos_samples_dic, neg_num)),
                   tqdm(all_users_list))
    pool.close()
    pool.join()

    neg_sample_dic = {}
    for idx, u in tqdm(enumerate(all_users_list)):
        neg_sample_dic[u] = res[idx]

    return [[k, i, 0] for k, v in neg_sample_dic.items() for i in v]


def dealsample(file, doc_data, user_data, s_data_str="2021-06-24 00:00:00", e_data_str="2021-06-30 23:59:59",
               neg_num=5):
    # 先处理时间问题
    data = pd.read_csv(file_path + file, sep="\t", index_col=0)
    data['expo_time'] = data['expo_time'].astype('str')
    data['expo_time'] = data['expo_time'].apply(lambda x: int(x[:10]))
    data['expo_time'] = pd.to_datetime(data['expo_time'], unit='s', errors='coerce')

    s_date = datetime.datetime.strptime(s_data_str, "%Y-%m-%d %H:%M:%S")
    e_date = datetime.datetime.strptime(e_data_str, "%Y-%m-%d %H:%M:%S") + datetime.timedelta(days=-1)
    t_date = datetime.datetime.strptime(e_data_str, "%Y-%m-%d %H:%M:%S")

    # 选取训练和测试所需的数据
    all_data_tmp = data[(data["expo_time"] >= s_date) & (data["expo_time"] <= t_date)]

    # 处理训练数据集  防止穿越样本
    # 1. merge 新闻信息，得到曝光时间和新闻创建时间； inner join 去除doc_data之外的新闻
    all_data_tmp = all_data_tmp.join(doc_data.set_index("article_id"), on="article_id", how='inner')

    # 发现还存在 ctime大于expo_time的交互存在  去除这部分错误数据
    all_data_tmp = all_data_tmp[(all_data_tmp["ctime"] <= all_data_tmp["expo_time"])]

    # 2. 去除与新闻的创建时间在测试数据时间内的交互  ()
    train_data = all_data_tmp[(all_data_tmp["expo_time"] >= s_date) & (all_data_tmp["expo_time"] <= e_date)]
    train_data = train_data[(train_data["ctime"] <= e_date)]

    print("有效的样本数：", train_data["expo_time"].count())

    # 负采样
    if os.path.exists(file_path + "neg_sample.pkl") and os.path.getsize(file_path + "neg_sample.pkl"):
        neg_samples = pd.read_pickle(file_path + "neg_sample.pkl")
        # train_neg_samples.insert(loc=2, column="click", value=[0] * train_neg_samples["user_id"].count())
    else:
        # 进行负采样的时候对于样本进行限制，只对一定时间范围之内的样本进行负采样
        doc_data_tmp = doc_data[
            (doc_data["ctime"] >= datetime.datetime.strptime("2021-06-01 00:00:00", "%Y-%m-%d %H:%M:%S"))]
        neg_samples = negSample_like_word2vec(train_data, doc_data_tmp[["article_id"]].values,
                                              user_data[["user_id"]].values, neg_num=neg_num)
        neg_samples = pd.DataFrame(neg_samples, columns=["user_id", "article_id", "click"])
        neg_samples.to_pickle(file_path + "neg_sample.pkl")

    train_pos_samples = train_data[train_data["click"] == 1][["user_id", "article_id", "expo_time", "click"]]  # 取正样本

    neg_samples_df = train_data[train_data["click"] == 0][["user_id", "article_id", "click"]]
    train_neg_samples = pd.concat([neg_samples_df.sample(n=train_pos_samples["click"].count()), neg_samples],
                                  axis=0)  # 取负样本

    print("训练集正样本数：", train_pos_samples["click"].count())
    print("训练集负样本数：", train_neg_samples["click"].count())

    train_data_df = pd.concat([train_neg_samples, train_pos_samples], axis=0)
    train_data_df = train_data_df.sample(frac=1)  # shuffle

    print("训练集总样本数：", train_data_df["click"].count())

    test_data_df = all_data_tmp[(all_data_tmp["expo_time"] > e_date) & (all_data_tmp["expo_time"] <= t_date)][
        ["user_id", "article_id", "expo_time", "click"]]

    print("测试集总样本数：", test_data_df["click"].count())
    print("测试集总样本数：", test_data_df["click"].count())

    all_data_df = pd.concat([train_data_df, test_data_df], axis=0)

    print("总样本数：", all_data_df["click"].count())

    return all_data_df
