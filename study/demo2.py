class ProductLayer(Layer):
    def __init__(self, units, use_inner=True, use_outer=False):
        super(ProductLayer, self).__init__()
        self.use_inner = use_inner
        self.use_outer = use_outer
        self.units = units  # 指的是原文中D1的大小

    def build(self, input_shape):
        # 需要注意input_shape也是一个列表，并且里面的每一个元素都是TensorShape类型，
        # 需要将其转换成list然后才能参与数值计算，不然类型容易错
        # input_shape[0] : feat_nums x embed_dims
        self.feat_nums = len(input_shape)
        self.embed_dims = input_shape[0].as_list()[-1]
        flatten_dims = self.feat_nums * self.embed_dims

        # Linear signals weight, 这部分是用于产生Z的权重，因为这里需要计算的是两个元素对应元素乘积然后再相加
        # 等价于先把矩阵拉成一维，然后相乘再相加
        self.linear_w = self.add_weight(name='linear_w', shape=(flatten_dims, self.units), initializer='glorot_normal')

        # inner product weight
        if self.use_inner:
            # 优化之后的内积权重是未优化时的一个分解矩阵，未优化时的矩阵大小为：D x N x N
            # 优化后的内积权重大小为：D x N
            self.inner_w = self.add_weight(name='inner_w', shape=(self.units, self.feat_nums),
                                           initializer='glorot_normal')

        if self.use_outer:
            # 优化之后的外积权重大小为：D x embed_dim x embed_dim, 因为计算外积的时候在特征维度通过求和的方式进行了压缩
            self.outer_w = self.add_weight(name='outer_w', shape=(self.units, self.embed_dims, self.embed_dims),
                                           initializer='glorot_normal')

    def call(self, inputs):
        # inputs是一个列表
        # 先将所有的embedding拼接起来计算线性信号部分的输出
        concat_embed = Concatenate(axis=1)(inputs)  # B x feat_nums x embed_dims
        # 将两个矩阵都拉成二维的，然后通过矩阵相乘得到最终的结果
        concat_embed_ = tf.reshape(concat_embed, shape=[-1, self.feat_nums * self.embed_dims])
        lz = tf.matmul(concat_embed_, self.linear_w)  # B x units

        # inner
        lp_list = []
        if self.use_inner:
            for i in range(self.units):
                # 相当于给每一个特征向量都乘以一个权重
                # self.inner_w[i] : (embed_dims, ) 添加一个维度变成 (embed_dims, 1)
                # concat_embed: B x feat_nums x embed_dims; delta = B x feat_nums x embed_dims
                delta = tf.multiply(concat_embed, tf.expand_dims(self.inner_w[i], axis=1))
                # 在特征之间的维度上求和
                delta = tf.reduce_sum(delta, axis=1)  # B x embed_dims
                # 最终在特征embedding维度上求二范数得到p
                lp_list.append(tf.reduce_sum(tf.square(delta), axis=1, keepdims=True))  # B x 1

        # outer
        if self.use_outer:
            # 外积的优化是将embedding矩阵，在特征间的维度上通过求和进行压缩
            feat_sum = tf.reduce_sum(concat_embed, axis=1)  # B x embed_dims

            # 为了方便计算外积，将维度进行扩展
            f1 = tf.expand_dims(feat_sum, axis=2)  # B x embed_dims x 1
            f2 = tf.expand_dims(feat_sum, axis=1)  # B x 1 x embed_dims

            # 求外积, a * a^T
            product = tf.matmul(f1, f2)  # B x embed_dims x embed_dims

            # 将product与外积权重矩阵对应元素相乘再相加
            for i in range(self.units):
                lpi = tf.multiply(product, self.outer_w[i])  # B x embed_dims x embed_dims
                # 将后面两个维度进行求和，需要注意的是，每使用一次reduce_sum就会减少一个维度
                lpi = tf.reduce_sum(lpi, axis=[1, 2])  # B
                # 添加一个维度便于特征拼接
                lpi = tf.expand_dims(lpi, axis=1)  # B x 1
                lp_list.append(lpi)

        # 将所有交叉特征拼接到一起
        lp = Concatenate(axis=1)(lp_list)

        # 将lz和lp拼接到一起
        product_out = Concatenate(axis=1)([lz, lp])

        return product_out
