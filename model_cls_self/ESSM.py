import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import Model


class CTCVRNet:
    def __init__(self, cate_feautre_dict):
        self.embed = dict()
        for k, v in cate_feautre_dict.items():
            self.embed[k] = layers.Embedding(v, 64)

    def build_ctr_model(self, ctr_user_numerical_input, ctr_user_cate_input, ctr_item_numerical_input,
                        ctr_item_cate_input, ctr_user_cate_feature_dict, ctr_item_cate_feature_dict):
        user_embeddings, item_embeddings = [], []
        for k, v in ctr_user_cate_feature_dict.items():
            embed = self.embed[k](tf.reshape(ctr_user_cate_input[:, v[0]], [-1, 1]))
            embed = layers.Reshape((64,))(embed)
            user_embeddings.append(embed)

        for k, v in ctr_item_cate_feature_dict.items():
            embed = self.embed[k](tf.reshape(ctr_item_cate_input[:, v[0]], [-1, 1]))
            embed = layers.Reshape((64,))(embed)
            item_embeddings.append(embed)
        user_feature = layers.concatenate([ctr_user_numerical_input] + user_embeddings, axis=-1)
        item_feature = layers.concatenate([ctr_item_numerical_input] + item_embeddings, axis=-1)

        user_feature = layers.Dropout(0.5)(user_feature)
        user_feature = layers.BatchNormalization()(user_feature)
        user_feature = layers.Dense(128, activation='relu')(user_feature)
        user_feature = layers.Dense(64, activation='relu')(user_feature)

        item_feature = layers.Dropout(0.5)(item_feature)
        item_feature = layers.BatchNormalization()(item_feature)
        item_feature = layers.Dense(128, activation='relu')(item_feature)
        item_feature = layers.Dense(64, activation='relu')(item_feature)

        dense_feature = layers.concatenate([user_feature, item_feature], axis=-1)
        dense_feature = layers.Dropout(0.5)(dense_feature)
        dense_feature = layers.BatchNormalization()(dense_feature)
        dense_feature = layers.Dense(64, activation='relu')(dense_feature)
        pred = layers.Dense(1, activation='sigmoid', name='ctr_output')(dense_feature)
        return pred

    def build_cvr_model(self, cvr_user_numerical_input, cvr_user_cate_input, cvr_item_numerical_input,
                        cvr_item_cate_input, cvr_user_cate_feature_dict, cvr_item_cate_feature_dict):
        user_embeddings, item_embeddings = [], []
        for k, v in cvr_user_cate_feature_dict.items():
            embed = self.embed[k](tf.reshape(cvr_user_cate_input[:, v[0]], [-1, 1]))
            embed = layers.Reshape((64,))(embed)
            user_embeddings.append(embed)

        for k, v in cvr_item_cate_feature_dict.items():
            embed = self.embed[k](tf.reshape(cvr_item_cate_input[:, v[0]], [-1, 1]))
            embed = layers.Reshape((64,))(embed)
            item_embeddings.append(embed)
        user_feature = layers.concatenate([cvr_user_numerical_input] + user_embeddings, axis=-1)
        item_feature = layers.concatenate([cvr_item_numerical_input] + item_embeddings, axis=-1)

        user_feature = layers.Dropout(0.5)(user_feature)
        user_feature = layers.BatchNormalization()(user_feature)
        user_feature = layers.Dense(128, activation='relu')(user_feature)
        user_feature = layers.Dense(64, activation='relu')(user_feature)

        item_feature = layers.Dropout(0.5)(item_feature)
        item_feature = layers.BatchNormalization()(item_feature)
        item_feature = layers.Dense(128, activation='relu')(item_feature)
        item_feature = layers.Dense(64, activation='relu')(item_feature)

        dense_feature = layers.concatenate([user_feature, item_feature], axis=-1)
        dense_feature = layers.Dropout(0.5)(dense_feature)
        dense_feature = layers.BatchNormalization()(dense_feature)
        dense_feature = layers.Dense(64, activation='relu')(dense_feature)
        pred = layers.Dense(1, activation='sigmoid', name='cvr_output')(dense_feature)
        return pred

    def build(self, user_cate_feature_dict, item_cate_feature_dict):
        # CTR model input
        ctr_user_numerical_input = layers.Input(shape=(5,))
        ctr_user_cate_input = layers.Input(shape=(5,))
        ctr_item_numerical_input = layers.Input(shape=(5,))
        ctr_item_cate_input = layers.Input(shape=(3,))

        # CVR model input
        cvr_user_numerical_input = layers.Input(shape=(5,))
        cvr_user_cate_input = layers.Input(shape=(5,))
        cvr_item_numerical_input = layers.Input(shape=(5,))
        cvr_item_cate_input = layers.Input(shape=(3,))

        ctr_pred = self.build_ctr_model(ctr_user_numerical_input, ctr_user_cate_input, ctr_item_numerical_input,
                                        ctr_item_cate_input, user_cate_feature_dict, item_cate_feature_dict)
        cvr_pred = self.build_cvr_model(cvr_user_numerical_input, cvr_user_cate_input, cvr_item_numerical_input,
                                        cvr_item_cate_input, user_cate_feature_dict, item_cate_feature_dict)
        ctcvr_pred = tf.multiply(ctr_pred, cvr_pred)
        model = Model(
            inputs=[ctr_user_numerical_input, ctr_user_cate_input, ctr_item_numerical_input, ctr_item_cate_input,
                    cvr_user_numerical_input, cvr_user_cate_input, cvr_item_numerical_input, cvr_item_cate_input],
            outputs=[ctr_pred, ctcvr_pred])

        return model


def train_model(cate_feature_dict, user_cate_feature_dict, item_cate_feature_dict, train_data, val_data):
    """
    model train and save as tf serving model
    :param cate_feature_dict: dict, categorical feature for data
    :param user_cate_feature_dict: dict, user categorical feature
    :param item_cate_feature_dict: dict, item categorical feature
    :param train_data: DataFrame, training data
    :param val_data: DataFrame, valdation data
    :return: None
    """
    ctcvr = CTCVRNet(cate_feature_dict)
    ctcvr_model = ctcvr.build(user_cate_feature_dict, item_cate_feature_dict)
    opt = optimizers.Adam(lr=0.003, decay=0.0001)
    ctcvr_model.compile(optimizer=opt, loss=["binary_crossentropy", "binary_crossentropy"], loss_weights=[1.0, 1.0],
                        metrics=[tf.keras.metrics.AUC()])

    # keras model save path
    filepath = "esmm_best.h5"

    # call back function
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.8, patience=2, min_lr=0.0001, verbose=1)
    earlystopping = EarlyStopping(
        monitor='val_loss', min_delta=0.0001, patience=8, verbose=1, mode='auto')
    callbacks = [checkpoint, reduce_lr, earlystopping]

    # load data
    ctr_user_numerical_feature_train, ctr_user_cate_feature_train, ctr_item_numerical_feature_train, \
        ctr_item_cate_feature_train, cvr_user_numerical_feature_train, cvr_user_cate_feature_train, \
        cvr_item_numerical_feature_train, cvr_item_cate_feature_train, ctr_target_train, cvr_target_train = train_data

    ctr_user_numerical_feature_val, ctr_user_cate_feature_val, ctr_item_numerical_feature_val, \
        ctr_item_cate_feature_val, cvr_user_numerical_feature_val, cvr_user_cate_feature_val, \
        cvr_item_numerical_feature_val, cvr_item_cate_feature_val, ctr_target_val, cvr_target_val = val_data

    # model train
    ctcvr_model.fit([ctr_user_numerical_feature_train, ctr_user_cate_feature_train, ctr_item_numerical_feature_train,
                     ctr_item_cate_feature_train, cvr_user_numerical_feature_train, cvr_user_cate_feature_train,
                     cvr_item_numerical_feature_train,
                     cvr_item_cate_feature_train], [ctr_target_train, cvr_target_train], batch_size=256, epochs=50,
                    validation_data=(
                        [ctr_user_numerical_feature_val, ctr_user_cate_feature_val, ctr_item_numerical_feature_val,
                         ctr_item_cate_feature_val, cvr_user_numerical_feature_val, cvr_user_cate_feature_val,
                         cvr_item_numerical_feature_val,
                         cvr_item_cate_feature_val], [ctr_target_val, cvr_target_val]), callbacks=callbacks,
                    verbose=0,
                    shuffle=True)

    # load model and save as tf_serving model
    saved_model_path = './esmm/{}'.format(int(time.time()))
    ctcvr_model = tf.keras.models.load_model('esmm_best.h5')
    tf.saved_model.save(ctcvr_model, saved_model_path)


if __name__ == '__main__':
    # data = pd.read_csv('../joy_data/test_data.csv')
    #
    # short_cat_cols = ['gender', 'language', 'channels', 'author_language']
    # long_cat_cols = ['country_code', 'author_country_code']
    # time_cat_cols = ['hour','dow','till_push','attachments_count']
    # id_cols = ['uid', 'author_id', 'post_id']
    # num_cols = ['friends_num', 'binding_toys_number', 'hits_rate', 'like_rate', 'collect_rate', 'comments_rate', 'score']
    # ctr_tag = ['is_like']

    ctr_user_numerical_feature_train = pd.DataFrame(np.random.random((10000, 5)),columns=['user_numerical_{}'.format(i) for i in range(5)])
    ctr_user_cate_feature_train = pd.DataFrame(np.random.randint(0, 10, size=(10000, 5)),columns=['user_cate_{}'.format(i) for i in range(5)])
    ctr_item_numerical_feature_train = pd.DataFrame(np.random.random((10000, 5)),columns=['item_numerical_{}'.format(i) for i in range(5)])
    ctr_item_cate_feature_train = pd.DataFrame(np.random.randint(0, 10, size=(10000, 3)),columns=['item_cate_{}'.format(i) for i in range(3)])

    cvr_user_numerical_feature_train = pd.DataFrame(np.random.random((10000, 5)),columns=['user_numerical_{}'.format(i) for i in range(5)])
    cvr_user_cate_feature_train = pd.DataFrame(np.random.randint(0, 10, size=(10000, 5)),columns=['user_cate_{}'.format(i) for i in range(5)])
    cvr_item_numerical_feature_train = pd.DataFrame(np.random.random((10000, 5)),columns=['item_numerical_{}'.format(i) for i in range(5)])
    cvr_item_cate_feature_train = pd.DataFrame(np.random.randint(0, 10, size=(10000, 3)),columns=['item_cate_{}'.format(i) for i in range(3)])

    ctr_user_numerical_feature_val = pd.DataFrame(np.random.random((10000, 5)),columns=['user_numerical_{}'.format(i) for i in range(5)])
    ctr_user_cate_feature_val = pd.DataFrame(np.random.randint(0, 10, size=(10000, 5)),columns=['user_cate_{}'.format(i) for i in range(5)])
    ctr_item_numerical_feature_val = pd.DataFrame(np.random.random((10000, 5)),columns=['item_numerical_{}'.format(i) for i in range(5)])
    ctr_item_cate_feature_val = pd.DataFrame(np.random.randint(0, 10, size=(10000, 3)),columns=['item_cate_{}'.format(i) for i in range(3)])

    cvr_user_numerical_feature_val = pd.DataFrame(np.random.random((10000, 5)),columns=['user_numerical_{}'.format(i) for i in range(5)])
    cvr_user_cate_feature_val = pd.DataFrame(np.random.randint(0, 10, size=(10000, 5)),columns=['user_cate_{}'.format(i) for i in range(5)])
    cvr_item_numerical_feature_val = pd.DataFrame(np.random.random((10000, 5)),columns=['item_numerical_{}'.format(i) for i in range(5)])
    cvr_item_cate_feature_val = pd.DataFrame(np.random.randint(0, 10, size=(10000, 3)),columns=['item_cate_{}'.format(i) for i in range(3)])

    ctr_target_train = pd.DataFrame(np.random.randint(0, 2, size=10000))
    cvr_target_train = pd.DataFrame(np.random.randint(0, 2, size=10000))

    ctr_target_val = pd.DataFrame(np.random.randint(0, 2, size=10000))
    cvr_target_val = pd.DataFrame(np.random.randint(0, 2, size=10000))

    train_data = [ctr_user_numerical_feature_train, ctr_user_cate_feature_train, ctr_item_numerical_feature_train,
                  ctr_item_cate_feature_train, cvr_user_numerical_feature_train, cvr_user_cate_feature_train,
                  cvr_item_numerical_feature_train, cvr_item_cate_feature_train, ctr_target_train, cvr_target_train]

    val_data = [ctr_user_numerical_feature_val, ctr_user_cate_feature_val, ctr_item_numerical_feature_val,
                ctr_item_cate_feature_val, cvr_user_numerical_feature_val, cvr_user_cate_feature_val,
                cvr_item_numerical_feature_val, cvr_item_cate_feature_val, ctr_target_val, cvr_target_val]

    cate_feature_dict = {}
    user_cate_feature_dict = {}
    item_cate_feature_dict = {}
    for idx, col in enumerate(ctr_user_cate_feature_train.columns):
        cate_feature_dict[col] = ctr_user_cate_feature_train[col].max() + 1
        user_cate_feature_dict[col] = (idx, ctr_user_cate_feature_train[col].max() + 1)
    for idx, col in enumerate(ctr_item_cate_feature_train.columns):
        cate_feature_dict[col] = ctr_item_cate_feature_train[col].max() + 1
        item_cate_feature_dict[col] = (idx, ctr_item_cate_feature_train[col].max() + 1)

    ctcvr = CTCVRNet(cate_feature_dict)
    ctcvr_model = ctcvr.build(user_cate_feature_dict, item_cate_feature_dict)
    opt = optimizers.Adam(lr=0.003, decay=0.0001)
    ctcvr_model.compile(optimizer=opt, loss=["binary_crossentropy", "binary_crossentropy"], loss_weights=[1.0, 1.0],
                        metrics=[tf.keras.metrics.AUC()])

    # keras model save path
    filepath = "esmm_best.h5"

    # call back function
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.8, patience=2, min_lr=0.0001, verbose=1)
    earlystopping = EarlyStopping(
        monitor='val_loss', min_delta=0.0001, patience=8, verbose=1, mode='auto')
    callbacks = [checkpoint, reduce_lr, earlystopping]

    # train model
    train_model(cate_feature_dict, user_cate_feature_dict, item_cate_feature_dict, train_data, val_data)
