import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import urllib.request
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_sc                                                                                                             ore, \
                            roc_auc_score, confusion_matrix, classification_repo                                                                                                             rt, \
                            matthews_corrcoef, cohen_kappa_score, log_loss


MODEL_NAME = "klue/bert-base"
model = TFBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3                                                                                                             , from_pt=True)
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

device_name = tf.test.gpu_device_name()
if device_name == '/device:GPU:0':
  print("GPU 작동 중")
  mirrored_strategy = tf.distribute.MirroredStrategy()
else:
  print("GPU 미작동 중")

DATASET_URL = "https://raw.githubusercontent.com/ukairia777/finance_sentiment_co                                                                                                             rpus/main/finance_data.csv"
DATASET_NAME = "finance_data.csv"


urllib.request.urlretrieve(DATASET_URL,
                           filename = DATASET_NAME
                           )
dataset = pd.read_csv(DATASET_NAME)
dataset.head()


del dataset['sentence']
dataset['labels'] = dataset['labels'].replace(['neutral', 'positive', 'negative'                                                                                                             ],[0, 1, 2])
dataset.head()
dataset.info()

# 중복 데이터 확인
dataset[dataset['kor_sentence'].duplicated()]

DATASET_PREP_FILE = '/home/ubuntu/Test_AlphaType/dataset_prep.csv'
# 중복 데이터 제거
dataset.drop_duplicates(subset = ['kor_sentence'], inplace = True)
dataset.to_csv(DATASET_PREP_FILE) # 구글 드라이브 내 data 폴더에 저장

LABEL_NUM_FILE = '/home/ubuntu/Test_AlphaType/label_number.png'
dataset['labels'].value_counts().plot(kind = 'bar')
plt.xlabel("Label")
plt.ylabel("Number")
plt.savefig(LABEL_NUM_FILE) # 구글 드라이브 내 figure 폴더에 저장

LABEL_RATIO_FILE = '/home/ubuntu/Test_AlphaType/label_ratio.png'
dataset['labels'].value_counts(normalize = True).plot(kind = 'bar', )
plt.xlabel("Label")
plt.ylabel("Ratio")
plt.savefig(LABEL_RATIO_FILE)

# 중립적인 기사문 59.27%, 긍정적인 기사문 28.22%, 부정적인 기사문 12.51%
dataset['labels'].value_counts(normalize = True)

# 입출력 데이터 분리
X_data = dataset['kor_sentence']
y_data = dataset['labels']



TEST_SIZE = 0.2 # Train: Test = 8 :2 분리
RANDOM_STATE = 42
# strtify = True 일 경우, 데이터 분리 이전의 라벨별 분포 고려
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
                                                    test_size = TEST_SIZE,
                                                    random_state = RANDOM_STATE,                                                                                                             
                                                    stratify = y_data)

print(f"훈련 입력 데이터 개수: {len(X_train)}")
print(f"테스트 입력 데이터 개수: {len(X_test)}")

# 훈련 데이터 라벨별 비율
y_train.value_counts(normalize = True)



# 테스트 데이터 라벨별 비율
y_test.value_counts(normalize = True)

# 입력 데이터(문장) 길이 제한
MAX_SEQ_LEN = 64

def convert_data(X_data, y_data):
    # BERT 입력으로 들어가는 token, mask, segment, target 저장용 리스트
    tokens, masks, segments, targets = [], [], [], []

    for X, y in tqdm(zip(X_data, y_data)):
        # token: 입력 문장 토큰화
        token = tokenizer.encode(X, truncation = True, padding = 'max_length', m                                                                                                             ax_length = MAX_SEQ_LEN)

        # Mask: 토큰화한 문장 내 패딩이 아닌 경우 1, 패딩인 경우 0으로 초기화
        num_zeros = token.count(0)
        mask = [1] * (MAX_SEQ_LEN - num_zeros) + [0] * num_zeros

        # segment: 문장 전후관계 구분: 오직 한 문장이므로 모두 0으로 초기화
        segment = [0]*MAX_SEQ_LEN

        tokens.append(token)
        masks.append(mask)
        segments.append(segment)
        targets.append(y)

    # numpy array로 저장
    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)
    targets = np.array(targets)

    return [tokens, masks, segments], targets


# train 데이터를 Bert의 Input 타입에 맞게 변환
train_x, train_y = convert_data(X_train, y_train)

# test 데이터를 Bert의 Input 타입에 맞게 변환
test_x, test_y = convert_data(X_test, y_test)


# token, mask, segment 입력 정의
token_inputs = tf.keras.layers.Input((MAX_SEQ_LEN,), dtype = tf.int32, name = 'i                                                                                                             nput_word_ids')
mask_inputs = tf.keras.layers.Input((MAX_SEQ_LEN,), dtype = tf.int32, name = 'in                                                                                                             put_masks')
segment_inputs = tf.keras.layers.Input((MAX_SEQ_LEN,), dtype = tf.int32, name =                                                                                                              'input_segment')
bert_outputs = model([token_inputs, mask_inputs, segment_inputs])


bert_outputs

bert_output = bert_outputs[0]

DROPOUT_RATE = 0.5
NUM_CLASS = 3
dropout = tf.keras.layers.Dropout(DROPOUT_RATE)(bert_output)
# Multi-class classification 문제이므로 activation function은 softmax로 설정
sentiment_layer = tf.keras.layers.Dense(NUM_CLASS, activation='softmax', kernel_                                                                                                             initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02))(dropout)
sentiment_model = tf.keras.Model([token_inputs, mask_inputs, segment_inputs], se                                                                                                             ntiment_layer)


# 옵티마이저 Rectified Adam 하이퍼파리미터 조정
OPTIMIZER_NAME = 'RAdam'
LEARNING_RATE = 5e-5
TOTAL_STEPS = 10000
MIN_LR = 1e-5
WARMUP_PROPORTION = 0.1
EPSILON = 1e-8
CLIPNORM = 1.0
optimizer = tfa.optimizers.RectifiedAdam(learning_rate = LEARNING_RATE,
                                          total_steps = TOTAL_STEPS,
                                          warmup_proportion = WARMUP_PROPORTION,                                                                                                             
                                          min_lr = MIN_LR,
                                          epsilon = EPSILON,
                                          clipnorm = CLIPNORM)

# 감정분류 모델 컴파일
sentiment_model.compile(optimizer = optimizer,
                        loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                        metrics = ['accuracy'])


MIN_DELTA = 1e-3
PATIENCE = 20

early_stopping = EarlyStopping(
    monitor = "val_accuracy",
    min_delta = MIN_DELTA,
    patience = PATIENCE)

# 최고 성능의 모델 파일을 저장할 이름과 경로 설정
BEST_MODEL_NAME = '/home/ubuntu/Test_AlphaType/best_model.h5'


model_checkpoint = ModelCheckpoint(
    filepath = BEST_MODEL_NAME,
    monitor = "val_loss",
    mode = "min",
    save_best_only = True, # 성능 향상 시에만 모델 저장
    verbose = 1
)


callbacks = [early_stopping, model_checkpoint]

EPOCHS = 100
BATCH_SZIE = 32


sentiment_model.fit(train_x, train_y,
                    epochs = EPOCHS,
                    shuffle = True,
                    batch_size = BATCH_SZIE,
                    validation_data = (test_x, test_y),
                    callbacks = callbacks
                    )

# 최고 성능의 모델 불러오기
sentiment_model_best = tf.keras.models.load_model(BEST_MODEL_NAME,
                                                  custom_objects={'TFBertForSequ                                                                                                             enceClassification': TFBertForSequenceClassification})


# 모델이 예측한 라벨 도출
predicted_value = sentiment_model_best.predict(test_x)
predicted_label = np.argmax(predicted_value, axis = 1)


# Classification Report 저장
CL_REPORT_FILE = "/home/ubuntu/Test_AlphaType/cl_report.csv"

cl_report = classification_report(test_y, predicted_label, output_dict = True)
cl_report_df = pd.DataFrame(cl_report).transpose()
cl_report_df = cl_report_df.round(3)
cl_report_df.to_csv(CL_REPORT_FILE)
print(cl_report_df)
