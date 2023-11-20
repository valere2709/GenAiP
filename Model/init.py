import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_addons as tfa

# 모델 경로 설정 (BEST_MODEL_NAME을 사용)
MODEL_PATH = '/home/ubuntu/Alpha/best_model.h5'

# BERT 토크나이저 로드
MODEL_NAME = "klue/bert-base"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Custom Layer 정의
class TFBertForSequenceClassificationWrapper(TFBertForSequenceClassification):
    def __init__(self, config, *inputs, **kwargs):
        super(TFBertForSequenceClassificationWrapper, self).__init__(config, *inputs, **kwargs)

# 옵티마이저를 로드하기 위해 Rectified Adam을 등록
custom_objects = {'TFBertForSequenceClassification': TFBertForSequenceClassificationWrapper,
                  'RectifiedAdam': tfa.optimizers.RectifiedAdam}

# 모델 불러오기
model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)

# 함수를 사용하여 파일에서 입력 데이터를 읽어옴
def read_input_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().splitlines()
    return data

# 입력 문장을 받아와 BERT 토크나이징 및 패딩 처리
def preprocess_input(sentence):
    MAX_SEQ_LEN = 64
    input_ids = tokenizer.encode(sentence, add_special_tokens=True)
    input_ids = pad_sequences([input_ids], maxlen=MAX_SEQ_LEN, dtype="long",
                              value=0, truncating="post", padding="post")
    attention_mask = np.where(input_ids != 0, 1, 0)
    return input_ids, attention_mask

# 입력 데이터 파일 경로
input_file_path = '/home/ubuntu/f2/uploads/combined_input.txt'

# 파일에서 데이터를 읽어옴
input_data_list = read_input_file(input_file_path)

# 결과를 저장할 파일 경로
result_file_path = '/home/ubuntu/f2/uploads/result.txt'

# 모든 입력 데이터에 대한 예측 수행 및 결과 저장
with open(result_file_path, 'w', encoding='utf-8') as result_file:
    for i, user_sentence in enumerate(input_data_list):
        # 입력 문장 전처리
        input_ids, attention_mask = preprocess_input(user_sentence)

        # 모델 입력 형태 맞추기
        input_data = {
            'input_word_ids': input_ids,
            'input_masks': attention_mask,
            'input_segment': np.zeros_like(input_ids)
        }

        # 모델을 사용하여 예측 수행
        predicted_value = model.predict(input_data)
        predicted_label = np.argmax(predicted_value, axis=1)

        # 예측된 라벨 출력
        if predicted_label == 0:
            sentiment = "Negative"
        elif predicted_label == 1:
            sentiment = "Positive"
        else:
            sentiment = "Negative"

        # 예측의 확률값을 얻어옴
        predicted_prob = np.max(tf.nn.softmax(predicted_value), axis=1)

        # 결과를 파일에 저장
        result_file.write(f"입력 문장 {i + 1}: '{user_sentence}'\n")
        result_file.write(f"감정 예측: '{sentiment}'\n")
        result_file.write(f"감정 예측 확률: {predicted_prob[0]:.2%}\n\n")

print(f"결과가 {result_file_path} 파일에 저장되었습니다.")