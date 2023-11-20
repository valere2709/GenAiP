from flask import Flask, render_template, request
import subprocess
import os

app = Flask(__name__)

# 업로드된 파일을 저장할 디렉터리 경로
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 홈 페이지
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return render_template('page1.html')
    return render_template('index.html')

@app.route('/run_init_script', methods=['POST'])
def run_init_script():
    try:
        # init.py를 실행
        result = subprocess.run(['sudo', 'python3', '/home/ubuntu/Alpha/init.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # 결과 출력 (예: stdout 및 stderr)
        output = result.stdout
        error = result.stderr

        # 파일 내용 읽기
        with open('/home/ubuntu/f2/uploads/result.txt', 'r') as file:
            file_contents = file.read()

        # 원하는 방식으로 결과를 처리하거나 렌더링
        return render_template('result.html', output=output, error=error, file_contents=file_contents)
    except Exception as e:
        return str(e)

@app.route('/page1', methods=['GET', 'POST'])
def page1():
    if request.method == 'POST':
        # 첫 번째 텍스트 상자에서 입력된 내용을 가져옴
        content1 = request.form['user_input1']
        content2 = request.form['user_input2']
        content3 = request.form['user_input3']
        content4 = request.form['user_input4']
        content5 = request.form['user_input5']
        content6 = request.form['user_input6']
        content7 = request.form['user_input7']
        content8 = request.form['user_input8']
        content9 = request.form['user_input9']
        content10 = request.form['user_input10']

        # 두 내용을 결합하여 하나의 문자열로 만듦
        combined_content = f"{content1}\n{content2}\n{content3}\n{content4}\n{content5}\n{content6}\n{content7}\n{content8}\n{content9}\n{content10}"

        # 결합된 내용을 텍스트 파일로 저장
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'combined_input.txt'), 'w') as file:
            file.write(combined_content)

    return render_template('page1.html')

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host="0.0.0.0", port=8080, debug=True)
