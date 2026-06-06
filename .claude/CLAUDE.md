https://github.com/jyun-chae/skku-2-openai_pa2.git -> 지금 현제 github repo
train_50k_256.zip, valid_10k_256.zip과 같은 형태로 google drive project02/data/안에 data가 들어가 있어
project02.pdf를 구현하고 stylegen2.pdf 논문을 사용하는 쪽으로 모델을 구성 및 훈련할거야 stylegen2 말고도 유사한 논문들도 참고해
colab A100 GPU 환경에서 훈련할거라 Google Drive와 연동해서 훈련을 진행하고 있어
1024×1024 얼굴 이미지 생성기를 ONNX 파일로 제출합니다.
입력 텐서: z, shape (N, 512), dtype float32 (배치 차원은 동적이거나 고정 가능)
출력 텐서: shape (N, 3, 1024, 1024), dtype float32, range [-1, 1]