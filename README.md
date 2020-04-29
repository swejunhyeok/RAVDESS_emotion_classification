# RAVDESS_emotion_classification
20-04-29

기존 RAVDESS를 활용하여 emotion classification을 진행한 Emotion Classification Based on Audiovisual 논문 구현을 중점으로 진행하였습니다. 우선 기존에 거의 pytorch만 사용했어서 pytorch로 초기 작업 진행하다가 학습이 40% 정확도에서 진행이 안되서 논문에서도 사용한 텐서플로로 진행하게 되었습니다. 텐서플로에 대한 지식이 pytorch에 비해 좀 부족해서 코드 부분이 많이 부족할 수 도 있습니다..

데이터셋은 song + speech의 동영상을 랜덤 샘플링하여 2000는 training 나머지는 test 데이터셋으로 분류하였습니다.

video_emotion_preprocessing.ipynb
preprocessing에서는 길이가 일정하지 않은 동영상이 input으로 들어오기 때문에
오디오는 최소값으로 잘라주었고, frame은 저번에 공유드린 Emotion Classification Based on Audiovisual 논문은 베이스로 했기 때문에 16개 추출을 하였습니다.
