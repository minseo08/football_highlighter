# Highlight Extraction from Football Match Video

이 프로젝트는 SoccerNet 데이터셋을 활용하여 축구 영상에서 하이라이트를 추출하는 프로젝트이다.
프로젝트의 메인 경로는 다음과 같이 지정되어 있다고 가정한다: /content/drive/MyDrive/CVA


코드 실행 전에 아래와 같은 디렉토리 구조를 만족하여야 잘 작동할 것이다.

## 프로젝트 디렉토리 구조

/content/drive/MyDrive/CVA/ (프로젝트 폴더)
│
├── download.ipynb # SoccerNet 데이터 다운로드 python
├── preprocessing.ipynb # highlight / non-highlight 클립 생성 python
├── model.ipynb # 모델 학습 및 추론 python
├── checkpoints/ # 모델 가중치 저장 폴더 (사전에 생성 필요)
├── highlight_extract/ # 클립 저장 디렉토리 (사전에 생성 필요)
├── output5/ # 최종 추론된 하이라이트 영상 저장 경로 (사전에 생성 필요)
├── soccernet_ds/ # SoccerNet 데이터셋 디렉토리 (데이터 다운로드나 아래 공유된 구글 드라이브 활용)
└── input/ # inference 하고 싶은 input 영상 저장 경로 (사전에 생성 필요, 과제 제출 시 영상 한 개를 시범적으로 첨부 하였음)

과제 제출로 첨부되는 파일은 현 README.md, report.pdf, download.ipynb, preprocessing.ipynb, model.ipynb, input 폴더(inference data) 이다.

train에 사용한 실질적인 dataset은 너무 고용량인 탓에 압축이 불가능하여 download.ipynb를 실행시켜 다운로드를 받아야 한다.
유사시를 위해 해당 google drive 폴더 공유도 허용해놓아서, 이를 통해 접근도 가능할 것이다.
Google Drive 공유 link: https://drive.google.com/drive/folders/1FcPVhHsDiUFnguExCNJ06ZeLXGwskvzZ?usp=drive_link

보고서에도 작성된 내용이지만, 본인은 google drive를 사용 시, 용량 부족으로 inference 영상을 추가로 담아두기가 불가능하여서 한 영상을 하나씩 inference 할 수 밖에 없었다.

다소 불편하지만, inference를 하기 위해서는 input 폴더에 inference를 위한 영상이 있어야 할 것이기에, dataset(soccernet_ds)에서 inference를 원하는 mkv 파일을 input 폴더에 미리 이동시켜서 inference로 진행하여야 한다.
train을 전반전 영상(1_224p.mkv)으로 진행하기에, soccernet_ds에서 2_224p.mkv로 저장된 영상을 input 폴더에 이동시키면 된다.
다만, 불편을 줄이고자 과제 제출 시에 input이라는 폴더에 마지막으로 진행한 한 경기 inference 영상(2_224p.mkv)넣어 놓았다. 코드 그대로 진행 시 해당 영상을 input으로 inference가 진행될 것이다.
soccernet 영상이 아닌 다른 영상을 inference 하고 싶다면 해당 영상을 input 폴더에 넣어 놓아도 작동 가능할 것이다.

원하는 프로젝트 폴더 위치에
첨부된 ipynb파일 3개를 위치해놓고, 필요한 디렉토리(폴더)를 미리 생성해놓으면 원활한 프로젝트 수행 및 검증이 가능하다.

## 1. 데이터 다운로드: 'download.ipynb'

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="/content/drive/MyDrive/CVA/soccernet_ds")
soccernet_ds 폴더에 각 리그별, 경기별 영상과 라벨 파일이 자동으로 저장된다.
코드 그대로 진행 시 위 경로를 사용한다.

## 2. 데이터 전처리: 'preprocessing.ipynb'

highlight / non-highlight 클립을 추출하여 저장한다.

create_save() 함수 내부에서 아래 경로들이 사용된다.
root_dir = "/content/drive/MyDrive/CVA/soccernet_ds" # 지정했던 데이터셋 경로
save_dir = "/content/drive/MyDrive/CVA/highlight_extract"

위 코드 그대로 실행 시 클립은 highlight_extract 폴더에 저장되며, 해당 폴더가 사전에 생성되어 있어야 한다.

## 3. 모델 학습 및 추론: 'model.ipynb'

학습을 위한 입력 디렉토리: video_dir : preprocessing.ipynb에서 생성한 highlight_extract 폴더 지정
가중치 저장 경로: checkpoints/ 폴더 : 사전에 생성되어 있어야 함 (/content/drive/MyDrive/CVA/checkpoints)

위 경로를 지정해주면 학습이 가능하다.

inference를 위한 경로 지정은 아래와 같다.
video_path = "/content/drive/MyDrive/CVA/input/2_224p.mkv"             # Input 영상 경로
video_tensor_path = "/content/drive/MyDrive/CVA/output5/video_tensor.pt"  # tensor 파일 저장 경로
checkpoint_path = "/content/drive/MyDrive/CVA/checkpoints/epoch{n}.pth"  # 가장 좋은 성능의 epoch

하이라이트 inference 결과 클립 영상은 save_clip() 함수를 통해 저장되고, 아래와 같이 경로 변경 가능하다.
output_path = os.path.join("/content/drive/MyDrive/CVA/output5", f"highlights_{i}.mkv")

inference 결과 클립(여러 개의 highlights_*.mkv 파일)을 하나의 연속된 영상으로 결합하려면 다음 코드를 실행한다.
subprocess.run([
    "ffmpeg", "-f", "concat", "-safe", "0", "-i", "/content/drive/MyDrive/CVA/output5/input.txt",
    "-c", "copy", "/content/drive/MyDrive/CVA/output5/output.mkv"
])

마지막 셀에서 output 클립들이 위치한 경로를 input_file, output_file 경로로 지정해주고 실행하면 된다. 현재 코드에서 input_file과 output_file이 같은 이유는 ffmpeg을 이용해 같은 코덱으로 맞추기 위함이고, txt파일의 경우 하나의 영상으로 변환하기 위해 각 clip 위치를 저장해놓는 과정이라고 생각해도 무방하다. 

마지막 subprocess.run([
    "ffmpeg", "-f", "concat", "-safe", "0", "-i", "/content/drive/MyDrive/CVA/output5/input.txt",
    "-c", "copy", "/content/drive/MyDrive/CVA/output5/output.mkv"
])
의 마지막 인자에 원하는 하나의 하이라이트 영상 이름(경로)를 지정해주면 된다.