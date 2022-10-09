<p align="center"><img width="600" alt="image" src="https://user-images.githubusercontent.com/61443621/194584545-e3e17f3e-18f0-4d89-8159-dfa0f1fdd402.png"></p>

<div align="center">

  
## Menu Recommendation
B조: C조였으면 추천C스템
  
</div>

# 💫 프로젝트 목표
- 사용자가 최근 먹었던 메뉴를 고려하여 사용자가 선호할 메뉴 추천
- This service recommends a new menu based on what the user has eaten so far using the BERT4Rec model trained by the menu planner data and the survey data.

# 💻 활용 장비
**개발환경**
- Ubuntu 18.04.6 LTS
- GPU Tesla T4 
 
**Colaborate**
- Notion
- Zoom

# 🏃🏃‍♂️ 프로젝트 팀 구성 및 역할
- **권수현**: 데이콘 식단 데이터 수집 및 전처리, 모델 서치, RNN 기반 모델 튜닝, 설문조사 데이터셋 전처리, 발표자료 정리 및 제작
- **김채은**: 나이스 급식 식단 데이터 수집 및 전처리, 발표 
- **김형민**: 설문조사 포맷 제작, 데이터셋 태깅, 성능평가지표 수정 및 적용
- **손승진**: 프로젝트 리딩 및 일정 관리, 데이터셋 EDA, 결측치 제거 및 태깅, 모델 서치, GNN 기반 모델 튜닝
- **전혜령**: 아이디어 제시, 데이터 및 모델 서치, 데이터셋 EDA. Transformer, AutoEncoder 기반 모델 튜닝, 설문조사 데이터셋 전처리, 수정된 성능평가지표 적용

# 📌 개요
<p align="center"><img width="700" alt="image" src="https://user-images.githubusercontent.com/61443621/194619777-01170383-32b3-4413-8c7e-ff8fa26aaf86.png">
</p>

- 다양한 모델의 실험을 통하여 가장 Hit rate가 좋은 모델 서치 후, 자체적으로 수정한 Hit rate를 적용하여 모델 성능 평가

# 📊 데이터 소개
**데이콘 구내 식당 식단 데이터셋**
- 총 1255일의 구내 식당 식단표 
- 각 조식메뉴, 중식메뉴, 석식메뉴 별로 메인 메뉴추출 후 MenuID 부여
- 날짜 및 시간 순서대로 SessionID 부여
- category, property column 추가하여 사용  
- [데이콘 구내 식당 식단 데이터셋](https://dacon.io/competitions/official/235743/data)

**나이스 교육정보 개방 포털 급식 식단 정보 데이터셋**
- 총 409일의 가락고 식단표 
- 중식메뉴 별로 메인 메뉴 추출 후 MenuID 부여
- 날짜 순서대로 SessionID 부여
- category, property column 추가하여 사용  
- [나이스 교육정보 개방 포털 급식 식단 정보 데이터셋](https://open.neis.go.kr/portal/data/service/selectServicePage.do?page=1&rows=10&sortColumn=&sortDirection=&infId=OPEN17320190722180924242823&infSeq=1)

**설문조사 데이터셋**
- 51명을 대상으로 지난 3일 동안 먹은 메뉴 조사
  - 1일째 아침/점심/저녁, 2일째 아침/점심/저녁, 3일째 아침/점심/저녁
- 일부 메뉴들에 대해서는 위의 구내 식당 식단 데이터셋, 급식 식단 정보 데이터셋의 메뉴명으로 통일
  - 예) 돼지갈비 → 돼지갈비찜 
  - 동일 메뉴에 대해 MenuID를 동일하게 가져가기 위함
- 응답 순서대로 SessionID 부여
- category, property column 추가하여 사용   

**사용한 데이터프레임 형식 예시**

SessionID(userID)|time stamp|Menu|MenuId|category|property|
|:---:|:---:|:---:|:---:|:---:|:---:|
|0|0|마라샹궈|0|중식|볶음류|
|0|1|고등어구이|1|한식|구이류|
|0|2|피자|2|양식|패스트푸드류|


# 📝 모델 결과 
<p align="center"><img width="700" alt="image" src="https://user-images.githubusercontent.com/61443621/194623868-f7d6f63c-9c48-433c-86e0-230fd451731e.png">
</p>

- Transformer 기반의 모델인 BERT4Rec의 성능이 가장 좋은 것을 확인 가능 

**메뉴 추천 결과 예시**  
   
|먹은 메뉴|추천 메뉴|
|:---:|:---:|
|제육볶음, 삼겹살 구이, 간장 계란장, 청국장 찌개, 컵라면&찐계란, 시리얼 과일 샐러드, 된장찌개 | 동파삼겹수육 |

# 📱 프로젝트 결과
<p align="center"><img width="700" alt="image" src="https://user-images.githubusercontent.com/61443621/194756391-6977844a-ee98-4fc6-8b5d-239437768e56.png">
</p>

- 위와 같이 활용 가능할 것으로 기대됨

# 💭 Reference
- https://lsjsj92.tistory.com/590  
- https://greeksharifa.github.io/machine_learning/2021/07/03/SRGNN/  
- https://github.com/SeongBeomLEE/RecsysTutorial  
- https://github.com/flowel1/gru4rec-keras/blob/master/Gru4Rec_Keras.ipynb  

