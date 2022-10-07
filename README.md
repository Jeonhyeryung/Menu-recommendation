<p align="center"><img width="600" alt="image" src="https://user-images.githubusercontent.com/61443621/194584545-e3e17f3e-18f0-4d89-8159-dfa0f1fdd402.png"></p>

<div align="center">

  
## Menu Recommendation
B조: C조였으면 추천C스템
  
</div>

# 💫 프로젝트 목표
- 사용자가 최근 먹었던 메뉴를 고려하여 사용자가 선호할 메뉴를 추천

# 💻 활용 장비
**개발환경**
- Ubuntu 18.04.6 LTS
- GPU Tesla T4
**Colaborate**
- Notion
- Zoom

# 🏃🏃‍♂️ 프로젝트 팀 구성 및 역할
- **권수현**: 
- **김채은**: 
- **김형민**:
- **손승진**:
- **전혜령**: 

# 📄 개요
<p align="center"><img width="750" alt="image" src="https://user-images.githubusercontent.com/61443621/194598433-b7329be3-416b-4baa-b34c-683d31ec01c1.png">
</p>
- 다양한 모델의 실험을 실험하여 가장 Hit rate가 좋은 모델 서치 후, 자체적으로 수정한 Hit rate를 적용하여 모델 성능 평가

# 📊 데이터 소개
**데이콘 구내 식당 식단 데이터셋**
- 각 조식메뉴, 중식메뉴, 석식메뉴 별로 메인 메뉴추출
- 날짜 및 시간 순서대로 SessinID 부여
- category, property column 추가하여 사용  
(다운로드 링크: https://dacon.io/competitions/official/235743/data)

**나이스 교육정보 개방 포털 급식 식단 정보 데이터셋**
- 총 409일의 가락고 식단표 
- 중식메뉴 별로 메인메뉴 1가지 추출
- 날짜 순서대로 SessinID 부여하여 사용
- category, property column 추가하여 사용  
(다운로드 링크: https://open.neis.go.kr/portal/data/service/selectServicePage.do?page=1&rows=10&sortColumn=&sortDirection=&infId=OPEN17320190722180924242823&infSeq=1)

**사용한 데이터프레임 형식 예시**

Session ID(user ID)|time stamp|Menu|Menu id|category|property|
|:---:|:---:|:---:|:---:|:---:|:---:|
|0|0|마라샹궈|0|중식|볶음류|
|0|1|고등어구이|1|한식|구이류|
|0|2|피자|2|양식|패스트푸드류|
 

# 모델 결과 

# 프로젝트 결과
