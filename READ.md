# DQE Napari Tool

X-ray image 분석용 로컬 GUI 도구입니다. Raw/Result(=Corrected) 2-pane 뷰어와 ROI 기반 처리, 라인 검출/표시, blemish 검출을 제공합니다.

## 실행
```
python dqe_v15_patched_v24_line_detect.py
```

## 동작 흐름 (요약)
1) 모델/타입 선택 후 Raw 파일 로드  
2) 모델별 ROI 프리셋 자동 적용  
3) ROI 기준 라인 검출 및 오버레이 표시  
4) Result 이미지 생성(라인 노이즈 보정)  
5) Blemish 검출/리스트/크롭 표시  

## UI 구성
- 좌: Raw 뷰
- 중: Result(=Corrected) 뷰
- 우: Result(탭) + Blemish 리스트/Param

## 라인 검출 로직
ROI 평균 대비 낮은 DN이 일정 비율 이상인 row/col을 라인으로 판단합니다.
- **Line Drop Ratio**: ROI 평균 대비 얼마나 낮아야 low로 볼지 (0.9 = 평균의 10% 이하)
- **Line Low Fraction**: 해당 row/col에서 low 픽셀 비율 기준 (0.05 = 5%)
- **Line Ignore Margin(px)**: ROI 경계에서 제외할 마진
- **Line Exclude X/Y**: 제외할 라인 좌표(콤마 구분, 예: `x256,y1032,x112`)
- **Line Break Jump**: 끊긴 라인 판정용 단차 비율 (ROI 평균 * ratio 이상 단차 발생 시 Broken)

## 라인 표시 규칙
- Result(Collected) 오버레이:
  - 일반 Line: **주황색**
  - 3Line 또는 Broken Line: **노란색**
  - 불투명도 100%
- 리스트 표기:
  - 좌표: `(X,-)` 또는 `(-,Y)`
  - Thr(%) 칸에 `Line`, `Broken`, `3Line`, `3Line/Broken`으로 표시

## Blemish 주요 파라미터
Param 창에서 조정합니다.
- **analysis_scope**: ROI / Full
- **use_roi_as_reference**: ROI 평균을 기준값으로 사용
- **bg_window / measure_window**: 로컬 통계 창 크기
- **threshold_ref_percent_u/l**: blemish 판정 상/하한
- **threshold_crop_percent**: crop 기반 임계값
- **min_area / max_area**: blemish 면적 필터
- **min_dn**: 최소 DN
- **min_circularity / max_eccentricity**: 형상 필터

## 설정 파일
- `blemish_config.json`에 모델/타입별 파라미터 저장
- 실행 폴더에 파일이 없으면, 배포본에 포함된 기본 config를 첫 실행 시 자동 복사

## 라인/Result 보정
- Result 이미지는 ROI 기반으로 line noise 보정 후 표시
- 필요 시 원본 방향(orientation) 보정 모델은 로드 시 적용
