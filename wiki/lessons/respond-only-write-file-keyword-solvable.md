---
id: respond-only-write-file-keyword-solvable
type: lesson
created: 2026-07-01
updated: 2026-07-01
tags: [eda, keyword, rule-based, respond-only, write-file]
related: [[class-bottleneck-analysis]]
summary: respond_only와 write_file은 키워드 규칙만으로 precision 90-100% 달성 가능
---

# respond_only / write_file은 키워드로 해결 가능

## Symptom
14 class 중 일부는 이미 F1>0.9인데, 이유를 분석해보니 강력한 키워드 신호가 있음.

## Root cause

### respond_only (F1=0.997)
다음 키워드가 있으면 **100% precision**:
- "여기까지", "마무리", "summarize", "오늘은", "충분한", "도움이", "마무리하자"
- 합성 데이터 생성기가 wrap-up 패턴에 고정 템플릿 사용

### write_file (F1=0.972)
- "만들어줘" precision=91.2%
- "써줘" precision=89.5%
- "골격" precision=96.4%
- "create" precision=57.8%, "rewrite" precision=74.0%
- 47.8%가 turn=1 (세션 초반, 새 파일 생성)

### plan_task (F1=0.583이지만 개선 여지)
- "손대기" 50%, "단계부터" 48.7%, "쪼개줘" 43%, "steps" 39%
- question_mark 40.3%, long_prompt 28.7%

### ask_user (F1=0.561이지만 개선 여지)
- "모르겠어" 59.9%, "도와줄래?" 58.9%, "좋을지" 48.1%
- error_name 10.6% (전체 1.8%), question_mark 42%

## Fix
TF-IDF가 이미 이런 키워드를 캡처하고 있어서 별도 rule은 불필요.
다만 plan_task/ask_user 구분을 위한 `error_x_question` interaction feature 추가.

## Generalization
- 합성 데이터는 class별 "템플릿"이 있어서 키워드 패턴이 규칙적
- F1이 이미 높은 class는 건드리지 말고, 낮은 class에 집중
- Macro-F1 최적화 = worst class 개선에 시간 투자
