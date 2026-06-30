# Data Documentation

이 디렉토리는 **데이터셋이 어떻게 만들어졌는지**, 어떤 오픈소스 데이터를 참고했는지, 어떤 도메인 특성이 있는지를 기록한다.

모든 agent(`/eda`, `/plan`, `/dev`, `/auto`)는 작업 시작 전 이 디렉토리를 읽어 도메인 컨텍스트를 확보한다.

## File Layout

```
data_docs/
├── README.md                   # 이 파일 (인덱스)
├── dataset_overview.md         # 데이터 스키마, 크기, 라벨 정의
├── generation_methodology.md   # 데이터 생성 방법 (사용자 작성)
├── domain_notes.md             # 도메인 특수성, 알려진 함정
└── references/                 # 참고한 오픈소스 데이터/논문
    ├── source_001_<name>.md
    └── source_002_<name>.md
```

## When to Update

| 시점 | 업데이트 항목 |
|---|---|
| 데이터 공개 직후 | `dataset_overview.md` (스키마, 컬럼, 샘플 수) |
| 사용자가 생성 방법을 알게 됨 | `generation_methodology.md` |
| 새 reference 발견 | `references/source_NNN_*.md` |
| EDA에서 새 도메인 패턴 발견 | `domain_notes.md` |

## Agent Reading Protocol

`/auto` Step 0과 `/plan` Step 0에서:
1. `ls data_docs/` 로 존재 여부 확인
2. 존재하면 모든 `*.md` 파일 읽기
3. 핵심 사실을 plan에 반영 (예: "데이터 출처가 X이므로 Y 도메인 지식 활용 가능")

## Reference Page Template

새 reference는 다음 frontmatter로 시작:
```yaml
---
source_id: source_NNN_short_name
url: https://...
license: <license>
relevance: high | medium | low
added: YYYY-MM-DD
---
```

내용:
- **출처**: 이 데이터/논문이 어디서 왔는지
- **참고한 부분**: 데이터 생성 시 어떤 점을 차용했는지
- **시사점**: 이 정보가 모델링에 주는 hint
