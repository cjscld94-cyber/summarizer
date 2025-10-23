import textwrap
import streamlit as st
from openai import OpenAI

# ---------------------------
# 설정 & 상수
# ---------------------------
st.set_page_config(page_title="학생 프로젝트 보고서 요약기+", page_icon="📝", layout="wide")
st.title("📝 학생 프로젝트 보고서 요약기+")
st.caption("보고서를 50/100/300/500자로 요약하고, AI 추천 질문 기반 관점 요약도 생성합니다.")

# OpenAI 클라이언트
client = OpenAI(api_key=st.secrets["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"])

SAMPLE_REPORT = (
    "우리 팀은 기후 변화로 인한 이상기온과 자연재해 발생을 예측하기 위해 인공지능 기술을 활용한 프로젝트를 진행하였다. "
    "먼저 지난 20년간의 국내외 기상 데이터를 수집하여 평균 기온, 강수량, 이산화탄소 농도 등의 주요 변수를 정리하였다. "
    "이후 데이터를 학습시키기 위해 Python과 TensorFlow를 활용하여 기온 예측 모델을 설계하였다. 초기에는 단순 선형회귀를 적용했지만 예측 오차가 컸기 때문에, "
    "다층 퍼셉트론(MLP) 모델로 구조를 바꾸고 학습률과 은닉층 수를 조정하면서 정확도를 높였다. 또한 기상청 오픈데이터 API를 통해 실시간 데이터를 추가로 받아 "
    "모델이 새로운 입력에도 대응할 수 있도록 했다. 모델 학습 결과, 평균 제곱 오차(MSE)가 0.15로 줄어들며 성능이 향상되었고, 시각화를 통해 특정 지역의 온도 상승 추세를 "
    "확인할 수 있었다. 예를 들어, 서울과 강릉 지역은 지난 10년간 여름철 평균기온이 꾸준히 상승하는 경향을 보였고, 우리 모델은 향후 5년간 평균기온이 약 1.2도 상승할 것으로 "
    "예측했다. 프로젝트 후반부에는 단순한 예측을 넘어 ‘기후 행동’으로의 연결을 고민하였다. 우리는 예측 결과를 바탕으로 지역별 온실가스 감축 시나리오를 제안하고, 이를 시각화 "
    "대시보드로 구현하였다. Streamlit을 이용해 누구나 접근 가능한 웹 형태로 배포했으며, 이를 통해 학급 친구들이 자신의 지역 데이터를 직접 탐색하고 기후 변화의 심각성을 "
    "체감할 수 있도록 했다. 이번 활동을 통해 우리는 인공지능이 단순한 기술이 아니라 사회 문제 해결의 강력한 도구가 될 수 있음을 배웠다. 또한 데이터의 품질과 전처리 과정의 "
    "중요성을 실감했으며, 앞으로는 더 다양한 기후 변수와 지역 데이터를 반영하여 예측의 정확도를 높이고 싶다. 무엇보다 협업 과정에서 각자의 역할을 책임감 있게 수행하는 것이 "
    "프로젝트 성공의 핵심이라는 점을 깨달았다."
)

# ---------------------------
# 세션 상태 기본값
# ---------------------------
if "report_input" not in st.session_state:
    st.session_state.report_input = ""
if "reco_questions" not in st.session_state:
    st.session_state.reco_questions = []
if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

# ---------------------------
# 사이드바: 모델/옵션
# ---------------------------
with st.sidebar:
    st.header("⚙️ 옵션")
    model = st.selectbox(
        "모델 선택",
        options=["gpt-4o-mini", "gpt-4o"],
        index=0,
        help="요약/질문 생성에 사용할 모델을 고릅니다.",
    )
    temperature = st.slider("창의성(temperature)", 0.0, 1.0, 0.2, 0.05)
    st.caption("※ 정확한 요약은 낮은 값 권장")

# ---------------------------
# 유틸 함수
# ---------------------------
def trim_to_chars(text: str, limit: int) -> str:
    """문장 자연스러움을 해치지 않도록 문자 수 제한 내로 자르기."""
    if len(text) <= limit:
        return text.strip()
    cut = text[:limit].rstrip()
    endings = ["다.", ".", "!", "?", "요.", "임.", "습니다.", "했다."]
    last_end = -1
    for end in endings:
        pos = cut.rfind(end)
        if pos > last_end:
            last_end = pos + len(end)
    if last_end >= max(10, int(limit * 0.4)):
        return cut[:last_end].strip()
    return cut.strip()

def summarize_with_limit(report: str, limit: int, teacher_hint: str | None = None) -> str:
    """OpenAI로 요약 후 문자 수 제한 보정."""
    base_rules = (
        "규칙:\n"
        "1) 한국어 한 단락\n"
        "2) 새로운 사실 추가 금지, 원문 핵심만\n"
        "3) 목적→주요 수행→성과/지표→배운 점/다음 단계 흐름 선호\n"
        "4) 수치/지표 존재 시 명시\n"
        f"5) 공백 포함 {limit}자 이내 목표\n"
    )
    perspective = ""
    if teacher_hint:
        perspective = f"\n교사 질문 관점 지시: '{teacher_hint}' 관점에서 관련성 높은 내용만 선별해 요약.\n"

    prompt = (
        "다음은 고등학생의 프로젝트 활동 보고서다. 지시에 따라 요약하라.\n\n"
        f"{base_rules}{perspective}\n"
        "[보고서 본문]\n"
        f"{report}\n\n"
        "출력은 불릿/번호 없이 한 단락으로만."
    )
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=float(temperature),
    )
    return trim_to_chars(resp.output_text, limit)

def generate_recommended_questions(report: str, k: int = 5) -> list:
    """보고서 기반 교사용 추천 질문 생성."""
    prompt = (
        "다음 학생 프로젝트 보고서를 읽고, 교사가 관점 요약에 활용할 수 있는 질문을 한국어로 5개 제안하라.\n"
        "- 각 질문은 한 줄, 30자 이내, 모호한 표현 지양, 구체적 관점 제시\n"
        "- 예: '데이터 전처리의 타당성 중심', '협업 과정의 역할 분담과 갈등 해결', '성과 지표의 신뢰도와 한계'\n"
        f"\n[보고서]\n{report}\n\n"
        "출력은 번호 없이 줄바꿈으로만 구분된 5개 질문."
    )
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.3,
    )
    lines = [ln.strip("-• ").strip() for ln in resp.output_text.split("\n") if ln.strip()]
    # 상위 k개만, 길이 필터
    cleaned = []
    for q in lines:
        if len(q) <= 30 and q not in cleaned:
            cleaned.append(q)
        if len(cleaned) == k:
            break
    # 부족하면 기본 백업 질문 추가
    backup = [
        "데이터 전처리의 타당성 중심",
        "모델 선택과 하이퍼파라미터 근거",
        "예측 결과의 신뢰도와 한계",
        "협업의 역할 분담·갈등 해결",
        "다음 단계와 개선 계획"
    ]
    for b in backup:
        if len(cleaned) >= k:
            break
        if b not in cleaned:
            cleaned.append(b)
    return cleaned[:k]

# ---------------------------
# 입력 영역
# ---------------------------
st.subheader("1) 1000자 보고서 붙여넣기")

col_top = st.columns([1, 2, 1])
with col_top[0]:
    use_sample = st.checkbox("샘플 입력 사용", value=False,
                             help="체크하면 입력창이 샘플 보고서로 채워집니다.")
with col_top[2]:
    clear_btn = st.button("입력 초기화", help="입력창과 추천 질문을 초기화합니다.")

# 상태 업데이트
if clear_btn:
    st.session_state.report_input = ""
    st.session_state.reco_questions = []
    st.session_state.selected_question = None

if use_sample and (not st.session_state.report_input or st.session_state.report_input.strip() == ""):
    # 샘플 사용이 체크되고 입력이 비어있으면 샘플로 채움
    st.session_state.report_input = SAMPLE_REPORT

report = st.text_area(
    "학생 보고서",
    key="report_input",
    height=280,
    placeholder="학생이 작성한 프로젝트 보고서를 붙여넣어 주세요.",
)

# ---------------------------
# 버튼 영역
# ---------------------------
colA, colB = st.columns([1, 1])
with colA:
    st.subheader("2) 자동 요약 (50/100/300/500자)")
    gen_default = st.button("요약 생성", use_container_width=True, type="primary")

with colB:
    st.subheader("3) AI 추천 질문 → 관점 요약")
    gen_questions = st.button("AI 추천 질문 생성", use_container_width=True)
    # 추천 질문 목록 표시 + 선택
    if st.session_state.reco_questions:
        st.markdown("**추천 질문 선택:**")
        st.session_state.selected_question = st.radio(
            label="질문을 선택하세요",
            options=st.session_state.reco_questions,
            index=0 if st.session_state.selected_question not in st.session_state.reco_questions else st.session_state.reco_questions.index(st.session_state.selected_question),
            key="selected_question_radio",
        )
        gen_q_summary = st.button("선택한 질문으로 관점 요약 생성", use_container_width=True)
    else:
        gen_q_summary = False

# ---------------------------
# 동작: 요약 생성
# ---------------------------
if gen_default:
    if not report.strip():
        st.warning("보고서를 먼저 입력해 주세요.")
    else:
        tabs = st.tabs(["50자", "100자", "300자", "500자"])
        for tab, limit in zip(tabs, [50, 100, 300, 500]):
            with tab:
                with st.spinner(f"{limit}자 요약 생성 중..."):
                    try:
                        summary = summarize_with_limit(report, limit)
                        st.write(summary)
                        st.caption(f"문자 수: {len(summary)}")
                    except Exception as e:
                        st.error(f"요약 중 오류가 발생했습니다: {e}")

# ---------------------------
# 동작: 추천 질문 생성
# ---------------------------
if gen_questions:
    if not report.strip():
        st.warning("보고서를 먼저 입력하거나 '샘플 입력 사용'을 체크해 주세요.")
    else:
        with st.spinner("추천 질문 생성 중..."):
            try:
                st.session_state.reco_questions = generate_recommended_questions(report, k=5)
                st.success("추천 질문이 생성되었습니다. 오른쪽에서 선택하세요.")
            except Exception as e:
                st.error(f"추천 질문 생성 중 오류가 발생했습니다: {e}")

# ---------------------------
# 동작: 선택 질문 관점 요약
# ---------------------------
if gen_q_summary:
    if not report.strip():
        st.warning("보고서를 먼저 입력해 주세요.")
    elif not st.session_state.selected_question:
        st.warning("추천 질문을 먼저 선택해 주세요.")
    else:
        q = st.session_state.selected_question
        with st.spinner(f"관점 요약 생성 중... ({q})"):
            try:
                q_limits = [300, 500]
                qt1, qt2 = st.tabs([f"관점 요약 {q_limits[0]}자", f"관점 요약 {q_limits[1]}자"])
                with qt1:
                    s1 = summarize_with_limit(report, q_limits[0], teacher_hint=q)
                    st.write(s1)
                    st.caption(f"문자 수: {len(s1)}")
                with qt2:
                    s2 = summarize_with_limit(report, q_limits[1], teacher_hint=q)
                    st.write(s2)
                    st.caption(f"문자 수: {len(s2)}")
            except Exception as e:
                st.error(f"관점 요약 생성 중 오류가 발생했습니다: {e}")

# ---------------------------
# 푸터
# ---------------------------
st.divider()
st.markdown(
    textwrap.dedent(
        """
        **사용 팁**
        - 보고서는 구체적으로 붙여넣을수록 요약 품질이 좋아집니다.
        - ‘AI 추천 질문’으로 생성된 항목을 선택하면, 그 관점에 특화된 요약을 생성합니다.
        """
    )
)
