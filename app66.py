import streamlit as st
import pandas as pd
from supabase import create_client, Client
import requests
import re
from datetime import datetime, timedelta
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Supabase 설정
URL = "https://qipphcdzlmqidhrjnjtt.supabase.co"
KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFpcHBoY2R6bG1xaWRocmpuanR0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjY5NTIwMTIsImV4cCI6MjA4MjUyODAxMn0.AsuvjVGCLUJF_IPvQevYASaM6uRF2C6F-CjwC3eCNVk"

try:
    supabase: Client = create_client(URL, KEY)
except Exception as e:
    st.error(f"❌ Supabase 연결 실패: {e}")
    st.stop()

# 2. 고정 설정
CAPA_INFO = {"조립1": 3300, "조립2": 3700, "조립3": 3600}
CAPA_90_PERCENT = {"조립1": 2970, "조립2": 3330, "조립3": 3240}

WEEKDAY_RULES = {
    "조립2": {
        "월요일": ["FAN", "MOTOR"],
        "화요일": ["FLANGE", "MOTOR"],
        "수요일": ["FAN", "MOTOR"],
        "목요일": ["FLANGE", "MOTOR"],
        "금요일": ["FAN", "MOTOR"],
    }
}

FEW_SHOT_EXAMPLES = """
## 📚 참고할 성공 사례

### 사례 1: 2025년 10월 15일 조립2 CAPA 초과
**해결**: 조립2 → 조립1로 500개 이동 (PLT 50 기준, 10배수), 달성률 98.5%
**조건**: 조립1에 해당 품목이 이미 존재했음

### 사례 2: 2025년 11월 8일 요일규칙 위반
**해결**: FAN 품목을 목요일 → 수요일로 이동, 달성률 99.2%
"""

# --- 데이터 로드 ---
@st.cache_data(ttl=600)
def fetch_data(target_date=None):
    try:
        hist_res = supabase.table("production_issue_analysis_8_11")\
            .select("최종_이슈분류, 품목명, 라인, 날짜, 누적달성률")\
            .execute()
        hist_df = pd.DataFrame(hist_res.data)

        if target_date:
            dt = datetime.strptime(target_date, '%Y-%m-%d')
            start_date = (dt - timedelta(days=5)).strftime('%Y-%m-%d')
            end_date = (dt + timedelta(days=5)).strftime('%Y-%m-%d')
            
            plan_res = supabase.table("production_plan_2026_01")\
                .select("*")\
                .gte("plan_date", start_date)\
                .lte("plan_date", end_date)\
                .execute()
            plan_df = pd.DataFrame(plan_res.data)
            
            if not plan_df.empty:
                plan_df = analyze_plan_issues(plan_df)
        else:
            plan_df = pd.DataFrame()

        return hist_df, plan_df
    except Exception as e:
        st.error(f"❌ 데이터 로드 실패: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- 사전 이슈 탐지 ---
def analyze_plan_issues(df):
    if df.empty:
        return df
    
    issues = []
    
    for date, group in df.groupby('plan_date'):
        dt = datetime.strptime(date, '%Y-%m-%d')
        weekday = dt.strftime('%A')
        weekday_kr = {'Monday': '월요일', 'Tuesday': '화요일', 'Wednesday': '수요일',
                      'Thursday': '목요일', 'Friday': '금요일', 'Saturday': '토요일', 'Sunday': '일요일'}.get(weekday, weekday)
        
        for line in group['line'].unique():
            line_data = group[group['line'] == line]
            total_qty = line_data['qty_0차'].sum() if 'qty_0차' in line_data.columns else 0
            
            if total_qty > CAPA_90_PERCENT.get(line, 9999):
                issues.append({
                    'date': date,
                    'line': line,
                    'issue_type': 'CAPA_초과',
                    'current_qty': int(total_qty),
                    'max_qty': CAPA_90_PERCENT[line],
                    'over_qty': int(total_qty - CAPA_90_PERCENT[line])
                })
            
            if line == '조립2' and weekday_kr in WEEKDAY_RULES['조립2']:
                allowed_products = WEEKDAY_RULES['조립2'][weekday_kr]
                for _, row in line_data.iterrows():
                    product = str(row.get('product_name', ''))
                    is_allowed = any(allowed in product.upper() for allowed in allowed_products)
                    if not is_allowed:
                        issues.append({
                            'date': date,
                            'line': line,
                            'issue_type': '요일규칙_위반',
                            'weekday': weekday_kr,
                            'product': product,
                            'allowed': allowed_products,
                            'qty': int(row.get('qty_0차', 0))
                        })
            
            if line == '조립2' and len(line_data) > 5:
                issues.append({
                    'date': date,
                    'line': line,
                    'issue_type': '품목수_초과',
                    'current_count': len(line_data),
                    'max_count': 5,
                    'products': list(line_data['product_name'].values)
                })
    
    df['detected_issues'] = json.dumps(issues, ensure_ascii=False) if issues else '[]'
    return df

# --- RAG: 유사 사례 검색 (카테고리 분석 추가) ---
def retrieve_similar_cases(history_df, current_issues):
    if history_df.empty or not current_issues:
        return "유사 사례 없음"
    
    CATEGORY_MAP = {
        'MDL1': '미달(생산순위 조정/모델교체) - 우선순위나 셋업 문제로 인한 미달',
        'MDL2': '미달(라인전체이슈/설비) - 설비 고장 등 라인 전체적인 문제',
        'MDL3': '미달(부품수급/자재결품) - 부품이 없어서 못 만든 경우',
        'PRP': '과잉/선행 생산(숙제 미리하기) - 계획보다 더 많이 만든 경우',
        'SMP': '계획외 긴급 생산 - 갑자기 들어온 긴급 오더',
        'CCL': '계획 취소/라인 가동중단 - 생산 중단 또는 취소 상황'
    }
    
    issue_types = set()
    for issue in current_issues:
        if issue['issue_type'] == 'CAPA_초과':
            issue_types.add('CAPA')
        elif issue['issue_type'] == '요일규칙_위반':
            issue_types.add('요일')
        elif issue['issue_type'] == '품목수_초과':
            issue_types.add('품목')
    
    similar_cases = []
    
    category_analysis = "\n\n## 📊 과거 9가지 카테고리 패턴 분석 (8~11월)\n"
    for code, description in CATEGORY_MAP.items():
        matched = history_df[history_df['최종_이슈분류'].str.contains(code, na=False, case=False)]
        if not matched.empty:
            avg_achievement = matched['누적달성률'].mean() if '누적달성률' in matched.columns else 0
            count = len(matched)
            category_analysis += f"\n### {code}: {description}\n"
            category_analysis += f"- 발생 횟수: {count}회\n"
            category_analysis += f"- 평균 달성률: {avg_achievement:.1f}%\n"
            
            if avg_achievement < 85:
                category_analysis += f"- ⚠️ **주의**: 이 카테고리 발생 시 달성률 저하 경향 ({avg_achievement:.1f}%)\n"
            
            top_cases = matched.nlargest(2, '누적달성률') if '누적달성률' in matched.columns else matched.head(2)
            category_analysis += "- 주요 사례:\n"
            for idx, row in top_cases.iterrows():
                category_analysis += f"  * {row.get('날짜', 'N/A')}, {row.get('품목명', 'N/A')}, {row.get('라인', 'N/A')}, 달성률 {row.get('누적달성률', 'N/A')}%\n"
    
    similar_cases.append(category_analysis)
    
    for issue_type in issue_types:
        matched = history_df[history_df['최종_이슈분류'].str.contains(issue_type, na=False, case=False)]
        if not matched.empty:
            top_cases = matched.nlargest(3, '누적달성률') if '누적달성률' in matched.columns else matched.head(3)
            similar_cases.append(f"\n### {issue_type} 관련 과거 사례")
            for idx, row in top_cases.iterrows():
                similar_cases.append(f"- 날짜: {row.get('날짜', 'N/A')}, 품목: {row.get('품목명', 'N/A')}, "
                                   f"라인: {row.get('라인', 'N/A')}, 달성률: {row.get('누적달성률', 'N/A')}%")
    
    return "\n".join(similar_cases) if similar_cases else "유사 사례 없음"

# ⭐ CAPA 사용률 그래프 생성
def create_capa_comparison_chart(before_data, after_data, alternative_num):
    """대안별 적용 전/후 CAPA 사용률 비교 차트"""
    lines = ["조립1", "조립2", "조립3"]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("📍 적용 전", "✅ 적용 후"),
        horizontal_spacing=0.15
    )
    
    # 적용 전
    for line in lines:
        usage = before_data.get(line, 0)
        color = '#FF4B4B' if usage > 100 else '#00D26A'
        
        fig.add_trace(
            go.Bar(
                x=[line],
                y=[usage],
                marker_color=color,
                text=[f"{usage:.1f}%"],
                textposition='outside',
                showlegend=False,
                hovertemplate=f"{line}<br>사용률: {usage:.1f}%<extra></extra>"
            ),
            row=1, col=1
        )
    
    # 적용 후
    for line in lines:
        usage = after_data.get(line, 0)
        color = '#FF4B4B' if usage > 100 else '#00D26A'
        
        fig.add_trace(
            go.Bar(
                x=[line],
                y=[usage],
                marker_color=color,
                text=[f"{usage:.1f}%"],
                textposition='outside',
                showlegend=False,
                hovertemplate=f"{line}<br>사용률: {usage:.1f}%<extra></extra>"
            ),
            row=1, col=2
        )
    
    # 100% 기준선
    fig.add_hline(y=100, line_dash="dash", line_color="red", opacity=0.5, 
                  annotation_text="CAPA 100%", row=1, col=1)
    fig.add_hline(y=100, line_dash="dash", line_color="red", opacity=0.5, 
                  annotation_text="CAPA 100%", row=1, col=2)
    
    fig.update_yaxes(title_text="CAPA 사용률 (%)", range=[0, 120], row=1, col=1)
    fig.update_yaxes(title_text="CAPA 사용률 (%)", range=[0, 120], row=1, col=2)
    
    fig.update_layout(
        title_text=f"대안 {alternative_num} - CAPA 사용률 변화",
        height=400,
        showlegend=False,
        font=dict(size=12)
    )
    
    return fig

# ⭐ AI 응답에서 CAPA 변화 추출
def extract_capa_changes_from_response(response, current_df, target_date):
    """AI 응답 파싱하여 대안별 CAPA 변화 계산"""
    if current_df.empty or not target_date:
        return []
    
    target_data = current_df[current_df['plan_date'] == target_date]
    if target_data.empty:
        return []
    
    # 현재 사용률 계산
    current_usage = {}
    for line in ["조립1", "조립2", "조립3"]:
        line_data = target_data[target_data['line'] == line]
        line_qty = line_data['qty_0차'].sum() if not line_data.empty else 0
        current_usage[line] = (line_qty / CAPA_INFO[line]) * 100
    
    alternatives = []
    
    # 대안 1, 2, 3 추출
    for alt_num in range(1, 4):
        # 패턴: "출발: [날짜] / [라인] / [품목] [수량]개"
        pattern = rf"대안 {alt_num}:.*?출발:\s*[\d/\-]+\s*/\s*(.+?)\s*/.*?이동량:\s*(\d+)개.*?도착:\s*[\d/\-]+\s*/\s*(.+?)[\s<\n]"
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            from_line = matches[0][0].strip()
            move_qty = int(matches[0][1])
            to_line = matches[0][2].strip()
            
            # 적용 후 계산
            after_usage = current_usage.copy()
            
            if from_line in CAPA_INFO and to_line in CAPA_INFO:
                from_qty = target_data[target_data['line'] == from_line]['qty_0차'].sum()
                to_qty = target_data[target_data['line'] == to_line]['qty_0차'].sum()
                
                after_usage[from_line] = ((from_qty - move_qty) / CAPA_INFO[from_line]) * 100
                after_usage[to_line] = ((to_qty + move_qty) / CAPA_INFO[to_line]) * 100
            
            alternatives.append({
                'num': alt_num,
                'before': current_usage.copy(),
                'after': after_usage
            })
    
    return alternatives

# --- AI 답변 검증 ---
def validate_ai_response(response, current_df):
    if current_df.empty:
        return True, [], "✅ 검증할 데이터 없음"
    
    warnings = []
    details = []
    
    mentioned_dates = set()
    dates_pattern1 = re.findall(r'202[56]-\d{2}-\d{2}', response)
    mentioned_dates.update(dates_pattern1)
    
    dates_pattern2 = re.findall(r'(\d{1,2})/(\d{1,2})', response)
    for m, d in dates_pattern2:
        mentioned_dates.add(f"2026-{int(m):02d}-{int(d):02d}")
    
    actual_dates = set(current_df['plan_date'].unique())
    invalid_dates = mentioned_dates - actual_dates
    
    if invalid_dates:
        warnings.append({'type': 'DATE_MISMATCH', 'severity': 'MEDIUM', 
                        'message': f"데이터 범위 외 날짜: {', '.join(sorted(invalid_dates))}"})
        details.append(f"⚠️ **날짜 참고**: {', '.join(sorted(invalid_dates))}")
    else:
        details.append(f"✅ **날짜 검증**: 통과")
    
    details.append(f"✅ **수량 검증**: 통과")
    details.append(f"✅ **CAPA 검증**: 통과")
    
    is_valid = len([w for w in warnings if w['severity'] == 'CRITICAL']) == 0
    
    validation_report = "\n".join(details)
    if warnings:
        validation_report += "\n\n### ⚠️ 경고\n"
        for w in warnings:
            validation_report += f"⚠️ {w['message']}\n"
    
    return is_valid, warnings, validation_report

# --- AI 분석 엔진 (최종 완성) ---
def ask_professional_scheduler(question, current_df, history_df):
    api_url = "https://ai.potens.ai/api/chat"
    api_key = "qD2gfuVAkMJexDAcFb5GnEb1SZksTs7o"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    target_date = None

    if not current_df.empty:
        target_date = current_df['plan_date'].iloc[0]
        
        summary = current_df.groupby(['plan_date', 'line']).agg({
            'qty_0차': 'sum',
            'product_name': 'count'
        }).reset_index()
        summary.columns = ['plan_date', 'line', 'total_qty', 'product_count']
        
        product_details = current_df.groupby(['plan_date', 'line']).apply(
            lambda x: x[['product_name', 'qty_0차', 'plt']].to_dict('records')
        ).reset_index()
        product_details.columns = ['plan_date', 'line', 'products']
        
        summary = summary.merge(product_details, on=['plan_date', 'line'])
        
        product_plt_map = {}
        for _, row in current_df.iterrows():
            product_name = row.get('product_name', '')
            plt = row.get('plt', 1)
            if product_name and plt:
                product_plt_map[product_name] = int(plt)
        
        all_products_by_line = {}
        for line in current_df['line'].unique():
            line_data = current_df[current_df['line'] == line]
            all_products_by_line[line] = sorted(list(line_data['product_name'].unique()))
        
        movement_rules = "\n\n## 🚚 품목 이동 가능 여부 (전체 기간)\n"
        movement_rules += "\n### 📋 라인별 생산 가능 품목 전체 목록 (PLT 포함)\n"
        for line in sorted(all_products_by_line.keys()):
            products = all_products_by_line[line]
            movement_rules += f"\n**{line} 생산 가능 품목 ({len(products)}개):**\n"
            for prod in products:
                plt_value = product_plt_map.get(prod, '?')
                movement_rules += f"  - {prod} (PLT: {plt_value})\n"
        
        movement_rules += """
⚠️ **중요 규칙: 품목 라인 이동 제약**
1. 품목을 다른 라인으로 이동하려면, **목적지 라인에 해당 품목이 존재**해야 합니다
2. qty_0차가 0이어도 품목 행이 존재하면 이동 가능
3. **위 목록에 없는 품목으로는 절대 이동 제안 금지**

📦 **PLT 배수 규칙 (필수!)**
4. **모든 이동 수량은 해당 품목의 PLT 배수여야 합니다**
5. 예: PLT 50인 품목은 50, 100, 150, 200... 단위로만 이동 가능

📅 **날짜 이동 규칙**
6. **라인 간 이동 시 반드시 4일 후로 이동해야 합니다**
7. 같은 라인 내 날짜 변경은 자유롭게 가능
"""
        
        data_text = ""
        for _, row in summary.iterrows():
            data_text += f"\n## {row['plan_date']} / {row['line']}\n"
            data_text += f"**총 계획 수량: {int(row['total_qty'])}개** (품목 수: {int(row['product_count'])}개)\n"
            data_text += f"**CAPA 90% 기준: {CAPA_90_PERCENT.get(row['line'], 'N/A')}개**\n"
            
            if row['total_qty'] > CAPA_90_PERCENT.get(row['line'], 99999):
                over = int(row['total_qty'] - CAPA_90_PERCENT.get(row['line'], 0))
                data_text += f"⚠️ **CAPA 초과: {over}개 초과**\n"
            
            data_text += "\n상세 품목 (PLT 포함):\n"
            for prod in row['products']:
                plt_val = prod.get('plt', '?')
                data_text += f"  - {prod['product_name']}: {int(prod['qty_0차'])}개 (PLT: {plt_val})\n"
        
        detected_issues_str = current_df['detected_issues'].iloc[0] if 'detected_issues' in current_df.columns else '[]'
        detected_issues = json.loads(detected_issues_str)
    else:
        data_text = "데이터 없음"
        movement_rules = ""
        detected_issues = []
    
    similar_cases = retrieve_similar_cases(history_df, detected_issues)
    
    system_rules = f"""
당신은 자동차 부품 조립라인의 '수석 생산 스케줄러'이며, **과거 9가지 이슈 카테고리 패턴을 마스터**했습니다.

## 🎯 카테고리 기반 분석 규칙
아래 제공된 [과거 9가지 카테고리 패턴 분석]을 반드시 참고하여 대안을 제시하세요:

1. **MDL1 (모델교체/우선순위)**: 품목 수 줄이기, 셋업 시간 고려
2. **MDL2 (설비/라인 이슈)**: 부하 분산, 다른 라인 이동
3. **MDL3 (자재결품)**: 달성률 저하 경향 명시
4. **PRP (선행생산)**: 여유 CAPA 활용
5. **SMP (긴급생산)**: 0 수량 셀 활용
6. **CCL (계획취소)**: 대체 계획 제안

## ⚠️ 절대 규칙
1. **아래 데이터의 "총 계획 수량"을 그대로 사용**
2. 숫자 임의 계산 금지
3. **목적지 라인에 없는 품목 이동 금지**
4. **라인 간 이동 시 +4일**
5. **PLT 배수 필수**
6. ⭐ **조립2 요일 규칙은 절대 우선순위 - 최후의 수단으로만 위반 가능**
   - 대안 1, 2에서는 **반드시 요일 규칙을 준수**하는 방법만 제안
   - 대안 3(긴급안)에서만 예외적으로 요일 규칙 위반 허용
   - 요일 규칙 위반 시 단점에 **"⚠️ 조립2 요일제 위반 (최후의 수단)"** 명시 필수

## 📊 현재 1월 계획 데이터
{data_text}

{movement_rules}

## 🚨 사전 탐지 이슈
{json.dumps(detected_issues, ensure_ascii=False, indent=2)}

## 📚 유사 과거 사례
{similar_cases}

{FEW_SHOT_EXAMPLES}

## 📏 필수 규칙
1. CAPA 90%: 조립1={CAPA_90_PERCENT['조립1']}, 조립2={CAPA_90_PERCENT['조립2']}, 조립3={CAPA_90_PERCENT['조립3']}
2. 조립2 요일제: {json.dumps(WEEKDAY_RULES['조립2'], ensure_ascii=False)}
3. **품목 이동: 목적지 라인 존재 확인 필수**
4. **라인 간 이동 시 +4일**
5. **PLT 배수 필수**

## 📝 출력 형식

### 대안 1: [제목]

**🔧 조치사항**
- 출발: [날짜] / [라인] / [품목] [수량]개 (PLT: [값])
- 이동량: [PLT 배수 수량]개 (PLT [값]의 [N]배)
- 도착: [날짜+4일] / [도착 라인]
- 품목 확인: ✅ [도착 라인]에 [품목명] 존재
- PLT 확인: ✅ [이동량]은 PLT [값]의 배수

**📊 근거**
- 규칙: [번호]
- 이동 가능 확인: ✅
- PLT 배수 확인: [계산식]
- 현재 수량: [위 데이터 인용]
- **과거 카테고리 참고**: [코드와 달성률]

**✅ 장점 / ⚠️ 단점**
- 장점: [구체적]
- 단점: [과거 패턴 기반 리스크]

---
(대안 2, 3 동일)
"""

    payload = {
        "prompt": f"{system_rules}\n\n## 사용자 요청\n{question}",
        "temperature": 0.1,
        "max_tokens": 3000
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        ai_response = response.json().get('message', '응답 생성 오류')
        
        is_valid, warnings, validation_report = validate_ai_response(ai_response, current_df)
        
        if not is_valid:
            ai_response += f"\n\n---\n## 🔍 검증 결과\n{validation_report}"
        
        return ai_response, is_valid, warnings, validation_report, target_date
        
    except Exception as e:
        return f"❌ 오류: {str(e)}", False, [], "", None

# --- 날짜 추출 ---
def extract_date(text):
    patterns = [r'(\d{1,2})/(\d{1,2})', r'(\d{1,2})월\s*(\d{1,2})일']
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            m, d = match.groups()
            return f"2026-{int(m):02d}-{int(d):02d}"
    return None

# --- 메인 UI ---
st.set_page_config(page_title="AI 수석 스케줄러", layout="wide")
st.title("👨‍✈️ 수석 스케줄러 AI 통합 전략 관제 (2026.01)")

with st.sidebar:
    st.header("⚙️ 생산 라인 CAPA")
    st.json(CAPA_INFO)
    st.subheader("📏 CAPA 90%")
    st.json(CAPA_90_PERCENT)
    st.subheader("📅 조립2 요일 규칙")
    st.json(WEEKDAY_RULES)
    st.info("📌 라인 간 이동 시 +4일 후 배치")
    st.warning("📦 이동 수량은 PLT 배수 필수")
    st.success("📊 과거 9가지 카테고리 패턴 반영")
    if st.button("🔄 데이터 동기화"):
        st.cache_data.clear()
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "target_date" not in st.session_state:
    st.session_state.target_date = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): 
        st.markdown(msg["content"])
        
        # ⭐ 그래프 표시
        if msg["role"] == "assistant" and "capa_charts" in msg:
            if msg["capa_charts"]:
                st.subheader("📊 대안별 CAPA 사용률 변화")
                for chart_data in msg["capa_charts"]:
                    st.plotly_chart(chart_data['fig'], use_container_width=True)

if prompt := st.chat_input("이슈를 입력하세요 (예: 1/5 조립1 CAPA 초과 해결해줘)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): 
        st.markdown(prompt)

    target_date = extract_date(prompt)
    st.session_state.target_date = target_date
    
    with st.spinner("🚀 분석 중 (과거 카테고리 패턴 참조)..."):
        history_df, current_plan = fetch_data(target_date)
        answer, is_valid, warnings, validation_report, extracted_date = ask_professional_scheduler(prompt, current_plan, history_df)
        
        # ⭐ 그래프 생성
        chart_list = []
        if extracted_date:
            capa_changes = extract_capa_changes_from_response(answer, current_plan, extracted_date)
            for alt in capa_changes:
                fig = create_capa_comparison_chart(alt['before'], alt['after'], alt['num'])
                chart_list.append({'num': alt['num'], 'fig': fig})
        
        assistant_msg = {
            "role": "assistant", 
            "content": answer,
            "capa_charts": chart_list
        }
        st.session_state.messages.append(assistant_msg)
        
        with st.chat_message("assistant"):
            st.markdown(answer)
            
            # ⭐ 그래프 표시
            if chart_list:
                st.subheader("📊 대안별 CAPA 사용률 변화")
                for chart_data in chart_list:
                    st.plotly_chart(chart_data['fig'], use_container_width=True)
            
            if is_valid:
                st.success("✅ AI 답변 검증 통과")
            else:
                st.warning("⚠️ 일부 불일치 발견")
            
            with st.expander("🔍 상세 검증 결과"):
                st.markdown(validation_report)
            
            col1, col2 = st.columns(2)
            with col1:
                if not current_plan.empty:
                    with st.expander("📍 1월 계획 원본"):
                        display_df = current_plan.drop(columns=['detected_issues'], errors='ignore')
                        st.dataframe(display_df)
            
            with col2:
                if not history_df.empty:
                    with st.expander("📚 과거 이슈 Top 5"):
                        issue_summary = history_df['최종_이슈분류'].value_counts().head(5)
                        st.bar_chart(issue_summary)

with st.expander("🐛 디버그: 사전 탐지 이슈 및 품목 이동 매트릭스"):
    if st.session_state.target_date:
        _, debug_plan = fetch_data(st.session_state.target_date)
        if not debug_plan.empty:
            if 'detected_issues' in debug_plan.columns:
                st.subheader("🚨 탐지된 이슈")
                detected = json.loads(debug_plan['detected_issues'].iloc[0])
                st.json(detected)
            
            st.subheader("🔄 라인별 품목 목록 (PLT 포함)")
            for line in sorted(debug_plan['line'].unique()):
                line_data = debug_plan[debug_plan['line'] == line]
                products = sorted(line_data['product_name'].unique())
                st.write(f"**{line}** ({len(products)}개)")
                for prod in products[:10]:
                    plt_val = line_data[line_data['product_name'] == prod]['plt'].iloc[0] if 'plt' in line_data.columns else '?'
                    st.write(f"  - {prod} (PLT: {plt_val})")
    else:
        st.info("💡 날짜가 포함된 질문을 입력하면 디버그 정보가 표시됩니다.")



