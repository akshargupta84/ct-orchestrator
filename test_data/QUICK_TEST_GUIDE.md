# CT Orchestrator - Quick Test Guide

## 🚀 Start the App

```bash
cd ct-orchestrator
source venv/bin/activate
cd frontend
streamlit run app.py
```

App will open at: http://localhost:8501

---

## 📋 Test Scenario 1: Small Campaign (Should Pass Validation)

**CT Planner Tab → Create New Plan**

| Field | Value |
|-------|-------|
| Campaign Name | Q1 Brand Awareness Push |
| Brand Name | FreshBite Snacks |
| Budget | 3,000,000 |
| Start Date | (pick 30 days from now) |
| End Date | (pick 90 days from now) |
| Primary KPI | Awareness |
| Videos | 2 |
| Display | 4 |

Expected: ✅ Validation passes, plan generated

---

## 📋 Test Scenario 2: Over Limit (Should Fail Validation)

| Field | Value |
|-------|-------|
| Campaign Name | Test Over Limit |
| Brand Name | Test Brand |
| Budget | 3,000,000 |
| Videos | 5 |
| Display | 10 |

Expected: ❌ Validation fails - exceeds video limit (max 2) and display limit (max 5)

---

## 📋 Test Scenario 3: Medium Campaign

| Field | Value |
|-------|-------|
| Campaign Name | Summer Product Launch |
| Brand Name | TechFlow Electronics |
| Budget | 15,000,000 |
| Primary KPI | Consideration |
| Videos | 6 |
| Display | 12 |

Expected: ✅ Passes (tier allows 8 videos, 15 display)

---

## 📊 Test Results Upload

**Results Tab → Upload CSV**

1. Upload `test_data/sample_results_mixed.csv`
2. Click "Process & Analyze Results"
3. Review recommendations

Expected Results for mixed.csv:
- VID001 (Hero Spot 30s): ✅ PASS - 8.7% lift, stat sig
- VID002 (15s Cutdown): ✅ PASS - 5.3% lift, stat sig
- VID003 (Product Demo): ❌ FAIL - 2.7% lift, not sig
- VID004 (Testimonial): ✅ PASS - 6.0% lift, stat sig
- VID005 (Lifestyle): ❌ FAIL - 2.3% lift, not sig
- VID006 (Tutorial): ✅ PASS - 7.3% lift, stat sig
- DSP001 (Product Shot): ✅ PASS - 5.9% lift, stat sig
- DSP002 (Lifestyle): ❌ FAIL - 3.3% lift, not sig
- DSP003 (Promo): ✅ PASS - 7.3% lift, stat sig
- DSP004 (Brand): ❌ FAIL - 2.6% lift, not sig
- DSP005 (Native): ✅ PASS - 5.3% lift, stat sig
- DSP006 (Carousel): ✅ PASS - 3.9% lift, stat sig

Pass Rate: 8/12 = 67%

---

## 💬 Test Insights Chat

**Insights Tab**

Try these questions:
- "What are the testing limits for a $10M campaign?"
- "How much does video testing cost?"
- "What is the turnaround time for display tests?"
- "What happens if a creative doesn't reach statistical significance?"

---

## ⚙️ Test Admin Functions

**Admin Tab** (select Superuser role in sidebar)

1. View current rules summary
2. Upload test_data/ct_rules_sample.md (convert to PDF first, or just review)
3. Check system information

---

## 🔢 Budget Tier Reference

| Budget | Video Limit | Display Limit |
|--------|-------------|---------------|
| < $5M | 2 | 5 |
| $5M - $35M | 8 | 15 |
| $35M - $100M | 15 | 30 |
| > $100M | 25 | 50 |

---

## 💰 Cost Reference

| Asset | Cost |
|-------|------|
| Video | $5,000 |
| Display | $3,000 |
| Audio | $2,500 |
| Expedited | +50% |

---

## ⏱️ Turnaround Reference

| Asset | Standard | Expedited |
|-------|----------|-----------|
| Video | 14 days | 7 days |
| Display | 10 days | 5 days |
