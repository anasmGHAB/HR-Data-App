
import io
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="HR Data Quality Check",
    page_icon="🧪",
    layout="wide",
)

CSS = """
<style>
:root {
    --bg1:#0f172a;
    --bg2:#111827;
    --bg3:#1e293b;
    --card:#ffffff;
    --text:#0f172a;
    --muted:#64748b;
    --good:#16a34a;
    --warn:#f59e0b;
    --bad:#dc2626;
    --accent:#14b8a6;
    --accent2:#7c3aed;
}
.stApp {
    background:
      radial-gradient(circle at top left, rgba(20,184,166,0.10), transparent 22%),
      radial-gradient(circle at top right, rgba(124,58,237,0.10), transparent 18%),
      linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
}
.hero {
    padding: 1.4rem 1.6rem;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(15,23,42,0.98), rgba(30,41,59,0.94));
    color: white;
    box-shadow: 0 20px 45px rgba(15, 23, 42, 0.18);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1.1rem;
}
.hero h1 { margin: 0 0 .35rem 0; font-size: 2.1rem; }
.hero p { margin: 0; color: #dbeafe; font-size: 1rem; }
.kpi-card {
    background: rgba(255,255,255,0.94);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 18px;
    padding: 1rem 1.1rem;
    box-shadow: 0 10px 25px rgba(15,23,42,0.06);
}
.section-card {
    background: rgba(255,255,255,0.96);
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 22px;
    padding: 1rem 1.1rem .8rem 1.1rem;
    box-shadow: 0 14px 30px rgba(15,23,42,0.06);
}
.badge {
    display: inline-block;
    padding: .20rem .55rem;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 700;
    margin-right: .35rem;
}
.badge-ok { background:#dcfce7; color:#166534; }
.badge-warn { background:#fef3c7; color:#92400e; }
.badge-bad { background:#fee2e2; color:#991b1b; }
.small-muted { color: #64748b; font-size: .88rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PLACEHOLDER_VALUES = {"", "n/a", "na", "null", "none", "unknown", "tbd", "missing", "-", "--"}
EMAIL_RE = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.I)

ALIASES = {
    "employee_id": ["employee_id", "emp_id", "employee number", "employee no", "person_id", "id"],
    "first_name": ["first_name", "firstname", "first name", "given_name", "given name"],
    "last_name": ["last_name", "lastname", "last name", "family_name", "surname"],
    "email": ["email", "email_address", "work_email", "corporate_email", "mail"],
    "phone": ["phone", "mobile", "phone_number", "telephone", "cell"],
    "gender": ["gender", "sex"],
    "date_of_birth": ["date_of_birth", "dob", "birth_date", "date of birth"],
    "hire_date": ["hire_date", "joining_date", "start_date", "employment_start_date", "hire date"],
    "termination_date": ["termination_date", "exit_date", "leave_date", "termination date", "end_date"],
    "employment_status": ["employment_status", "status", "employee_status", "employment status"],
    "employment_type": ["employment_type", "contract_type", "employee_type", "employment type"],
    "department": ["department", "dept", "function", "division"],
    "job_title": ["job_title", "title", "job", "role", "position"],
    "manager_id": ["manager_id", "supervisor_id", "reports_to", "line_manager_id"],
    "country": ["country", "country_name", "country code", "work_country"],
    "city": ["city", "location_city", "office_city"],
    "salary": ["salary", "base_salary", "salary_mad", "salary_eur", "annual_salary", "monthly_salary"],
    "fte": ["fte", "fte_ratio", "full_time_equivalent", "workload"],
    "performance_rating": ["performance_rating", "rating", "performance score", "performance_score"],
}

EXPECTED_COLUMNS = [
    "employee_id", "first_name", "last_name", "email", "date_of_birth", "hire_date",
    "employment_status", "department", "job_title", "manager_id"
]

VALID_GENDER = {"female", "male", "non-binary", "nonbinary", "undisclosed", "other"}
ACTIVE_STATUSES = {"active", "actif"}
TERMINATED_STATUSES = {"terminated", "inactive"}
LEAVE_STATUSES = {"leave", "on leave", "leave of absence"}


def normalize_column(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def find_matching_column(columns: List[str], aliases: List[str]) -> str | None:
    normalized_aliases = {normalize_column(a) for a in aliases}
    for col in columns:
        if normalize_column(col) in normalized_aliases:
            return col
    return None


def detect_schema(df: pd.DataFrame) -> Dict[str, str]:
    detected = {}
    for canonical, aliases in ALIASES.items():
        match = find_matching_column(df.columns.tolist(), aliases)
        if match:
            detected[canonical] = match
    return detected


def coerce_str(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def standardize_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            s = coerce_str(out[col]).str.lower()
            out.loc[s.isin(PLACEHOLDER_VALUES), col] = pd.NA
    return out


def add_issue(store: List[dict], severity: str, category: str, column: str, count: int, description: str, examples=None):
    if count <= 0:
        return
    if examples is None:
        examples = []
    examples = [str(x) for x in examples if pd.notna(x)]
    store.append(
        {
            "Severity": severity.title(),
            "Category": category,
            "Column": column,
            "Rows impacted": int(count),
            "Description": description,
            "Examples": ", ".join(examples[:4]),
        }
    )


def parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def parse_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype("string").str.replace(",", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def issue_table_to_excel(issues_df: pd.DataFrame, flagged_rows: pd.DataFrame, profiling_df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        issues_df.to_excel(writer, index=False, sheet_name="Issue_Log")
        flagged_rows.to_excel(writer, index=False, sheet_name="Flagged_Rows")
        profiling_df.to_excel(writer, index=False, sheet_name="Column_Profile")
    return output.getvalue()


def color_for_score(score: float) -> str:
    if score >= 90:
        return "🟢"
    if score >= 75:
        return "🟡"
    return "🔴"


def analyze_hr_data(df: pd.DataFrame) -> dict:
    raw = df.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    clean = standardize_missing(raw)

    schema = detect_schema(clean)
    issues: List[dict] = []
    flags = pd.DataFrame(index=clean.index)
    n_rows = len(clean)
    n_cols = len(clean.columns)

    placeholder_counts = []
    for col in raw.columns:
        if pd.api.types.is_object_dtype(raw[col]) or pd.api.types.is_string_dtype(raw[col]):
            original = coerce_str(raw[col]).str.lower()
            placeholder_counts.append(
                {"Column": col, "Placeholder-like values": int(original.isin(PLACEHOLDER_VALUES).sum())}
            )
    placeholder_df = pd.DataFrame(placeholder_counts).sort_values("Placeholder-like values", ascending=False)

    missing_pct = clean.isna().mean().mul(100).round(1).sort_values(ascending=False)
    overall_missing = float(clean.isna().mean().mean() * 100)

    duplicate_rows_mask = clean.fillna("__MISSING__").duplicated()
    flags["duplicate_row"] = duplicate_rows_mask
    add_issue(
        issues, "high", "Uniqueness", "All columns", int(duplicate_rows_mask.sum()),
        "Exact duplicate records detected."
    )

    missing_expected = [col for col in EXPECTED_COLUMNS if col not in schema]
    profiling = []
    for col in clean.columns:
        profiling.append(
            {
                "Column": col,
                "Detected dtype": str(clean[col].dtype),
                "Missing %": round(float(clean[col].isna().mean() * 100), 1),
                "Distinct values": int(clean[col].nunique(dropna=True)),
                "Top sample values": ", ".join(map(str, clean[col].dropna().astype(str).head(3).tolist())),
            }
        )
    profiling_df = pd.DataFrame(profiling)

    # Employee ID checks
    dup_id_count = 0
    if "employee_id" in schema:
        col = schema["employee_id"]
        ids = coerce_str(clean[col]).str.upper()
        miss_id = ids.isna() | (ids == "")
        flags["employee_id_missing"] = miss_id
        add_issue(issues, "critical", "Completeness", col, int(miss_id.sum()), "Missing employee identifier values.")
        dup_mask = ids.notna() & ids.duplicated(keep=False)
        flags["employee_id_duplicate"] = dup_mask
        dup_id_count = int(dup_mask.sum())
        examples = ids[dup_mask].dropna().unique()[:4].tolist()
        add_issue(issues, "critical", "Uniqueness", col, dup_id_count, "Duplicate employee identifiers detected.", examples)
    else:
        add_issue(issues, "high", "Schema", "employee_id", 1, "No employee ID column detected. Uniqueness checks are limited.")

    # Name checks
    for logical in ["first_name", "last_name"]:
        if logical in schema:
            col = schema[logical]
            miss = clean[col].isna()
            flags[f"{logical}_missing"] = miss
            add_issue(issues, "medium", "Completeness", col, int(miss.sum()), f"Missing {logical.replace('_', ' ')} values.")

    # Email checks
    dup_email_count = 0
    if "email" in schema:
        col = schema["email"]
        emails = coerce_str(clean[col]).str.lower()
        miss = emails.isna()
        flags["email_missing"] = miss
        add_issue(issues, "medium", "Completeness", col, int(miss.sum()), "Missing email addresses.")
        valid_mask = emails.fillna("").str.match(EMAIL_RE)
        invalid_email = emails.notna() & ~valid_mask
        flags["email_invalid"] = invalid_email
        add_issue(issues, "high", "Validity", col, int(invalid_email.sum()), "Invalid email format detected.", emails[invalid_email].tolist()[:4])
        dup_email = emails.notna() & emails.duplicated(keep=False)
        flags["email_duplicate"] = dup_email
        dup_email_count = int(dup_email.sum())
        add_issue(issues, "high", "Uniqueness", col, dup_email_count, "Duplicate email addresses detected.", emails[dup_email].unique()[:4].tolist())
        domains = emails.dropna().str.extract(r"@(.+)$", expand=False)
        if not domains.empty:
            dominant_domain = domains.mode().iloc[0] if not domains.mode().empty else None
            off_domain = domains.notna() & (domains != dominant_domain)
            flags["email_off_domain"] = off_domain
            add_issue(
                issues, "medium", "Consistency", col, int(off_domain.sum()),
                f"Email domain differs from dominant corporate domain ({dominant_domain}).",
                domains[off_domain].unique()[:4].tolist()
            )

    # Gender
    if "gender" in schema:
        col = schema["gender"]
        values = coerce_str(clean[col]).str.lower()
        unexpected = values.notna() & ~values.isin(VALID_GENDER)
        flags["gender_unexpected"] = unexpected
        add_issue(issues, "low", "Standardization", col, int(unexpected.sum()), "Unexpected gender labels outside the controlled list.", values[unexpected].unique()[:4].tolist())

    # Date checks
    invalid_date_count = 0
    dob = hire = term = None
    if "date_of_birth" in schema:
        col = schema["date_of_birth"]
        dob = parse_dates(clean[col])
        invalid_parse = clean[col].notna() & dob.isna()
        too_young = dob.notna() & (((pd.Timestamp.today().normalize() - dob).dt.days / 365.25) < 16)
        too_old = dob.notna() & (((pd.Timestamp.today().normalize() - dob).dt.days / 365.25) > 80)
        future_dob = dob.notna() & (dob > pd.Timestamp.today().normalize())
        flags["dob_invalid_parse"] = invalid_parse
        flags["dob_too_young"] = too_young
        flags["dob_too_old"] = too_old
        flags["dob_future"] = future_dob
        invalid_date_count += int(invalid_parse.sum() + too_young.sum() + too_old.sum() + future_dob.sum())
        add_issue(issues, "high", "Validity", col, int(invalid_parse.sum()), "Unparseable dates of birth.")
        add_issue(issues, "critical", "Validity", col, int(too_young.sum() + future_dob.sum()), "Implausible dates of birth (future or under-age employees).")

    if "hire_date" in schema:
        col = schema["hire_date"]
        hire = parse_dates(clean[col])
        invalid_parse = clean[col].notna() & hire.isna()
        future_hire = hire.notna() & (hire > (pd.Timestamp.today().normalize() + pd.Timedelta(days=30)))
        flags["hire_invalid_parse"] = invalid_parse
        flags["hire_future"] = future_hire
        invalid_date_count += int(invalid_parse.sum() + future_hire.sum())
        add_issue(issues, "high", "Validity", col, int(invalid_parse.sum()), "Unparseable hire dates.")
        add_issue(issues, "medium", "Validity", col, int(future_hire.sum()), "Future hire dates detected beyond 30 days.")

    if "termination_date" in schema:
        col = schema["termination_date"]
        term = parse_dates(clean[col])
        invalid_parse = clean[col].notna() & term.isna()
        flags["termination_invalid_parse"] = invalid_parse
        invalid_date_count += int(invalid_parse.sum())
        add_issue(issues, "high", "Validity", col, int(invalid_parse.sum()), "Unparseable termination dates.")

    # Employment status checks
    if "employment_status" in schema:
        col = schema["employment_status"]
        status = coerce_str(clean[col]).str.lower()
        allowed = ACTIVE_STATUSES | TERMINATED_STATUSES | LEAVE_STATUSES
        unexpected = status.notna() & ~status.isin(allowed)
        flags["status_unexpected"] = unexpected
        add_issue(issues, "medium", "Standardization", col, int(unexpected.sum()), "Unexpected employment status values.", status[unexpected].unique()[:4].tolist())

        if term is not None:
            active_with_term = status.isin(ACTIVE_STATUSES) & term.notna()
            term_without_date = status.isin(TERMINATED_STATUSES) & term.isna()
            flags["status_active_with_term"] = active_with_term
            flags["status_term_without_date"] = term_without_date
            add_issue(issues, "critical", "Consistency", col, int(active_with_term.sum()), "Active employees should not have a termination date.")
            add_issue(issues, "critical", "Consistency", col, int(term_without_date.sum()), "Terminated or inactive employees should have a termination date.")

    # Date consistency: termination before hire
    if hire is not None and term is not None:
        term_before_hire = hire.notna() & term.notna() & (term < hire)
        flags["termination_before_hire"] = term_before_hire
        add_issue(issues, "critical", "Consistency", "Hire_Date / Termination_Date", int(term_before_hire.sum()), "Termination date occurs before hire date.")

    # Manager hierarchy
    if "manager_id" in schema and "employee_id" in schema:
        mid_col = schema["manager_id"]
        emp_col = schema["employee_id"]
        manager_id = coerce_str(clean[mid_col]).str.upper()
        employee_id = coerce_str(clean[emp_col]).str.upper()
        valid_ids = set(employee_id.dropna().tolist())
        missing_manager = manager_id.isna()
        invalid_manager = manager_id.notna() & ~manager_id.isin(valid_ids)
        self_manager = manager_id.notna() & employee_id.notna() & (manager_id == employee_id)
        flags["manager_missing"] = missing_manager
        flags["manager_invalid"] = invalid_manager
        flags["manager_self"] = self_manager
        add_issue(issues, "medium", "Completeness", mid_col, int(missing_manager.sum()), "Missing manager references.")
        add_issue(issues, "high", "Integrity", mid_col, int(invalid_manager.sum()), "Manager IDs do not match any employee ID.", manager_id[invalid_manager].unique()[:4].tolist())
        add_issue(issues, "critical", "Integrity", mid_col, int(self_manager.sum()), "Employees cannot be their own manager.")

    # Salary, FTE, rating
    invalid_numeric_count = 0
    if "salary" in schema:
        col = schema["salary"]
        sal = parse_numeric(clean[col])
        invalid = clean[col].notna() & sal.isna()
        nonpositive = sal.notna() & (sal <= 0)
        q1, q3 = sal.dropna().quantile([0.25, 0.75]) if sal.dropna().shape[0] > 3 else (np.nan, np.nan)
        iqr = q3 - q1 if pd.notna(q3) and pd.notna(q1) else np.nan
        high_outlier = sal.notna() & pd.notna(iqr) & (sal > q3 + 1.5 * iqr)
        flags["salary_invalid"] = invalid
        flags["salary_nonpositive"] = nonpositive
        flags["salary_outlier"] = high_outlier
        invalid_numeric_count += int(invalid.sum() + nonpositive.sum() + high_outlier.sum())
        add_issue(issues, "high", "Validity", col, int(invalid.sum()), "Salary values are not numeric.")
        add_issue(issues, "critical", "Validity", col, int(nonpositive.sum()), "Salary must be strictly positive.")
        add_issue(issues, "medium", "Outlier", col, int(high_outlier.sum()), "Salary outliers detected using IQR.", sal[high_outlier].astype(str).tolist()[:4])

    if "fte" in schema:
        col = schema["fte"]
        fte = parse_numeric(clean[col])
        invalid = clean[col].notna() & fte.isna()
        out_of_range = fte.notna() & ((fte <= 0) | (fte > 1.2))
        flags["fte_invalid"] = invalid
        flags["fte_out_of_range"] = out_of_range
        invalid_numeric_count += int(invalid.sum() + out_of_range.sum())
        add_issue(issues, "high", "Validity", col, int(invalid.sum()), "FTE values are not numeric.")
        add_issue(issues, "high", "Validity", col, int(out_of_range.sum()), "FTE should typically be between 0 and 1.2.", fte[out_of_range].astype(str).tolist()[:4])

    if "performance_rating" in schema:
        col = schema["performance_rating"]
        rating = parse_numeric(clean[col])
        invalid = clean[col].notna() & rating.isna()
        out_of_range = rating.notna() & ((rating < 1) | (rating > 5))
        flags["rating_invalid"] = invalid
        flags["rating_out_of_range"] = out_of_range
        invalid_numeric_count += int(invalid.sum() + out_of_range.sum())
        add_issue(issues, "medium", "Validity", col, int(invalid.sum()), "Performance ratings are not numeric.")
        add_issue(issues, "medium", "Validity", col, int(out_of_range.sum()), "Performance ratings should be between 1 and 5.", rating[out_of_range].astype(str).tolist()[:4])

    if "phone" in schema:
        col = schema["phone"]
        phone_digits = coerce_str(clean[col]).str.replace(r"\D", "", regex=True)
        invalid_phone = clean[col].notna() & ~phone_digits.str.len().between(8, 15)
        flags["phone_invalid"] = invalid_phone
        add_issue(issues, "medium", "Validity", col, int(invalid_phone.sum()), "Phone numbers have implausible length.", clean.loc[invalid_phone, col].astype(str).tolist()[:4])

    # Sensitive columns
    sensitive_markers = ["national", "passport", "iban", "bank", "tax", "salary", "birth", "phone", "email"]
    sensitive_cols = [c for c in clean.columns if any(marker in normalize_column(c) for marker in sensitive_markers)]

    # Flagged rows
    flags = flags.fillna(False)
    flag_columns = [c for c in flags.columns if flags[c].any()]
    flagged_rows = clean.copy()
    if flag_columns:
        flagged_rows["DQ_Flags"] = flags.apply(lambda row: ", ".join([col for col, val in row.items() if bool(val)]), axis=1)
        flagged_rows = flagged_rows[flagged_rows["DQ_Flags"] != ""].copy()
    else:
        flagged_rows["DQ_Flags"] = ""
        flagged_rows = flagged_rows.iloc[0:0].copy()

    issues_df = pd.DataFrame(issues)
    if issues_df.empty:
        issues_df = pd.DataFrame(columns=["Severity", "Category", "Column", "Rows impacted", "Description", "Examples"])
    else:
        severity_order = pd.CategoricalDtype(["Critical", "High", "Medium", "Low"], ordered=True)
        issues_df["Severity"] = issues_df["Severity"].astype(severity_order)
        issues_df = issues_df.sort_values(["Severity", "Rows impacted"], ascending=[True, False]).reset_index(drop=True)
        issues_df["Severity"] = issues_df["Severity"].astype(str)

    # Component scores
    completeness_score = max(0.0, 100 - overall_missing - min(15, float(placeholder_df["Placeholder-like values"].sum()) / max(n_rows, 1) * 4))
    uniqueness_penalty = (
        duplicate_rows_mask.mean() * 100 * 1.8
        + (dup_id_count / max(n_rows, 1)) * 100 * 2.5
        + (dup_email_count / max(n_rows, 1)) * 100 * 1.8
    )
    uniqueness_score = max(0.0, 100 - uniqueness_penalty)
    validity_penalty = (invalid_date_count + invalid_numeric_count) / max(n_rows, 1) * 100 * 2.5
    validity_penalty += float((issues_df["Category"] == "Validity").sum()) * 1.0
    validity_score = max(0.0, 100 - validity_penalty)
    consistency_rows = 0
    for col in flags.columns:
        if any(token in col for token in ["status_", "manager_", "termination_before_hire"]):
            consistency_rows += int(flags[col].sum())
    consistency_score = max(0.0, 100 - (consistency_rows / max(n_rows, 1)) * 100 * 2.5)
    overall_score = round(0.30 * completeness_score + 0.20 * uniqueness_score + 0.30 * validity_score + 0.20 * consistency_score, 1)

    issue_severity_counts = issues_df.groupby("Severity", observed=False)["Rows impacted"].sum().reset_index()
    category_counts = issues_df.groupby("Category", observed=False)["Rows impacted"].sum().reset_index().sort_values("Rows impacted", ascending=False)

    summary = {
        "rows": n_rows,
        "columns": n_cols,
        "overall_score": overall_score,
        "critical_issues": int(issues_df.loc[issues_df["Severity"] == "Critical", "Rows impacted"].sum()) if not issues_df.empty else 0,
        "overall_missing_pct": round(overall_missing, 1),
        "duplicate_rows": int(duplicate_rows_mask.sum()),
        "missing_expected": missing_expected,
        "sensitive_cols": sensitive_cols,
        "components": {
            "Completeness": round(completeness_score, 1),
            "Uniqueness": round(uniqueness_score, 1),
            "Validity": round(validity_score, 1),
            "Consistency": round(consistency_score, 1),
        },
    }

    distributions = {}
    for key in ["employment_status", "department", "country", "gender"]:
        if key in schema:
            col = schema[key]
            dist = clean[col].fillna("Missing").astype(str).value_counts().reset_index()
            dist.columns = ["Value", "Count"]
            distributions[key] = dist

    return {
        "clean_df": clean,
        "schema": schema,
        "summary": summary,
        "missing_pct": missing_pct.reset_index().rename(columns={"index": "Column", 0: "Missing %"}),
        "placeholder_df": placeholder_df,
        "issues_df": issues_df,
        "profiling_df": profiling_df,
        "flagged_rows": flagged_rows,
        "severity_counts": issue_severity_counts,
        "category_counts": category_counts,
        "distributions": distributions,
        "flags": flags,
    }


def read_uploaded_file(uploaded_file) -> Tuple[Dict[str, pd.DataFrame], str]:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return {"CSV_Data": df}, "CSV_Data"
    sheets = pd.read_excel(uploaded_file, sheet_name=None)
    largest_sheet = max(sheets.items(), key=lambda x: len(x[1]))[0]
    return sheets, largest_sheet


def gauge(score: float):
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#14b8a6"},
                "steps": [
                    {"range": [0, 60], "color": "#fee2e2"},
                    {"range": [60, 80], "color": "#fef3c7"},
                    {"range": [80, 100], "color": "#dcfce7"},
                ],
            },
            title={"text": "Global HR Data Quality Score"},
        )
    ).update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20))


def section_title(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="section-card">
            <h3 style="margin:.1rem 0;">{title}</h3>
            <div class="small-muted">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.markdown(
        """
        <div class="hero">
            <h1>HR Data Quality Check</h1>
            <p>Upload an Excel or CSV HR file and get a polished audit on completeness, uniqueness, validity, consistency and HR-specific integrity rules.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## ⚙️ Inputs")
        uploaded_file = st.file_uploader("Upload HR Excel / CSV", type=["xlsx", "xls", "csv"])
        st.markdown("### Included checks")
        st.markdown(
            """
            - Missing values & placeholder text  
            - Duplicate employees, rows and emails  
            - Date logic & status consistency  
            - Salary / FTE / rating validation  
            - Manager hierarchy integrity  
            - Sensitive field detection  
            """
        )

    if uploaded_file is None:
        st.info("Upload your HR file to start the quality check.")
        demo_cols = st.columns(3)
        demo_cols[0].markdown('<div class="kpi-card"><b>Expected HR fields</b><br><span class="small-muted">Employee ID, Names, Email, DOB, Hire Date, Status, Department, Manager ID...</span></div>', unsafe_allow_html=True)
        demo_cols[1].markdown('<div class="kpi-card"><b>Best for</b><br><span class="small-muted">Employee master data, payroll extracts, workforce snapshots, onboarding / offboarding files.</span></div>', unsafe_allow_html=True)
        demo_cols[2].markdown('<div class="kpi-card"><b>Outputs</b><br><span class="small-muted">Quality score, issue log, flagged rows, schema detection and downloadable audit report.</span></div>', unsafe_allow_html=True)
        st.stop()

    try:
        sheets, default_sheet = read_uploaded_file(uploaded_file)
    except Exception as e:
        st.error(f"Unable to read the file: {e}")
        st.stop()

    with st.sidebar:
        selected_sheet = st.selectbox("Select sheet", list(sheets.keys()), index=list(sheets.keys()).index(default_sheet))
        show_preview_rows = st.slider("Preview rows", min_value=5, max_value=30, value=10)

    df = sheets[selected_sheet]
    analysis = analyze_hr_data(df)

    summary = analysis["summary"]
    issues_df = analysis["issues_df"]
    flagged_rows = analysis["flagged_rows"]
    profiling_df = analysis["profiling_df"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Global score", f"{color_for_score(summary['overall_score'])} {summary['overall_score']}/100")
    c2.metric("Rows × columns", f"{summary['rows']} × {summary['columns']}")
    c3.metric("Critical impact", f"{summary['critical_issues']}")
    c4.metric("Overall missingness", f"{summary['overall_missing_pct']}%")

    tabs = st.tabs(["Overview", "Missingness & Validity", "Consistency & Integrity", "HR Lens", "Issue Log & Export"])

    with tabs[0]:
        col_left, col_right = st.columns([1.05, 1])
        with col_left:
            st.plotly_chart(gauge(summary["overall_score"]), use_container_width=True)
        with col_right:
            comp_df = pd.DataFrame(
                {"Dimension": list(summary["components"].keys()), "Score": list(summary["components"].values())}
            )
            fig = px.bar(comp_df, x="Dimension", y="Score", text="Score", range_y=[0, 100])
            fig.update_traces(marker_color="#7c3aed", textposition="outside")
            fig.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

        info1, info2 = st.columns([1.15, 1])
        with info1:
            st.markdown("### Detected schema")
            if analysis["schema"]:
                schema_df = pd.DataFrame(
                    [{"Canonical field": k, "Detected column": v} for k, v in analysis["schema"].items()]
                )
                st.dataframe(schema_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No recognizable HR schema detected. Generic profiling still works.")

            if summary["missing_expected"]:
                st.markdown(
                    "Missing recommended columns: "
                    + " ".join([f'<span class="badge badge-warn">{c}</span>' for c in summary["missing_expected"]]),
                    unsafe_allow_html=True,
                )
        with info2:
            st.markdown("### Dataset preview")
            st.dataframe(df.head(show_preview_rows), use_container_width=True, hide_index=True)

    with tabs[1]:
        left, right = st.columns([1.1, 1])
        with left:
            st.markdown("### Top missing fields")
            miss_df = analysis["missing_pct"].sort_values("Missing %", ascending=False).head(15)
            fig = px.bar(miss_df, x="Missing %", y="Column", orientation="h", text="Missing %")
            fig.update_traces(marker_color="#14b8a6", textposition="outside")
            fig.update_layout(height=460, yaxis={"categoryorder": "total ascending"}, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            st.markdown("### Placeholder-like values")
            st.dataframe(
                analysis["placeholder_df"][analysis["placeholder_df"]["Placeholder-like values"] > 0],
                use_container_width=True,
                hide_index=True,
            )
            validity_issues = issues_df[issues_df["Category"].isin(["Validity", "Standardization", "Outlier"])]
            st.markdown("### Validity highlights")
            st.dataframe(validity_issues, use_container_width=True, hide_index=True)

    with tabs[2]:
        left, right = st.columns([1, 1])
        with left:
            st.markdown("### Issues by category")
            if not analysis["category_counts"].empty:
                fig = px.bar(analysis["category_counts"], x="Category", y="Rows impacted", text="Rows impacted")
                fig.update_traces(marker_color="#0f766e", textposition="outside")
                fig.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=60))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No issue categories identified.")
        with right:
            st.markdown("### Severity footprint")
            if not analysis["severity_counts"].empty:
                color_map = {"Critical": "#dc2626", "High": "#f59e0b", "Medium": "#2563eb", "Low": "#10b981"}
                fig = px.pie(analysis["severity_counts"], values="Rows impacted", names="Severity", hole=0.52, color="Severity", color_discrete_map=color_map)
                fig.update_layout(height=360, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No severity footprint to display.")

        st.markdown("### Flagged rows sample")
        if flagged_rows.empty:
            st.success("No row-level anomalies detected.")
        else:
            st.dataframe(flagged_rows.head(50), use_container_width=True, hide_index=True)

    with tabs[3]:
        dists = analysis["distributions"]
        chart_cols = st.columns(2)
        idx = 0
        for logical_key, title in [
            ("employment_status", "Employment status mix"),
            ("department", "Department distribution"),
            ("country", "Country distribution"),
            ("gender", "Gender labels"),
        ]:
            if logical_key in dists and not dists[logical_key].empty:
                with chart_cols[idx % 2]:
                    fig = px.bar(dists[logical_key].head(12), x="Value", y="Count", text="Count")
                    fig.update_traces(marker_color="#7c3aed", textposition="outside")
                    fig.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=50), xaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)
                idx += 1

        st.markdown("### Sensitive columns detected")
        if summary["sensitive_cols"]:
            st.markdown(
                " ".join([f'<span class="badge badge-bad">{c}</span>' for c in summary["sensitive_cols"]]),
                unsafe_allow_html=True,
            )
            st.caption("These columns may contain personal or confidential HR data. Apply extra access controls and masking where relevant.")
        else:
            st.caption("No obviously sensitive HR fields detected by the rule-based scan.")

        st.markdown("### Column profiling")
        st.dataframe(profiling_df, use_container_width=True, hide_index=True)

    with tabs[4]:
        st.markdown("### Full issue log")
        st.dataframe(issues_df, use_container_width=True, hide_index=True)

        report_bytes = issue_table_to_excel(issues_df, flagged_rows, profiling_df)
        st.download_button(
            "⬇️ Download audit report (Excel)",
            data=report_bytes,
            file_name="hr_data_quality_audit_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        csv_bytes = flagged_rows.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download flagged rows (CSV)",
            data=csv_bytes,
            file_name="hr_flagged_rows.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
