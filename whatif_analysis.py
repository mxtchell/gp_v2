from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pandas as pd
import jinja2
from ar_analytics import BreakoutAnalysisTemplateParameterSetup, ArUtils
from analysis_class_overrides.dimension_breakout import InsuranceLegacyBreakout
from ar_analytics.defaults import dimension_breakout_config, get_table_layout_vars, default_ppt_table_layout
from genpact_formatting import smart_title, genpact_format_number
from analysis_class_overrides.templates.default_table_with_chart import default_table_with_chart_layout
from skill_framework import SkillInput, SkillVisualization, skill, SkillParameter, SkillOutput, ParameterDisplayDescription
from skill_framework.layouts import wire_layout
from skill_framework.preview import preview_skill
from skill_framework.skills import ExportData

logger = logging.getLogger(__name__)

WHATIF_INSIGHT_PROMPT = """Analyze this what-if scenario and provide insights:

{{facts}}

The user is modeling a {{impact_pct}}% change in claims expense. Explain:
1. **Scenario Impact** - How the claims expense change affects each country/segment
2. **Profit Impact** - The projected effect on operating profit
3. **Key Takeaway** - Which areas would be most affected

Use markdown formatting. Be specific with numbers. Keep it under 150 words."""

WHATIF_MAX_PROMPT = """Answer the what-if question in 50 words or less using these facts:
{{facts}}
Impact percentage: {{impact_pct}}%"""

@skill(
    name="What-If Analysis",
    llm_name="whatif_analysis",
    description="Models what-if scenarios showing how a % change in claims_expense impacts operating_profit by country. METRICS ARE FIXED: claims_expense and operating_profit - do NOT ask user to select metrics.",
    capabilities="Model impact of claims expense changes on operating profit by country. Metrics are hardcoded - just need impact_pct, breakout, filters, and period.",
    limitations="Metrics are FIXED to claims_expense and operating_profit. Breakout by country. Assumes direct 1:1 inverse relationship (claims up = profit down).",
    example_questions="What if claims expense increases by 25% in Europe? How would a 10% rise in claims affect profitability by country? Model 15% claims increase impact on P&C.",
    parameter_guidance="Metrics are HARDCODED (claims_expense, operating_profit) - do NOT select metrics. Only need: impact_pct (e.g., 25 for +25%), breakouts (country), other_filters (region, business), periods.",
    parameters=[
        SkillParameter(
            name="metrics",
            is_multi=True,
            constrained_to="metrics",
            description="Metrics for what-if analysis. First metric is the input (changed by impact_pct), second is the output (impacted).",
            default_value=["claims_expense", "underwriting_profit"]
        ),
        SkillParameter(
            name="impact_pct",
            description="The percentage change to model for claims expense. Use positive for increase (e.g., 25 for +25%), negative for decrease (e.g., -10 for -10%).",
            default_value=10
        ),
        SkillParameter(
            name="periods",
            constrained_to="date_filter",
            is_multi=True,
            description="Time period for analysis. Format: 'q2 2023', '2021', 'jan 2023', 'ytd', 'mat', etc."
        ),
        SkillParameter(
            name="breakouts",
            is_multi=True,
            constrained_to="dimensions",
            description="Dimension to break out results by (e.g., country, business_segment)."
        ),
        SkillParameter(
            name="limit_n",
            description="Limit the number of rows returned",
            default_value=10
        ),
        SkillParameter(
            name="other_filters",
            is_multi=True,
            constrained_to="filters",
            description="Filters to apply (e.g., region, line_of_business)."
        ),
        SkillParameter(
            name="max_prompt",
            parameter_type="prompt",
            description="Prompt for max response.",
            default_value=WHATIF_MAX_PROMPT
        ),
        SkillParameter(
            name="insight_prompt",
            parameter_type="prompt",
            description="Prompt for detailed insights.",
            default_value=WHATIF_INSIGHT_PROMPT
        ),
        SkillParameter(
            name="table_viz_layout",
            parameter_type="visualization",
            description="Table Viz Layout",
            default_value=default_table_with_chart_layout
        )
    ]
)
def whatif_analysis(parameters: SkillInput):
    """What-If Analysis - model claims expense impact on underwriting profit"""

    # Extract parameters
    metrics = getattr(parameters.arguments, 'metrics', ["claims_expense", "underwriting_profit"]) or ["claims_expense", "underwriting_profit"]
    impact_pct = float(getattr(parameters.arguments, 'impact_pct', 10) or 10)
    periods = getattr(parameters.arguments, 'periods', []) or []
    breakouts = getattr(parameters.arguments, 'breakouts', None)
    limit_n = getattr(parameters.arguments, 'limit_n', 10) or 10
    other_filters = getattr(parameters.arguments, 'other_filters', []) or []

    # Get the two metrics - first is input (claims), second is output (profit)
    CLAIMS_METRIC = metrics[0] if len(metrics) > 0 else "claims_expense"
    PROFIT_METRIC = metrics[1] if len(metrics) > 1 else "underwriting_profit"

    print(f"What-If Analysis - Metrics: {CLAIMS_METRIC}, {PROFIT_METRIC}, Impact: {impact_pct}%, Breakouts: {breakouts}, Filters: {other_filters}")

    # Run breakout analysis for claims_expense
    claims_env = SimpleNamespace(
        periods=periods,
        metrics=[CLAIMS_METRIC],
        limit_n=limit_n,
        breakouts=breakouts,
        growth_type="Y/Y",
        other_filters=other_filters,
        growth_trend=None,
        calculated_metric_filters=None
    )
    BreakoutAnalysisTemplateParameterSetup(env=claims_env)
    claims_env.ba = InsuranceLegacyBreakout.from_env(env=claims_env)
    claims_env.ba.run_from_env()
    claims_tables = claims_env.ba.get_display_tables()

    # Run breakout analysis for operating_profit
    profit_env = SimpleNamespace(
        periods=periods,
        metrics=[PROFIT_METRIC],
        limit_n=limit_n,
        breakouts=breakouts,
        growth_type="Y/Y",
        other_filters=other_filters,
        growth_trend=None,
        calculated_metric_filters=None
    )
    BreakoutAnalysisTemplateParameterSetup(env=profit_env)
    profit_env.ba = InsuranceLegacyBreakout.from_env(env=profit_env)
    profit_env.ba.run_from_env()
    profit_tables = profit_env.ba.get_display_tables()

    # Build what-if table
    whatif_tables = build_whatif_tables(claims_tables, profit_tables, impact_pct, breakouts)

    # Build parameter display info
    param_info = [
        ParameterDisplayDescription(key="Scenario", value=f"{'+' if impact_pct >= 0 else ''}{impact_pct}% Claims Change"),
        ParameterDisplayDescription(key="Metrics", value="Claims Expense, Operating Profit"),
    ]
    if breakouts:
        param_info.append(ParameterDisplayDescription(key="Breakout", value=", ".join(breakouts) if isinstance(breakouts, list) else breakouts))

    # Build facts for insights
    facts = []
    for name, table_info in whatif_tables.items():
        facts.append(table_info["df"].to_dict(orient='records'))

    # Render layout
    viz, insights, final_prompt, export_data = render_whatif_layout(
        whatif_tables,
        claims_env.ba.title,
        f"What-If: {'+' if impact_pct >= 0 else ''}{impact_pct}% Claims Expense Change",
        facts,
        impact_pct,
        parameters.arguments.max_prompt,
        parameters.arguments.insight_prompt,
        parameters.arguments.table_viz_layout
    )

    return SkillOutput(
        final_prompt=final_prompt,
        narrative=insights,
        visualizations=viz,
        parameter_display_descriptions=param_info,
        followup_questions=[],
        export_data=[ExportData(name=name, data=df) for name, df in export_data.items()]
    )


def build_whatif_tables(claims_tables, profit_tables, impact_pct, breakouts):
    """Build combined what-if tables showing claims and profit impact."""

    whatif_tables = {}
    impact_multiplier = impact_pct / 100.0

    # Get the first breakout dimension's data from each
    claims_key = list(claims_tables.keys())[0] if claims_tables else None
    profit_key = list(profit_tables.keys())[0] if profit_tables else None

    if not claims_key or not profit_key:
        return whatif_tables

    claims_df = claims_tables[claims_key]["df"].copy()
    profit_df = profit_tables[profit_key]["df"].copy()

    print(f"Claims DF columns: {claims_df.columns.tolist()}")
    print(f"Profit DF columns: {profit_df.columns.tolist()}")

    # Get the dimension column (first column)
    dim_col = claims_df.columns[0]

    # Find the current and previous columns dynamically
    claims_curr_col = [c for c in claims_df.columns if '(Current)' in c][0]
    claims_prev_col = [c for c in claims_df.columns if '(Previous)' in c][0]
    profit_curr_col = [c for c in profit_df.columns if '(Current)' in c][0]
    profit_prev_col = [c for c in profit_df.columns if '(Previous)' in c][0]

    print(f"Claims cols: {claims_curr_col}, {claims_prev_col}")
    print(f"Profit cols: {profit_curr_col}, {profit_prev_col}")

    # Merge on dimension
    merged_df = claims_df[[dim_col, claims_curr_col, claims_prev_col]].merge(
        profit_df[[dim_col, profit_curr_col, profit_prev_col]],
        on=dim_col,
        how='outer'
    )

    # Calculate what-if projections
    # Parse numeric values from formatted strings
    def parse_value(val):
        if pd.isna(val) or val == '':
            return 0
        if isinstance(val, (int, float)):
            return val
        # Remove $, M, B, K suffixes and parse
        val_str = str(val).replace('$', '').replace(',', '').strip()
        multiplier = 1
        if val_str.endswith('B'):
            multiplier = 1_000_000_000
            val_str = val_str[:-1]
        elif val_str.endswith('M'):
            multiplier = 1_000_000
            val_str = val_str[:-1]
        elif val_str.endswith('K'):
            multiplier = 1_000
            val_str = val_str[:-1]
        try:
            return float(val_str) * multiplier
        except:
            return 0

    # Calculate projected values
    results = []
    for _, row in merged_df.iterrows():
        claims_prev = parse_value(row[claims_prev_col])
        claims_curr = parse_value(row[claims_curr_col])
        profit_prev = parse_value(row[profit_prev_col])
        profit_curr = parse_value(row[profit_curr_col])

        # What-if: claims increase by impact_pct from current
        claims_projected = claims_curr * (1 + impact_multiplier)
        claims_impact = claims_projected - claims_curr

        # Profit impact: direct inverse (claims up = profit down)
        profit_projected = profit_curr - claims_impact
        profit_impact = -claims_impact
        profit_impact_pct = (profit_impact / profit_curr * 100) if profit_curr != 0 else 0

        results.append({
            'Country': row[dim_col],
            'Claims (Current)': f"${genpact_format_number(claims_curr)}",
            'Claims (Projected)': f"${genpact_format_number(claims_projected)}",
            'Claims Impact': f"${genpact_format_number(claims_impact)}",
            'Profit (Current)': f"${genpact_format_number(profit_curr)}",
            'Profit (Projected)': f"${genpact_format_number(profit_projected)}",
            'Profit Impact': f"${genpact_format_number(profit_impact)}",
            'Profit Impact %': f"{profit_impact_pct:+.1f}%"
        })

    result_df = pd.DataFrame(results)

    whatif_tables["What-If Scenario"] = {
        "df": result_df,
        "chart_vars": {}
    }

    return whatif_tables


def render_whatif_layout(tables, title, subtitle, facts, impact_pct, max_prompt, insight_prompt, viz_layout):
    """Render the what-if analysis layout."""

    # Generate insights
    insight_template = jinja2.Template(insight_prompt).render(facts=facts, impact_pct=impact_pct)
    max_response_prompt = jinja2.Template(max_prompt).render(facts=facts, impact_pct=impact_pct)

    ar_utils = ArUtils()
    insights = ar_utils.get_llm_response(insight_template)

    viz_list = []
    export_data = {}

    general_vars = {
        "headline": smart_title(title) if title else "What-If Analysis",
        "sub_headline": subtitle,
        "hide_growth_warning": True,
        "exec_summary": insights if insights else "No Insights.",
        "warning": None
    }

    viz_layout_parsed = json.loads(viz_layout)

    for name, table_info in tables.items():
        table_df = table_info["df"]
        export_data[name] = table_df

        table_vars = get_table_layout_vars(table_df)
        table_vars["hide_footer"] = True
        table_vars["hide_chart"] = True

        rendered = wire_layout(viz_layout_parsed, {**general_vars, **table_vars})
        viz_list.append(SkillVisualization(title=name, layout=rendered))

    return viz_list, insights, max_response_prompt, export_data


if __name__ == '__main__':
    skill_input: SkillInput = whatif_analysis.create_input(arguments={
        'impact_pct': 25,
        'breakouts': ["country"],
        'periods': ["q2 2025"],
        'other_filters': [{"dim": "region", "op": "=", "val": ["Europe"]}]
    })
    out = whatif_analysis(skill_input)
    preview_skill(whatif_analysis, out)
