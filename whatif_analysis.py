from __future__ import annotations

import json
import logging

import pandas as pd
import jinja2
from ar_analytics import ArUtils
from ar_analytics.defaults import get_table_layout_vars
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
    description="Analyzes what-if scenarios for claims expense impact on operating profit. Models how a percentage change in claims expense would affect profitability by country or segment.",
    capabilities="Model impact of claims expense changes on operating profit. Show projected values and changes by dimension. Generate insights about scenario impact.",
    limitations="Fixed to claims_expense and operating_profit metrics only. Assumes direct 1:1 relationship between claims increase and profit decrease.",
    example_questions="What if claims expense increases by 25% in Europe? How would a 10% rise in claims affect UK profitability? Model 15% claims increase impact on P&C by country.",
    parameter_guidance="Always specify impact_pct (the percentage change to model). Use positive values for increases, negative for decreases. Use filters for region, business segment, time period. Breakout is fixed to country.",
    parameters=[
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
    """What-If Analysis - model claims expense impact on operating profit"""

    # Extract parameters
    impact_pct = getattr(parameters.arguments, 'impact_pct', 10) or 10
    periods = getattr(parameters.arguments, 'periods', []) or []
    limit_n = getattr(parameters.arguments, 'limit_n', 10) or 10
    other_filters = getattr(parameters.arguments, 'other_filters', []) or []

    # Convert impact_pct to float if string
    if isinstance(impact_pct, str):
        impact_pct = float(impact_pct)

    print(f"What-If Analysis - Impact: {impact_pct}%, Periods: {periods}, Filters: {other_filters}")

    # Build filter clause for SQL
    filter_clause = ""
    if other_filters:
        filter_conditions = []
        for f in other_filters:
            if isinstance(f, str) and ":" in f:
                dim, val = f.split(":", 1)
                filter_conditions.append(f"LOWER({dim}) = LOWER('{val}')")
            elif isinstance(f, dict):
                dim = f.get("dim") or f.get("dimension")
                val = f.get("val") or f.get("value")
                if dim and val:
                    if isinstance(val, list):
                        val_list = ", ".join([f"'{v}'" for v in val])
                        filter_conditions.append(f"LOWER({dim}) IN ({val_list.lower()})")
                    else:
                        filter_conditions.append(f"LOWER({dim}) = LOWER('{val}')")
        if filter_conditions:
            filter_clause = " AND " + " AND ".join(filter_conditions)

    # Use ArUtils for SQL execution
    ar_utils = ArUtils()

    # Query claims_expense and operating_profit by country
    sql = f"""
    SELECT
        country,
        SUM(claims_expense) as claims_expense,
        SUM(operating_profit) as operating_profit
    FROM source_data
    WHERE 1=1 {filter_clause}
    GROUP BY country
    ORDER BY claims_expense DESC
    LIMIT {limit_n}
    """

    print(f"Executing SQL: {sql}")

    try:
        result_df = ar_utils.execute_sql(sql)
    except Exception as e:
        print(f"SQL execution error: {e}")
        # Return error output
        return SkillOutput(
            final_prompt=f"Error executing what-if analysis: {str(e)}",
            narrative=None,
            visualizations=[],
            parameter_display_descriptions=[],
            followup_questions=[],
            export_data=[]
        )

    if result_df is None or result_df.empty:
        return SkillOutput(
            final_prompt="No data found for the specified filters.",
            narrative=None,
            visualizations=[],
            parameter_display_descriptions=[],
            followup_questions=[],
            export_data=[]
        )

    # Calculate what-if projections
    impact_multiplier = impact_pct / 100.0
    whatif_df = calculate_whatif_impact(result_df, impact_multiplier)

    # Build parameter display info
    param_info = [
        ParameterDisplayDescription(key="Scenario", value=f"{'+' if impact_pct >= 0 else ''}{impact_pct}% Claims Change"),
        ParameterDisplayDescription(key="Metrics", value="Claims Expense, Operating Profit"),
        ParameterDisplayDescription(key="Breakout", value="Country"),
    ]

    # Build facts for insights
    facts = whatif_df.to_dict(orient='records')

    # Determine title from filters
    title = "What-If Analysis"
    if other_filters:
        for f in other_filters:
            if isinstance(f, str) and ":" in f:
                dim, val = f.split(":", 1)
                title = f"{val.title()} What-If Analysis"
                break

    # Render layout
    viz, insights, final_prompt, export_data = render_whatif_layout(
        {"What-If Scenario": {"df": whatif_df, "chart_vars": {}}},
        title,
        f"What-If: {'+' if impact_pct >= 0 else ''}{int(impact_pct)}% Claims Expense Change",
        [facts],
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


def calculate_whatif_impact(df, impact_multiplier):
    """Calculate what-if impact on claims and profit."""
    results = []

    for _, row in df.iterrows():
        claims_current = row['claims_expense']
        profit_current = row['operating_profit']

        # What-if: claims increase by impact percentage
        claims_projected = claims_current * (1 + impact_multiplier)
        claims_impact = claims_projected - claims_current

        # Profit impact: direct inverse (claims up = profit down)
        profit_projected = profit_current - claims_impact
        profit_impact = -claims_impact
        profit_impact_pct = (profit_impact / profit_current * 100) if profit_current != 0 else 0

        results.append({
            'Country': row['country'],
            'Claims (Current)': f"${genpact_format_number(claims_current)}",
            'Claims (Projected)': f"${genpact_format_number(claims_projected)}",
            'Claims Impact': f"${genpact_format_number(claims_impact)}",
            'Profit (Current)': f"${genpact_format_number(profit_current)}",
            'Profit (Projected)': f"${genpact_format_number(profit_projected)}",
            'Profit Impact': f"${genpact_format_number(profit_impact)}",
            'Profit Impact %': f"{profit_impact_pct:+.1f}%"
        })

    return pd.DataFrame(results)


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
        'periods': ["q2 2025"],
        'other_filters': ["region:europe"]
    })
    out = whatif_analysis(skill_input)
    preview_skill(whatif_analysis, out)
