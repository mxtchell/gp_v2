from typing import Dict
from ar_analytics import ArUtils
from ar_analytics.driver_analysis import DriverAnalysis
from ar_analytics.metric_tree import MetricTreeAnalysis
from ar_analytics.breakout_drivers import BreakoutDrivers
from ar_analytics.helpers.utils import Connector, fmt_sign_num
import pandas as pd
from analysis_class_overrides.insurance_utilities import InsuranceSharedFn

class InsuranceDriverAnalysis(DriverAnalysis):
    def __init__(self, dim_hierarchy, dim_val_map={}, sql_exec:Connector=None, constrained_values={}, compare_date_warning_msg=None, df_provider=None, sp=None):
        self.mta = MetricTreeAnalysis(sql_exec, df_provider=df_provider, sp=sp)
        self.ba = BreakoutDrivers(dim_hierarchy, dim_val_map, sql_exec, df_provider=df_provider, sp=sp)
        self.helper = InsuranceSharedFn()
        self.allowed_metrics = constrained_values.get("metric", [])
        self.alloed_breakouts = constrained_values.get("breakout", [])
        self.notes = []
        self.compare_date_warning_msg = compare_date_warning_msg
        self.sp=sp
        self.ar_utils = ArUtils()

    def _create_breakout_chart_vars(self, raw_b_df: pd.DataFrame, dim: str, rename_dict: Dict[str, str]):
        from genpact_formatting import genpact_format_number
        import logging
        import math
        logger = logging.getLogger(__name__)
        
        logger.info(f"DEBUG** Starting metric drivers chart creation")
        logger.info(f"DEBUG** DataFrame shape: {raw_b_df.shape}")
        logger.info(f"DEBUG** DataFrame columns: {raw_b_df.columns.tolist()}")
        
        categories = raw_b_df[dim].tolist()
        logger.info(f"DEBUG** Categories: {categories}")

        # Get metric format to determine formatting approach
        formatter = self.ar_utils.python_to_highcharts_format(self.ba.target_metric["fmt"])
        metric_format = self.ba.target_metric.get("fmt", "")
        is_percentage = "%" in metric_format
        is_currency = "$" in metric_format
        
        logger.info(f"DEBUG** Metric format: {metric_format}, is_percentage: {is_percentage}, is_currency: {is_currency}")

        # Process data for both curr and prev with proper formatting
        curr_values = raw_b_df["curr"].tolist()
        prev_values = raw_b_df["prev"].tolist()
        
        logger.info(f"DEBUG** Curr values: {curr_values}")
        logger.info(f"DEBUG** Prev values: {prev_values}")
        
        # Prepare formatted data like dimension breakout
        curr_data = []
        prev_data = []
        
        for i, (category, curr_val, prev_val) in enumerate(zip(categories, curr_values, prev_values)):
            # Process current values
            if pd.isna(curr_val):
                curr_formatted = "N/A"
                curr_y = 0
            else:
                if is_percentage:
                    if isinstance(curr_val, str) and "%" in curr_val:
                        curr_formatted = curr_val
                        curr_y = float(curr_val.replace("%", ""))
                    elif isinstance(curr_val, (int, float)):
                        percentage_value = curr_val * 100
                        curr_formatted = f"{percentage_value:.2f}%"
                        curr_y = percentage_value
                    else:
                        curr_formatted = str(curr_val)
                        curr_y = 0
                else:
                    try:
                        if isinstance(curr_val, str):
                            curr_numeric = float(curr_val)
                        else:
                            curr_numeric = curr_val
                        
                        if is_currency:
                            curr_formatted = f"${genpact_format_number(curr_numeric)}"
                        else:
                            curr_formatted = genpact_format_number(curr_numeric)
                        curr_y = curr_numeric
                    except (ValueError, TypeError):
                        curr_formatted = str(curr_val)
                        curr_y = 0
            
            curr_data.append({
                "name": category,
                "y": curr_y,
                "formatted": curr_formatted
            })
            
            # Process previous values
            if pd.isna(prev_val):
                prev_formatted = "N/A"
                prev_y = 0
            else:
                if is_percentage:
                    if isinstance(prev_val, str) and "%" in prev_val:
                        prev_formatted = prev_val
                        prev_y = float(prev_val.replace("%", ""))
                    elif isinstance(prev_val, (int, float)):
                        percentage_value = prev_val * 100
                        prev_formatted = f"{percentage_value:.2f}%"
                        prev_y = percentage_value
                    else:
                        prev_formatted = str(prev_val)
                        prev_y = 0
                else:
                    try:
                        if isinstance(prev_val, str):
                            prev_numeric = float(prev_val)
                        else:
                            prev_numeric = prev_val
                        
                        if is_currency:
                            prev_formatted = f"${genpact_format_number(prev_numeric)}"
                        else:
                            prev_formatted = genpact_format_number(prev_numeric)
                        prev_y = prev_numeric
                    except (ValueError, TypeError):
                        prev_formatted = str(prev_val)
                        prev_y = 0
            
            prev_data.append({
                "name": category,
                "y": prev_y,
                "formatted": prev_formatted
            })

        # Create Y-axis with M/K/B formatting like dimension breakout
        all_values = [item.get('y', 0) for item in curr_data + prev_data if isinstance(item.get('y'), (int, float))]
        max_value = max(all_values) if all_values else 0
        min_value = min(all_values) if all_values else 0
        
        logger.info(f"DEBUG** Y-axis range: {min_value} to {max_value}")
        
        if is_percentage:
            y_axis = [{"title": "", "labels": {"format": "{value:.1f}%"}}]
        elif is_currency and max_value >= 1000:
            # Scale data for better display like dimension breakout
            import math
            scaled_max = max_value / 1000000  # Convert to millions
            
            if scaled_max <= 500:
                y_axis = [{
                    "title": "",
                    "min": 0,
                    "max": math.ceil(scaled_max / 100) * 100,
                    "tickInterval": 100,
                    "labels": {"format": "${value}M"}
                }]
            else:
                y_axis = [{
                    "title": "",
                    "min": 0,
                    "max": math.ceil(scaled_max / 200) * 200, 
                    "tickInterval": 200,
                    "labels": {"format": "${value}M"}
                }]
            
            # Scale the data to match the axis
            for item in curr_data:
                if isinstance(item.get('y'), (int, float)):
                    item['y'] = item['y'] / 1000000
            
            for item in prev_data:
                if isinstance(item.get('y'), (int, float)):
                    item['y'] = item['y'] / 1000000
            
            logger.info(f"DEBUG** Scaled curr data (first 3): {curr_data[:3]}")
            logger.info(f"DEBUG** Scaled prev data (first 3): {prev_data[:3]}")
        else:
            y_axis = [{
                "title": "",
                "labels": {"format": formatter.get('value_format')}
            }]

        # Simple two-color scheme: light blue for current, light orange for previous
        current_color = "#5DADE2"   # Light blue for current period
        previous_color = "#F8C471"  # Light orange for comparison period

        data = []

        # Current series with single light blue color - add metric name
        metric_name = self.ba.target_metric.get("label", self.ba.target_metric.get("name", "Metric"))
        data.append({
            "name": f"{metric_name} (Current)",
            "data": curr_data,
            "color": current_color,  # Single color for all bars
            "dataLabels": {
                "enabled": False
            },
            "tooltip": {
                "pointFormat": "<b>{series.name}</b>: {point.formatted}"
            }
        })

        # Previous series with single light orange color - add metric name
        data.append({
            "name": f"{metric_name} (Previous)",
            "data": prev_data,
            "color": previous_color,  # Single color for all bars
            "dataLabels": {
                "enabled": False
            },
            "tooltip": {
                "pointFormat": "<b>{series.name}</b>: {point.formatted}"
            }
        })
        
        logger.info(f"DEBUG** Created {len(data)} series with vibrant colors")

        return {
            "chart_categories": categories,
            "chart_y_axis": y_axis,
            "chart_title": "",
            "chart_data": data
        }

    def _format_metric_column(self, row, col):
        """Format metric column, with special handling for bps growth metrics."""
        metric_props = self.helper.get_metric_prop(row.name, self.metric_props)

        if col == "growth":
            growth_fmt = metric_props.get("growth_fmt", "")
            # Check if this is a bps metric - use diff value instead of growth
            if "bps" in growth_fmt.lower():
                # diff is already in percentage points, multiply by 100 to get bps
                diff_val = row.get("diff") if hasattr(row, "get") else row["diff"]
                if pd.notna(diff_val) and isinstance(diff_val, (int, float)):
                    bps_value = diff_val * 100
                    return f"{bps_value:.0f} bps"
                return ""
            return self.helper.get_formatted_num(row[col], growth_fmt)
        else:
            fmt = metric_props.get("fmt", "")
            return self.helper.get_formatted_num(row[col], fmt)

    def get_display_tables(self):
        metric_df = self._metric_df.copy()
        breakout_df = self._breakout_df.copy()
        breakout_chart_df = self._breakout_df.copy()

        # Define required columns for metric_df
        metric_tree_required_columns = ["curr", "prev", "diff", "growth"]
        if self.include_sparklines:
            metric_tree_required_columns.append("sparkline")

        if "impact" in metric_df.columns:
            metric_tree_required_columns.append("impact")

        # Filter metric_df to include only the required columns
        metric_df = metric_df[metric_tree_required_columns]

        # Apply formatting for metric_df
        for col in ["curr", "prev", "diff", "growth"]:
            metric_df[col] = metric_df.apply(
                lambda row, c=col: self._format_metric_column(row, c),
                axis=1
            )

        if "impact" in metric_df.columns:
            metric_df["impact"] = metric_df.apply(
                lambda row: self.helper.get_formatted_num(row["impact"], self.mta.impact_format), axis=1
            )

        # rename columns
        metric_df = metric_df.rename(
            columns={'curr': 'Value', 'prev': 'Prev Value', 'diff': 'Change', 'growth': '% Growth', 'sparkline': 'SPARKLINE (L24M)'})
        
        metric_df = metric_df.reset_index()

        # rename index to metric labels
        metric_df["index"] = metric_df["index"].apply(lambda x: self.helper.get_metric_prop(x, self.metric_props).get("label", x))

        # indent non target metric
        metric_df["index"] = metric_df["index"].apply(lambda x: f"  {x}" if x != self.mta.target_metric else x)

        metric_df = metric_df.rename(columns={"index": ""})

        # Define required columns for breakout_df
        breakout_required_columns = ["curr", "prev", "diff", "diff_pct", "rank_change"]
        if self.include_sparklines:
            breakout_required_columns.append("sparkline")

        breakout_dfs = {}
        breakout_chart_vars = {}

        # Apply formatting for breakout_df
        growth_fmt = self.ba.target_metric.get("growth_fmt", "")
        is_bps_metric = "bps" in growth_fmt.lower()

        for col in ["curr", "prev", "diff", "diff_pct"]:
            if col == "diff_pct" and is_bps_metric:
                # For bps metrics, use diff value converted to bps
                breakout_df[col] = breakout_df.apply(
                    lambda row: f"{row['diff'] * 100:.0f} bps" if pd.notna(row['diff']) and isinstance(row['diff'], (int, float)) else "",
                    axis=1
                )
            else:
                breakout_df[col] = breakout_df.apply(
                    lambda row: self.helper.get_formatted_num(row[col],
                                                              self.ba.target_metric["fmt"] if col != "diff_pct" else
                                                              self.ba.target_metric["growth_fmt"]),
                    axis=1
                )

        # Format rank column
        breakout_df["rank_curr"] = breakout_df["rank_curr"]
        breakout_df["rank_change"] = breakout_df.apply(lambda row: f"{int(row['rank_curr'])} ({fmt_sign_num(row['rank_change'])})"
                                                    if (row['rank_change'] and pd.notna(row['rank_change']) and row['rank_change'] != 0)
                                                    else row['rank_curr'], axis=1)
        breakout_df = breakout_df.reset_index()
        breakout_chart_df = breakout_chart_df.reset_index()

        breakout_dims = list(breakout_df["dim"].unique())
        if self.ba.dim_hier:
            # display according to the dim hierarchy ordering
            ordering_dict = {value: index for index, value in enumerate(self.ba.dim_hier.get_hierarchy_ordering())}
            # rename cols to dim labels
            ordering_dict = {self.helper.get_dimension_prop(k, self.dim_props).get("label", k): v for k, v in ordering_dict.items()}
            # sort dims by hierarchy order
            breakout_dims.sort(key=lambda x: (ordering_dict.get(x, len(ordering_dict)), x))

        comp_dim = None
        if self.ba._owner_dim:
            comp_dim = next((d for d in breakout_dims if d.lower() == self.ba._owner_dim.lower()), None)

        if comp_dim:
            breakout_dims = [comp_dim] + [x for x in breakout_dims if x != comp_dim]

        for dim in breakout_dims:
            b_df = breakout_df[breakout_df["dim"] == dim]
            raw_b_df = breakout_chart_df[breakout_chart_df["dim"] == dim]
            if str(dim).lower() == str(comp_dim).lower():
                viz_name = "Benchmark"
            else:
                viz_name = dim
            raw_b_df = raw_b_df.rename(columns={'dim_value': dim})
            b_df = b_df.rename(columns={'dim_value': dim})
            b_df = b_df[[dim] + breakout_required_columns]

            rename_dict = {'curr': 'Value', 'prev': 'Prev Value', 'diff': 'Change', 'diff_pct': '% Growth',
                         'rank_change': 'Rank Change', 'sparkline': 'SPARKLINE (L24M)'}

            # rename columns
            b_df = b_df.rename(
                columns=rename_dict)
            breakout_dfs[viz_name] = {
                "df": b_df,
                "chart_vars": self._create_breakout_chart_vars(raw_b_df, dim, rename_dict)
            }

        return {"viz_metric_df": metric_df, "viz_breakout_dfs": breakout_dfs}

class InsuranceMetricTreeAnalysis(MetricTreeAnalysis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = InsuranceSharedFn()

class InsuranceBreakoutDrivers(BreakoutDrivers):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = InsuranceSharedFn()