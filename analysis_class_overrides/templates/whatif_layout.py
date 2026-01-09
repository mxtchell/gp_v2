whatif_layout = """
{
	"layoutJson": {
		"type": "Document",
		"rows": 90,
		"columns": 160,
		"rowHeight": "1.11%",
		"colWidth": "0.625%",
		"gap": "0px",
		"style": {
			"backgroundColor": "#ffffff",
			"width": "100%",
			"height": "max-content",
			"padding": "15px",
			"gap": "20px"
		},
		"children": [
			{
				"name": "CardContainer0",
				"type": "CardContainer",
				"children": "",
				"minHeight": "80px",
				"rows": 2,
				"columns": 1,
				"style": {
					"border-radius": "11.911px",
					"background": "#2563EB",
					"padding": "10px",
					"fontFamily": "Arial"
				},
				"hidden": false
			},
			{
				"name": "Header0",
				"type": "Header",
				"children": "",
				"text": "What-If Analysis",
				"style": {
					"fontSize": "20px",
					"fontWeight": "700",
					"color": "#ffffff",
					"textAlign": "left",
					"alignItems": "center"
				},
				"parentId": "CardContainer0",
				"hidden": false
			},
			{
				"name": "Paragraph0",
				"type": "Paragraph",
				"children": "",
				"text": "What-If Scenario",
				"style": {
					"fontSize": "15px",
					"fontWeight": "normal",
					"textAlign": "center",
					"verticalAlign": "start",
					"color": "#fafafa",
					"border": "null",
					"textDecoration": "null",
					"writingMode": "horizontal-tb",
					"alignItems": "center"
				},
				"parentId": "CardContainer0",
				"hidden": false
			},
			{
				"name": "FlexContainer5",
				"type": "FlexContainer",
				"minHeight": "300px",
				"direction": "row",
				"style": {
					"maxWidth": "90%",
					"width": "90%"
				}
			},
			{
				"name": "FlexContainer4",
				"type": "FlexContainer",
				"children": "",
				"minHeight": "250px",
				"direction": "column",
				"maxHeight": "1200px"
			},
			{
				"name": "DataTable0",
				"type": "DataTable",
				"children": "",
				"columns": [
					{
						"name": "Column 1"
					},
					{
						"name": "Column 2"
					},
					{
						"name": "Column 3"
					},
					{
						"name": "Column 4"
					}
				],
				"data": [
					[
						"Row 1",
						0,
						0,
						0
					]
				],
				"parentId": "FlexContainer4",
				"caption": "",
				"styles": {
					"td": {
						"vertical-align": "middle"
					}
				}
			},
			{
				"name": "HighchartsChart0",
				"type": "HighchartsChart",
				"minHeight": "400px",
				"chartOptions": {
					"chart": {
						"type": "column"
					},
					"title": {
						"text": ""
					},
					"xAxis": {
						"categories": []
					},
					"yAxis": {
						"title": {
							"text": ""
						}
					},
					"series": []
				},
				"options": {
					"chart": {
						"type": "column",
						"polar": false
					},
					"title": {
						"text": "",
						"style": {
							"fontSize": "20px"
						}
					},
					"xAxis": {
						"categories": [],
						"title": {
							"text": ""
						}
					},
					"yAxis": {
						"title": {
							"text": ""
						}
					},
					"series": [],
					"credits": {
						"enabled": false
					},
					"legend": {
						"enabled": true,
						"align": "center",
						"verticalAlign": "bottom",
						"layout": "horizontal"
					},
					"plotOptions": {
						"column": {
							"dataLabels": {
								"style": {
									"fontSize": ""
								},
								"enabled": false
							}
						}
					}
				},
				"parentId": "FlexContainer5",
				"hidden": false
			}
		]
	},
	"inputVariables": [
		{
			"name": "sub_headline",
			"isRequired": false,
			"defaultValue": null,
			"targets": [
				{
					"elementName": "Paragraph0",
					"fieldName": "text"
				}
			]
		},
		{
			"name": "headline",
			"isRequired": false,
			"defaultValue": null,
			"targets": [
				{
					"elementName": "Header0",
					"fieldName": "text"
				}
			]
		},
		{
			"name": "data",
			"isRequired": false,
			"defaultValue": null,
			"targets": [
				{
					"elementName": "DataTable0",
					"fieldName": "data"
				}
			]
		},
		{
			"name": "col_defs",
			"isRequired": false,
			"defaultValue": null,
			"targets": [
				{
					"elementName": "DataTable0",
					"fieldName": "columns"
				}
			]
		},
		{
			"name": "chart_title",
			"isRequired": false,
			"defaultValue": null,
			"targets": [
				{
					"elementName": "HighchartsChart0",
					"fieldName": "options.title.text"
				}
			]
		},
		{
			"name": "chart_categories",
			"isRequired": false,
			"defaultValue": null,
			"targets": [
				{
					"elementName": "HighchartsChart0",
					"fieldName": "options.xAxis.categories"
				}
			]
		},
		{
			"name": "chart_y_axis",
			"isRequired": false,
			"defaultValue": null,
			"targets": [
				{
					"elementName": "HighchartsChart0",
					"fieldName": "options.yAxis"
				}
			]
		},
		{
			"name": "chart_data",
			"isRequired": false,
			"defaultValue": null,
			"targets": [
				{
					"elementName": "HighchartsChart0",
					"fieldName": "options.series"
				}
			]
		},
		{
			"name": "hide_chart",
			"isRequired": false,
			"defaultValue": null,
			"targets": [
				{
					"elementName": "FlexContainer5",
					"fieldName": "hidden"
				}
			]
		}
	]
}
"""
