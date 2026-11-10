# Report generation for enterprise-rag-evaluation-suite

from typing import Dict, Any, List
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

def generate_report(eval_results: Dict[str, Dict[str, float]], settings, output_dir: Path) -> Dict[str, Path]:
    """Generate evaluation reports in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_paths = {}
    
    # JSON report
    if "json" in settings.output.formats:
        json_path = output_dir / "evaluation_report.json"
        with open(json_path, "w") as f:
            json.dump({
                "settings": {
                    "dataset": settings.dataset.name,
                    "retriever": settings.retrieval.embedder,
                    "reranker": settings.reranking.model,
                    "generator": settings.generation.model,
                    "sample_size": settings.evaluation.sample_size,
                    "metrics": settings.evaluation.metrics,
                },
                "results": eval_results,
            }, f, indent=2)
        report_paths["json"] = json_path
        logger.info(f"JSON report saved to {json_path}")
    
    # HTML report
    if "html" in settings.output.formats:
        html_path = output_dir / "evaluation_report.html"
        generate_html_report(eval_results, settings, html_path)
        report_paths["html"] = html_path
        logger.info(f"HTML report saved to {html_path}")
    
    # Markdown report
    if "markdown" in settings.output.formats:
        md_path = output_dir / "evaluation_report.md"
        generate_markdown_report(eval_results, settings, md_path)
        report_paths["markdown"] = md_path
        logger.info(f"Markdown report saved to {md_path}")
    
    return report_paths

def generate_html_report(eval_results: Dict[str, Dict[str, float]], settings, output_path: Path):
    """Generate interactive HTML report with charts."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Evaluation Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #1a1a2e; border-bottom: 3px solid #4361ee; padding-bottom: 10px; }}
        h2 {{ color: #16213e; margin-top: 40px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; border-radius: 8px; padding: 20px; border-left: 4px solid #4361ee; }}
        .metric-card h3 {{ margin: 0 0 10px 0; color: #4361ee; font-size: 14px; text-transform: uppercase; }}
        .metric-value {{ font-size: 32px; font-weight: bold; color: #1a1a2e; }}
        .chart-container {{ margin: 30px 0; height: 400px; }}
        .config-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .config-table th, .config-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
        .config-table th {{ background: #f0f2f5; font-weight: 600; }}
        .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }}
        .badge-primary {{ background: #e0e7ff; color: #3730a3; }}
        .badge-success {{ background: #dcfce7; color: #166534; }}
        .badge-warning {{ background: #fef3c7; color: #92400e; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 RAG Evaluation Report</h1>
        
        <h2>Configuration</h2>
        <table class="config-table">
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Dataset</td><td><span class="badge badge-primary">{settings.dataset.name}</span></td></tr>
            <tr><td>Retriever</td><td><span class="badge badge-success">{settings.retrieval.embedder}</span></td></tr>
            <tr><td>Reranker</td><td><span class="badge badge-success">{settings.reranking.model}</span></td></tr>
            <tr><td>Generator</td><td><span class="badge badge-success">{settings.generation.model}</span></td></tr>
            <tr><td>Sample Size</td><td>{settings.evaluation.sample_size}</td></tr>
            <tr><td>Metrics</td><td>{', '.join(settings.evaluation.metrics)}</td></tr>
        </table>
        
        <h2>Summary Metrics</h2>
        <div class="metrics-grid" id="metrics-grid">
        </div>
        
        <h2>Detailed Results</h2>
        <div id="charts">
        </div>
    </div>
    
    <script>
        const results = {json.dumps(eval_results)};
        const metricsGrid = document.getElementById('metrics-grid');
        const chartsDiv = document.getElementById('charts');
        
        // Create metric cards
        Object.entries(results).forEach(([category, metrics]) => {{
            Object.entries(metrics).forEach(([metric, value]) => {{
                const card = document.createElement('div');
                card.className = 'metric-card';
                card.innerHTML = `
                    <h3>${{category}}.${{metric}}</h3>
                    <div class="metric-value">${{typeof value === 'number' ? value.toFixed(4) : value}}</div>
                `;
                metricsGrid.appendChild(card);
            }});
        }});
        
        // Create charts for each category
        Object.entries(results).forEach(([category, metrics]) => {{
            const chartDiv = document.createElement('div');
            chartDiv.className = 'chart-container';
            chartDiv.id = `chart-${{category}}`;
            chartsDiv.appendChild(chartDiv);
            
            const names = Object.keys(metrics);
            const values = Object.values(metrics).map(v => typeof v === 'number' ? v : 0);
            
            Plotly.newPlot(chartDiv.id, [{
                x: names,
                y: values,
                type: 'bar',
                marker: {{ color: '#4361ee' }},
                text: values.map(v => v.toFixed(4)),
                textposition: 'auto',
            }], {{
                title: `${{category.charAt(0).toUpperCase() + category.slice(1)}} Metrics`,
                xaxis: {{ title: 'Metric' }},
                yaxis: {{ title: 'Score', range: [0, 1] }},
                margin: {{ t: 50, b: 80 }},
            }});
        }});
    </script>
</body>
</html>"""
    
    with open(output_path, "w") as f:
        f.write(html)

def generate_markdown_report(eval_results: Dict[str, Dict[str, float]], settings, output_path: Path):
    """Generate Markdown report."""
    md = f"""# RAG Evaluation Report

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | {settings.dataset.name} |
| Retriever | {settings.retrieval.embedder} |
| Reranker | {settings.reranking.model} |
| Generator | {settings.generation.model} |
| Sample Size | {settings.evaluation.sample_size} |
| Metrics | {', '.join(settings.evaluation.metrics)} |

## Results

"""
    
    for category, metrics in eval_results.items():
        md += f"### {category.title()}\n\n"
        md += "| Metric | Value |\n|--------|-------|\n"
        for metric, value in metrics.items():
            if isinstance(value, float):
                md += f"| {metric} | {value:.4f} |\n"
            else:
                md += f"| {metric} | {value} |\n"
        md += "\n"
    
    md += """---
*Generated by Enterprise RAG Evaluation Suite*
"""
    
    with open(output_path, "w") as f:
        f.write(md)