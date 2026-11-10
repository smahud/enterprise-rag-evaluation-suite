# CLI entry point for enterprise-rag-evaluation-suite

import click
import asyncio
import logging
import sys
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import get_settings
from .datasets import load_dataset
from .retrieval import get_retriever
from .reranking import get_reranker
from .generation import get_generator
from .evaluation import evaluate_retrieval, evaluate_reranking, evaluate_generation, evaluate_hallucination
from .reporting import generate_report

console = Console()

def setup_logging(level: str, format: str):
    """Configure logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    if format == "json":
        from pythonjsonlogger import jsonlogger
        handler = logging.StreamHandler()
        handler.setFormatter(jsonlogger.JsonFormatter())
        logging.root.addHandler(handler)
    else:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    logging.root.setLevel(log_level)

@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
@click.option("--log-level", default="INFO", help="Log level")
@click.option("--log-format", default="console", type=click.Choice(["console", "json"]), help="Log format")
@click.pass_context
def cli(ctx, config, log_level, log_format):
    """RAG Evaluation Suite - Benchmark semantic search, reranking, and generation."""
    setup_logging(log_level, log_format)
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config

@cli.command()
@click.option("--dataset", "-d", default="msmarco", help="Dataset name")
@click.option("--retriever", "-r", default="bge-large-en-v1.5", help="Retriever model")
@click.option("--reranker", "-k", default="bge-reranker-large", help="Reranker model")
@click.option("--generator", "-g", default="gpt-4o-mini", help="Generator model")
@click.option("--sample-size", "-s", default=1000, help="Number of samples")
@click.option("--output", "-o", default="./results", help="Output directory")
@click.option("--format", "-f", multiple=True, default=["json", "html"], type=click.Choice(["json", "html", "markdown"]), help="Output formats")
@click.option("--metrics", "-m", multiple=True, default=["recall", "ndcg", "mrr", "faithfulness", "hallucination_rate"], help="Metrics to compute")
@click.pass_context
def run(ctx, dataset, retriever, reranker, generator, sample_size, output, format, metrics):
    """Run full RAG evaluation pipeline."""
    settings = get_settings()
    
    # Override settings from CLI
    settings.dataset.name = dataset
    settings.dataset.max_samples = sample_size
    settings.retrieval.embedder = retriever
    settings.reranking.model = reranker
    settings.generation.model = generator
    settings.evaluation.metrics = list(metrics)
    settings.evaluation.sample_size = sample_size
    settings.output.directory = output
    settings.output.formats = list(format)
    
    console.print(f"[bold]Starting RAG Evaluation[/bold]")
    console.print(f"Dataset: {dataset}, Retriever: {retriever}, Reranker: {reranker}, Generator: {generator}")
    console.print(f"Samples: {sample_size}, Metrics: {', '.join(metrics)}")
    
    asyncio.run(run_evaluation(settings))

async def run_evaluation(settings):
    """Run the full evaluation pipeline."""
    output_dir = Path(settings.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load dataset
        task = progress.add_task("Loading dataset...", total=None)
        dataset = load_dataset(settings.dataset)
        progress.update(task, description=f"Loaded {len(dataset)} samples")
        
        # Initialize components
        progress.update(task, description="Initializing retriever...")
        retriever = get_retriever(settings.retrieval)
        
        progress.update(task, description="Initializing reranker...")
        reranker = get_reranker(settings.reranking)
        
        progress.update(task, description="Initializing generator...")
        generator = get_generator(settings.generation)
        
        # Run retrieval
        progress.update(task, description="Running retrieval...")
        retrieval_results = await retriever.retrieve(dataset.queries, top_k=settings.retrieval.top_k)
        
        # Run reranking
        progress.update(task, description="Running reranking...")
        reranking_results = await reranker.rerank(dataset.queries, retrieval_results, top_k=settings.reranking.top_k)
        
        # Run generation
        progress.update(task, description="Running generation...")
        generation_results = await generator.generate(dataset.queries, reranking_results)
        
        # Evaluate
        progress.update(task, description="Evaluating results...")
        eval_results = {}
        
        if "recall" in settings.evaluation.metrics or "ndcg" in settings.evaluation.metrics or "mrr" in settings.evaluation.metrics:
            eval_results["retrieval"] = evaluate_retrieval(dataset, retrieval_results, settings.evaluation.metrics)
            eval_results["reranking"] = evaluate_reranking(dataset, reranking_results, settings.evaluation.metrics)
        
        if "faithfulness" in settings.evaluation.metrics or "hallucination_rate" in settings.evaluation.metrics or "answer_relevance" in settings.evaluation.metrics:
            eval_results["generation"] = evaluate_generation(dataset, generation_results, settings.evaluation.metrics)
            eval_results["hallucination"] = evaluate_hallucination(dataset, generation_results)
        
        # Generate report
        progress.update(task, description="Generating report...")
        report_paths = generate_report(eval_results, settings, output_dir)
    
    # Print summary
    console.print("\n[bold green]Evaluation Complete![/bold green]")
    console.print(f"Results saved to: {output_dir}")
    
    table = Table(title="Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for category, results in eval_results.items():
        for metric, value in results.items():
            if isinstance(value, float):
                table.add_row(f"{category}.{metric}", f"{value:.4f}")
            else:
                table.add_row(f"{category}.{metric}", str(value))
    
    console.print(table)
    
    for fmt, path in report_paths.items():
        console.print(f"  {fmt}: {path}")

@cli.command()
@click.argument("results_dir", type=click.Path(exists=True))
@click.option("--format", "-f", multiple=True, default=["html"], type=click.Choice(["json", "html", "markdown"]))
@click.pass_context
def report(ctx, results_dir, format):
    """Generate report from existing results."""
    settings = get_settings()
    settings.output.formats = list(format)
    settings.output.directory = results_dir
    
    # Load results and generate report
    console.print(f"Generating report from {results_dir}")
    # Implementation would load results and call generate_report
    console.print("[yellow]Report generation from existing results not yet implemented[/yellow]")

@cli.command()
@click.option("--dataset", "-d", default="msmarco", help="Dataset to download")
@click.option("--cache-dir", default="./cache/datasets", help="Cache directory")
def download(dataset, cache_dir):
    """Download and cache a dataset."""
    from .datasets import download_dataset
    console.print(f"Downloading {dataset} to {cache_dir}...")
    download_dataset(dataset, cache_dir)
    console.print("[green]Download complete![/green]")

@cli.command()
@click.option("--retriever1", "-r1", default="bge-large-en-v1.5", help="First retriever")
@click.option("--retriever2", "-r2", default="bge-base-en-v1.5", help="Second retriever")
@click.option("--dataset", "-d", default="msmarco", help="Dataset")
@click.option("--sample-size", "-s", default=500, help="Sample size")
def compare(retriever1, retriever2, dataset, sample_size):
    """Compare two retrievers side by side."""
    console.print(f"Comparing {retriever1} vs {retriever2} on {dataset}")
    console.print("[yellow]Comparison mode not yet implemented[/yellow]")

if __name__ == "__main__":
    cli()