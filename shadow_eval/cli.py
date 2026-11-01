"""`shadow` command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .config import load_domain
from .store import Store

app = typer.Typer(add_completion=False, help="Shadow-mode readiness evaluator for AI agents.", no_args_is_help=True)

DomainOpt = typer.Option("domains/insurance_claims.yaml", "--domain", "-d", help="Domain config YAML.")
DbOpt = typer.Option("shadow.db", "--db", help="SQLite database file.")


def _progress(label: str):
    def cb(done: int, total: int) -> None:
        if total and (done == total or done % max(1, total // 20) == 0):
            typer.echo(f"  {label}: {done}/{total}")

    return cb


@app.command()
def demo(
    db: str = typer.Option("demo.db", "--db"),
    report: str = typer.Option("demo_report.html", "--report"),
    serve: bool = typer.Option(False, "--serve", help="Start the web app afterwards."),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Build the offline demo (synthetic claims, simulated agent and expert). Free, no API key."""
    from .demo.build import DEFAULT_DOMAIN, build_demo
    from .report import write_report
    from .score import scorecard

    domain = load_domain(DEFAULT_DOMAIN)
    typer.echo("Building demo: 600 synthetic claims, simulated agent, simulated expert review…")
    store = build_demo(db, domain)
    sc = scorecard(store, domain)
    write_report(sc, report)
    m = sc.overall.metrics
    typer.echo(f"Readiness: {sc.readiness.value}  ·  accuracy {m.accuracy:.1%}  ·  {m.critical} critical errors  ·  {m.human_errors} team mistakes caught")
    typer.echo(f"Report written to {report}")
    if serve:
        _serve(store, domain, port, banner="Demo data: synthetic claims, simulated agent and expert.")


@app.command()
def generate(out: str = typer.Option("data/demo", "--out"), n: int = typer.Option(600, "--n"), seed: int = typer.Option(7, "--seed")) -> None:
    """Generate the synthetic insurance-claims dataset (cases, human decisions, ground truth)."""
    from .demo.generate import generate as gen, write_dataset

    data = gen(n=n, seed=seed)
    write_dataset(data, out)
    seeded = sum(1 for r in data["truth"] if r["seeded_human_error"])
    typer.echo(f"Wrote {n} cases to {out} ({seeded} seeded human errors)")


@app.command()
def ingest(
    cases: Optional[Path] = typer.Option(None, "--cases", help="Cases file (.json, .jsonl or .csv)."),
    human: Optional[Path] = typer.Option(None, "--human", help="Human decisions file."),
    domain: str = DomainOpt,
    db: str = DbOpt,
) -> None:
    """Load historical cases and/or human decisions."""
    from .capture.batch import load_cases, load_human

    dom, store = load_domain(domain), Store(db)
    if cases:
        typer.echo(f"Loaded {load_cases(store, dom, cases)} cases")
    if human:
        typer.echo(f"Loaded {load_human(store, dom, human)} human decisions")


@app.command()
def run(
    agent: str = typer.Option("ollama", "--agent", "-a", help="claude | ollama | hf | simulated"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name for the provider."),
    cases: Optional[Path] = typer.Option(None, "--cases"),
    human: Optional[Path] = typer.Option(None, "--human"),
    workers: int = typer.Option(4, "--workers", "-w"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Only run this many new cases."),
    retry_errors: bool = typer.Option(False, "--retry-errors"),
    domain: str = DomainOpt,
    db: str = DbOpt,
) -> None:
    """Batch replay: run the agent in shadow mode on every case without an agent decision."""
    from .agents import make_agent
    from .capture.batch import load_cases, load_human, run_shadow

    dom, store = load_domain(domain), Store(db)
    if cases:
        typer.echo(f"Loaded {load_cases(store, dom, cases)} cases")
    if human:
        typer.echo(f"Loaded {load_human(store, dom, human)} human decisions")
    a = make_agent(agent, dom, model)
    typer.echo(f"Running {a.name} in shadow mode…")
    stats = run_shadow(store, a, workers=workers, limit=limit, retry_errors=retry_errors, on_progress=_progress("decided"))
    typer.echo(f"Done: {stats['ran']} decided, {stats['errors']} failed, {stats['skipped']} already had a decision")


@app.command()
def suggest(
    adjudicator: str = typer.Option("ollama", "--adjudicator", help="claude | ollama | hf | rules (demo domain only)"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    workers: int = typer.Option(4, "--workers", "-w"),
    redo: bool = typer.Option(False, "--redo", help="Re-suggest items that already have a suggestion."),
    domain: str = DomainOpt,
    db: str = DbOpt,
) -> None:
    """Ask an adjudicator to suggest who was right for every item in the review queue."""
    from .adjudicate import LLMAdjudicator, suggest_all
    from .llm import make_backend

    dom, store = load_domain(domain), Store(db)
    if adjudicator == "rules":
        from .demo.simulated import PolicyRulesAdjudicator

        adj = PolicyRulesAdjudicator(dom)
    else:
        kwargs = {"effort": dom.agent.claude_effort} if adjudicator == "claude" else {}
        default = {"claude": dom.agent.claude_model, "ollama": dom.agent.ollama_model}.get(adjudicator)
        adj = LLMAdjudicator(dom, make_backend(adjudicator, model or default, **kwargs))
    n = suggest_all(store, dom, adj, workers=workers, redo=redo, on_progress=_progress("suggested"))
    typer.echo(f"{n} suggestions written. Review them with: shadow serve")


@app.command()
def score(domain: str = DomainOpt, db: str = DbOpt) -> None:
    """Print the readiness scorecard."""
    from .score import scorecard

    dom, store = load_domain(domain), Store(db)
    sc = scorecard(store, dom)
    m = sc.overall.metrics
    typer.echo(f"\nOVERALL: {sc.readiness.value}{'  (provisional)' if sc.provisional else ''}")
    typer.echo(
        f"  accuracy {m.accuracy:.1%} [CI {m.acc_low:.1%}–{m.acc_high:.1%}] · critical {m.critical} ({m.critical_rate:.1%}) · "
        f"cost/100 {m.cost_per_100:.0f} · disagreements {m.disagreements} ({m.unreviewed} unreviewed)"
    )
    for dim, segs in sc.segments.items():
        typer.echo(f"\n  {dim}")
        for s in segs:
            g = f"  guardrail: conf ≥ {s.guardrail.threshold:.0%} covers {s.guardrail.coverage:.0%}" if s.guardrail else ""
            typer.echo(f"    {s.value:<12} n={s.metrics.n:<4} acc={s.metrics.accuracy:6.1%}  {s.readiness.value}{g}")
    typer.echo("")


@app.command()
def report(out: str = typer.Option("readiness_report.html", "--out", "-o"), domain: str = DomainOpt, db: str = DbOpt) -> None:
    """Write the executive readiness report (HTML)."""
    from .report import write_report
    from .score import scorecard

    dom, store = load_domain(domain), Store(db)
    path = write_report(scorecard(store, dom), out)
    typer.echo(f"Report written to {path}")


def _serve(store: Store, dom, port: int, agent=None, adjudicator=None, banner: str | None = None, host: str = "127.0.0.1") -> None:
    import uvicorn

    from .web.app import create_app

    typer.echo(f"Open http://{host}:{port}")
    uvicorn.run(create_app(store, dom, agent=agent, adjudicator=adjudicator, banner=banner), host=host, port=port, log_level="warning")


@app.command()
def serve(
    port: int = typer.Option(8000, "--port"),
    host: str = typer.Option("127.0.0.1", "--host"),
    agent: Optional[str] = typer.Option(None, "--agent", help="Agent for live capture via the API: claude | ollama | hf | simulated"),
    model: Optional[str] = typer.Option(None, "--model"),
    banner: Optional[str] = typer.Option(None, "--banner", help="Notice shown at the top of every page."),
    domain: str = DomainOpt,
    db: str = DbOpt,
) -> None:
    """Start the web app: review queue, dashboard, report and live-capture API."""
    from .agents import make_agent

    dom, store = load_domain(domain), Store(db)
    _serve(store, dom, port, agent=make_agent(agent, dom, model) if agent else None, banner=banner, host=host)


@app.command("verify-audit")
def verify_audit(db: str = DbOpt) -> None:
    """Check the hash-chained audit log for edits or deletions."""
    r = Store(db).verify_audit()
    if r.ok:
        typer.echo(f"OK: {r.events} audit events, chain intact")
    else:
        typer.echo(f"TAMPERED at event {r.broken_at}: {r.reason}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
