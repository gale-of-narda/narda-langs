"""Rendering helpers that turn already-computed parser data into Rich
renderables for console output. Kept free of any parsing logic.
"""

from rich import box
from rich.table import Table

from scripts.parser_dataclasses import ParsingResult, Type
from scripts.parser_entities import Node, Tree


def type_table(tree: Tree, types: list[Type], verbose: bool = False) -> Table:
    """Builds a table of the tree's composition and permutation types, titled
    with the word. Type names are looked up in the given type list to supply
    the (optional) description column.
    """
    table = Table(
        title=f"[dim]'{tree.working_string}'[/dim]",
        title_justify="left",
        box=box.SIMPLE_HEAD,
        highlight=True,
        title_style="bold",
    )
    table.add_column("Type", style="cyan", no_wrap=True)
    table.add_column("Argument", style="yellow")
    if verbose:
        table.add_column("Description", style="dim")
    for category, name in (
        ("Composition", tree.ctype),
        ("Permutation", tree.ptype),
    ):
        record = next(
            (t for t in types if t.type == category and t.argument_name == name),
            None,
        )
        row = [category, name or "Undefined"]
        if verbose:
            row.append(record.argument_description if record else "")
        table.add_row(*row)
    return table


def feature_table(
    featured: list[Node], featureless: list[Node], verbose: bool = False
) -> Table:
    """Builds a table of the interpreted features of a tree's nodes. Nodes
    without an interpretation are listed in the caption.
    """
    table = Table(
        title_justify="left",
        box=box.SIMPLE_HEAD,
        highlight=True,
        title_style="bold",
    )
    table.add_column("", style="cyan", no_wrap=True)
    table.add_column("Function", style="green")
    table.add_column("Argument", style="yellow")
    if verbose:
        table.add_column("Description", style="dim")
    for node in featured:
        feature = node.feature
        if feature is None:
            continue
        row = [
            str(node.content[0]),
            feature.function_name,
            feature.argument_name,
        ]
        if verbose:
            row.append(feature.argument_description)
        table.add_row(*row)
    if featureless:
        table.caption = (
            "[dim]No interpretation: "
            + ", ".join(str(n) for n in featureless)
            + "[/dim]"
        )
    return table


def result_table(result: ParsingResult) -> Table:
    """Builds a table of the four parsing criteria and their outcomes, each
    shown as ✓ (met), ✗ (unmet), or ⍰ (undetermined).
    """
    table = Table(
        box=box.SIMPLE_HEAD,
        highlight=True,
        title_style="bold",
    )
    table.add_column("Criterion", style="cyan", no_wrap=True)
    table.add_column("Result", justify="center")
    glyph = {
        True: "[green]✓[/green]",
        False: "[red]✗[/red]",
        None: "[dim]⍰[/dim]",
    }
    for name, value in (
        ("Intelligibility", result.intelligibility),
        ("Grammaticality", result.grammaticality),
        ("Interpretability", result.interpretability),
        ("Felicity", result.felicity),
    ):
        table.add_row(name, glyph[value])
    return table


def gloss_table(tokens: list[tuple[str, str]]) -> Table:
    """Wraps a list of (form, gloss) pairs into a table."""
    table = Table(
        title_justify="left",
        box=box.SIMPLE_HEAD,
        show_header=False,
        highlight=False,
        title_style="bold",
        padding=(0, 1),
    )
    for _ in tokens:
        table.add_column(no_wrap=True)
    table.add_row(*[f"[italic]{f}[/italic]" for f, _ in tokens])
    table.add_row(*[f"[bold cyan]{g}[/bold cyan]" for _, g in tokens])
    return table
