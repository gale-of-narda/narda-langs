import json
import shlex
import typer
import logging

from typing import Annotated

from rich.console import Console
from rich.logging import RichHandler

from scripts.parser_procedure import Processor, StructureError


class ExitException(Exception):
    pass


cli = typer.Typer(add_completion=False)
console = Console(force_terminal=True)
processor: Processor = None

logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, show_path=False)],
)


@cli.command()
def parse(
    text: str,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    level: Annotated[
        int | None,
        typer.Option("--level", "-l", help="Max level to parse (default: all)."),
    ] = None,
):
    old_level = processor.max_level
    if level is not None:
        processor.max_level = level
    try:
        res = processor.process(text, verbose=verbose)
    finally:
        processor.max_level = old_level
    success = f"[bold green]String '{text}' is grammatical[/bold green]"
    fail = f"[bold red]String '{text}' is not grammatical[/bold red]"
    res_string = success if res else fail
    console.print(res_string)
    return


@cli.command()
def reload():
    """Reloads all parameters from the standard destination."""
    processor.loader.reload()
    console.print("[green]Reloaded all parameters.[/green]")
    return


@cli.command()
def load(path: Annotated[str, typer.Argument(help="Directory or parameter file.")]):
    """Loads parameters from a path (a directory, or a single standard file)."""
    try:
        loaded = processor.loader.load(path)
        console.print(f"[green]Loaded {loaded}.[/green]")
    except (StructureError, ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        console.print(f"[bold red]Error:[/] {e}")
    return


@cli.command()
def set(name: str, value: str):
    """Sets a parameter to a value (parsed as JSON when possible)."""
    try:
        processor.loader.set(name, value)
        console.print(f"[green]Set '{name}'.[/green]")
    except (StructureError, ValueError, KeyError, TypeError, IndexError) as e:
        console.print(f"[bold red]Error:[/] {e}")
    return


@cli.command()
def reset(name: str):
    """Reloads a single parameter from its standard destination file."""
    try:
        processor.loader.reset(name)
        console.print(f"[green]Reset '{name}'.[/green]")
    except (StructureError, ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        console.print(f"[bold red]Error:[/] {e}")
    return


@cli.command()
def get(name: str):
    """Prints the current value of a parameter."""
    try:
        console.print_json(data=processor.loader.get(name))
    except ValueError as e:
        console.print(f"[bold red]Error:[/] {e}")
    return


@cli.command()
def describe(
    lvl: Annotated[int, typer.Option(help="The level of the tree.")] = 0,
    num: Annotated[int, typer.Option(help="Ordinal number of the tree.")] = 0,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    tree = processor.trees[lvl][num]
    description = processor.interpreter.describe(tree, verbose=verbose, rich=True)
    console.print(description)
    return


@cli.command()
def gloss(
    lvl: Annotated[int, typer.Option(help="The level of the tree.")] = 0,
    num: Annotated[int, typer.Option(help="Ordinal number of the tree.")] = 0,
):
    tree = processor.trees[lvl][num]
    gloss_string = processor.interpreter.gloss(tree)
    console.print(gloss_string)
    return


@cli.command()
def draw(
    lvl: Annotated[int, typer.Option(help="The level of the tree.")] = 0,
    num: Annotated[int, typer.Option(help="Ordinal number of the tree.")] = 0,
):
    tree = processor.trees[lvl][num]
    tree_string = processor.interpreter.draw_tree(tree)
    console.print(tree_string)
    return


@cli.command()
def exit():
    raise ExitException


def main(path: str = ""):
    global processor
    try:
        processor = Processor(path=path)
    except Exception as e:
        console.print(f"[bold red]Error on load:[/]\n{e}")
        return

    console.rule("[bold green]Ready")

    while True:
        try:
            command = console.input("\n[bold blue]>[/] ")
            if not command.strip():
                continue
            cli(shlex.split(command), standalone_mode=False)
        except ExitException:
            break
        except Exception as e:
            console.print(f"[red]Unexpected error:[/red] {e}")


if __name__ == "__main__":
    typer.run(main)
