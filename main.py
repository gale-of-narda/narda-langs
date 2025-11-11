import shlex
import typer
import logging

from typing import Annotated

from rich.console import Console
from rich.logging import RichHandler

from scripts.parser_procedure import Processor


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
def parse(text: str, verbose: bool = typer.Option(False, "--verbose", "-v")):
    res = processor.process(text, verbose=verbose)
    success = f"[bold green]String '{text}' is grammatical[/bold green]"
    fail = f"[bold red]String '{text}' is not grammatical[/bold red]"
    res_string = success if res else fail
    console.print(res_string)
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
