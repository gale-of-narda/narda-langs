import typer
import logging

from rich.console import Console
from rich.logging import RichHandler

from scripts.parser_procedure import Processor


class ExitException(Exception):
    pass


cli = typer.Typer(add_completion=False)
console = Console(force_terminal=True)
processor: Processor = None

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, show_path=False)],
)


@cli.command()
def process(text: str, verbose: bool = typer.Option(False, "--verbose", "-v")):
    res = processor.process(text, verbose=verbose)
    success = f"[bold green]Input string {text} is well-formed[/bold green]"
    fail = f"[bold red]Input string {text} is not well-formed[/bold red]"
    res_string = success if res else fail
    console.print(res_string)
    return


@cli.command()
def describe(verbose: bool = typer.Option(False, "--verbose", "-v")):
    processor.interpreter.describe(verbose=verbose)
    return


@cli.command()
def gloss():
    gloss_text = processor.interpreter.gloss()
    console.print(gloss_text)
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
            cli(command.split(), standalone_mode=False)
        except ExitException:
            break
        except Exception as e:
            console.print(f"[red]Unexpected error:[/red] {e}")


if __name__ == "__main__":
    typer.run(main)
