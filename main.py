import typer
import logging

from rich.console import Console
from rich.logging import RichHandler

from scripts.parser_procedure import Processor


class ReallyExit(Exception):
    pass


app = typer.Typer(add_completion=False)
console = Console()
processor: Processor = None

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, show_path=False)],
)


@app.command()
def process(text: str, verbose: bool = typer.Option(False, "--verbose", "-v")):
    res = processor.process(text, verbose=verbose)
    success = "[bold green]Successfully parsed[/bold green]"
    fail = "[bold red]Parsing failed for:[/bold red]"
    res_string = success if res else fail
    console.print(f"{res_string} {text}")
    return


@app.command()
def describe(verbose: bool = typer.Option(False, "--verbose", "-v")):
    processor.interpreter.describe(verbose=verbose)
    return


@app.command()
def gloss():
    gloss_text = processor.interpreter.gloss()
    console.print(gloss_text)
    return


@app.command()
def exit():
    raise ReallyExit


def main(path: str = ""):
    global processor

    with console.status("[bold green]Initializing..."):
        try:
            processor = Processor(path=path)
        except Exception as e:
            console.print(f"[bold red]Error loading config:[/]\n{e}")
            return

    console.rule("[bold cyan]Ready")

    while True:
        try:
            command = console.input("\n[bold cyan]>[/] ")
            if not command.strip():
                continue
            app(command.split(" "), standalone_mode=False)
        except ReallyExit:
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


if __name__ == "__main__":
    typer.run(main)
